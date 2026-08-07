"""Built-in content-specific codecs on the Entroly codec contract.

A codec offers candidate representations and reports distortion; it never
declares task sufficiency. Every lossy representation carries a reference to the
complete original source, stored through Entroly's hardened recovery subsystem.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Callable

from .codec import (
    Representation,
    RecoveryStore,
    SupportDecision,
    content_digest,
    estimate_tokens,
)

_JSON_MAX_SCAN_NODES = 4096
_JSON_MAX_PROTECTED_VALUES = 256
_JSON_MAX_PROTECTED_BYTES = 16_384


class _ProtectionOverflow(ValueError):
    """The bounded evidence scan could not prove a lossy form safe."""


def _full_representation(
    *,
    text: str,
    source_id: str,
    content_type: str,
    codec: str,
    version: str,
) -> Representation:
    return Representation(
        representation_id=f"{source_id}#{codec}.full",
        source_id=source_id,
        content_type=content_type,
        text=text,
        token_cost=estimate_tokens(text),
        codec=codec,
        codec_version=version,
        source_sha256=content_digest(text),
        distortion_risk=0.0,
    )


def _unique(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))



# Nesting beyond this is declined without parsing. Real payloads are shallow --
# JSON APIs rarely exceed ~20 levels -- while a nesting bomb is thousands deep.
# Catching RecursionError was not enough: where it fires depends on the
# platform's stack, so the guard passed on Windows and the same input still
# recursed on Linux CI. Counting brackets is deterministic everywhere.
_JSON_MAX_NESTING = 200


def _exceeds_nesting_limit(text: str, limit: int = _JSON_MAX_NESTING) -> bool:
    """True when bracket nesting exceeds `limit`, ignoring brackets in strings."""
    depth = 0
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
            if depth > limit:
                return True
        elif ch in "]}":
            depth -= 1
    return False


class JsonCodec:
    name = "json"
    version = "4"

    def __init__(self, store: RecoveryStore | None = None) -> None:
        self.store = store if store is not None else RecoveryStore()
        # One-slot parse cache. supports() must parse to answer honestly --
        # "opens like JSON" is not the same as "is JSON" -- and the registry
        # then calls representations() on the same instance with the same text,
        # so parsing twice was pure waste. Measured at 1MB: routing cost
        # 106.8 ms p50 against 44.0 ms to do the actual work.
        #
        # Keyed by the exact string, so a different payload never reuses it.
        self._parsed_key: str | None = None
        self._parsed_value: Any = None

    def _parse(self, stripped: str) -> Any:
        """Parse once per distinct payload; raises as json.loads would."""
        if self._parsed_key is not None and self._parsed_key == stripped:
            return self._parsed_value
        value = json.loads(stripped)
        self._parsed_key = stripped
        self._parsed_value = value
        return value

    def supports(self, text: str, content_type: str = "") -> SupportDecision:
        if content_type == "json":
            return SupportDecision(True, 1.0, "declared content type")
        stripped = text.strip()
        if not stripped.startswith(("{", "[")):
            return SupportDecision(False, 0.0, "does not open as a JSON value")
        if _exceeds_nesting_limit(stripped):
            return SupportDecision(
                False, 0.0, f"nesting deeper than {_JSON_MAX_NESTING} levels"
            )
        try:
            self._parse(stripped)
        except ValueError:
            return SupportDecision(False, 0.0, "opens like JSON but does not parse")
        except RecursionError:
            # A nesting bomb -- thousands of levels of "[" exhaust the parser's
            # stack. Declining hands the original back untouched, which is the
            # safe outcome; attempting it takes the process down.
            return SupportDecision(False, 0.0, "nesting too deep to parse safely")
        return SupportDecision(True, 0.9, "parses as JSON")

    def representations(
        self, text: str, source_id: str = "", **options: Any
    ) -> list[Representation]:
        from .universal_compress import _is_load_bearing_key, _json_to_schema

        full = _full_representation(
            text=text,
            source_id=source_id,
            content_type="json",
            codec=self.name,
            version=self.version,
        )
        reps = [full]
        try:
            data = self._parse(text.strip())
        except (ValueError, RecursionError):
            # Unparseable or too deeply nested: offer only the verbatim
            # representation rather than a partial rewrite of content this
            # codec could not read.
            return reps

        # Columnar form is computed before anything schema-related, because two
        # separate early exits below would otherwise skip it on exactly the
        # payloads it exists to serve:
        #
        #   * `len(elided) >= len(text)` -- the schema form is *larger* than the
        #     original for identifier-dense records, so the codec returned
        #     nothing at all;
        #   * `_ProtectionOverflow` -- enumerating every protected value
        #     exceeds its budget once a payload carries many identifiers, and
        #     overflow declines every lossy form.
        #
        # Measured on 200 records: {"id","name"} compressed while {"id","sku"}
        # produced 0%, the only difference being a key name that pushed it past
        # one of these limits. The columnar form depends on neither: it keeps
        # whole load-bearing *columns* verbatim, checked structurally by
        # _columnar_preserves_columns. Column identity is a stronger statement
        # than "each of these substrings appears somewhere", and is O(columns)
        # rather than O(values).
        # what happens to identifier-dense records, the payloads most worth
        # compressing. Measured on 200 records: {"id","name"} compressed while
        # {"id","sku"} produced nothing, because the second overflowed.
        #
        # The columnar form does not need the enumeration. It keeps whole
        # load-bearing *columns* verbatim, so preservation is structural and is
        # verified as such by _columnar_preserves_columns. Column identity is a
        # stronger statement than "each of these 400 substrings appears
        # somewhere", and it costs nothing to check.
        columnar = _columnar_json(data, _is_load_bearing_key)
        if (
            columnar is not None
            and len(columnar) < len(text)
            and _columnar_preserves_columns(columnar, data, _is_load_bearing_key)
        ):
            reps.append(
                Representation(
                    representation_id=f"{source_id}#json.columnar",
                    source_id=source_id,
                    content_type="json",
                    text=columnar,
                    token_cost=estimate_tokens(columnar),
                    codec=self.name,
                    codec_version=self.version,
                    source_sha256=content_digest(text),
                    # A bounded sample for the receipt. The guarantee is the
                    # structural check above, not this list.
                    protected_evidence=_columnar_evidence_sample(
                        data, _is_load_bearing_key
                    ),
                    distortion_risk=1.0 - (len(columnar) / max(len(text), 1)),
                    recovery=self.store.put(
                        text,
                        item_count=_count_elided_json_records(data),
                        note=(
                            "complete original JSON for "
                            f"{source_id or 'payload'} (columnar form kept "
                            "load-bearing columns verbatim)"
                        ),
                    ),
                )
            )

        elided = json.dumps(
            _json_to_schema(data, depth=0, max_depth=4),
            indent=2,
            ensure_ascii=True,
        )
        if len(elided) >= len(text):
            return reps

        try:
            protected = _protected_json_values(data, _is_load_bearing_key)
        except _ProtectionOverflow:
            return reps
        if any(value not in elided for value in protected):
            return reps

        omitted_count = _count_elided_json_records(data)
        recovery = self.store.put(
            text,
            item_count=omitted_count,
            note=f"complete original JSON for {source_id or 'payload'}",
        )
        reps.append(
            Representation(
                representation_id=f"{source_id}#json.elided",
                source_id=source_id,
                content_type="json",
                text=elided,
                token_cost=estimate_tokens(elided),
                codec=self.name,
                codec_version=self.version,
                source_sha256=content_digest(text),
                protected_evidence=protected,
                distortion_risk=1.0 - (len(elided) / max(len(text), 1)),
                recovery=recovery,
            )
        )
        return reps


def _protected_json_values(
    obj: Any,
    is_load_bearing_key: Callable[[str], bool],
) -> tuple[str, ...]:
    """Return serialized protected scalars from the complete source.

    Repetitive records remain compressible because only values under explicit
    load-bearing keys are mandatory. The walk covers every list element and is
    bounded; exceeding the bound makes the codec decline the lossy form.
    """

    out: list[str] = []
    stack: list[tuple[Any, bool]] = [(obj, False)]
    visited = 0
    serialized_bytes = 0

    while stack:
        node, inherited = stack.pop()
        visited += 1
        if visited > _JSON_MAX_SCAN_NODES:
            raise _ProtectionOverflow("JSON evidence scan exceeded node budget")

        if isinstance(node, dict):
            for child_key, value in reversed(list(node.items())):
                stack.append((value, inherited or is_load_bearing_key(str(child_key))))
            continue
        if isinstance(node, list):
            stack.extend((value, inherited) for value in reversed(node))
            continue
        if not inherited:
            continue

        rendered = json.dumps(
            node,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        if rendered not in out:
            out.append(rendered)
            serialized_bytes += len(rendered.encode("utf-8"))
            if (
                len(out) > _JSON_MAX_PROTECTED_VALUES
                or serialized_bytes > _JSON_MAX_PROTECTED_BYTES
            ):
                raise _ProtectionOverflow(
                    "JSON protected evidence exceeded bounded storage budget"
                )

    return _unique(out)


#: Ceiling on records considered for the columnar form. Above this the verbatim
#: identifier columns stop being cheaper than the original, and the walk cost
#: stops being negligible on a request path.
_JSON_COLUMNAR_MAX_RECORDS = 5000


def _record_array(data: Any) -> tuple[str | None, list[dict[str, Any]]] | None:
    """Find a flat array of records, either bare or under a single key.

    Real payloads are usually ``[{...}, ...]`` or ``{"data": [{...}, ...]}``,
    so both shapes resolve to the same columnar treatment. Anything else --
    several sibling arrays, nested records, ragged keys -- is left alone rather
    than guessed at.
    """
    if isinstance(data, list):
        rows, key = data, None
    elif isinstance(data, dict):
        arrays = [
            (k, v) for k, v in data.items()
            if isinstance(v, list) and len(v) >= 2
        ]
        if len(arrays) != 1:
            return None
        key, rows = arrays[0][0], arrays[0][1]
    else:
        return None

    if not 2 <= len(rows) <= _JSON_COLUMNAR_MAX_RECORDS:
        return None
    if not all(isinstance(row, dict) for row in rows):
        return None
    # Scalars only: a nested object under a record key has no columnar form
    # that preserves it without recursing, and recursing here would quietly
    # reintroduce the summarisation this representation exists to avoid.
    if any(
        isinstance(value, (dict, list))
        for row in rows
        for value in row.values()
    ):
        return None

    first = set(rows[0])
    if not first or any(set(row) != first for row in rows):
        return None
    return key, rows


def _column_summary(values: list[Any]) -> dict[str, Any]:
    """Describe a column that is not load-bearing, without keeping its values."""
    distinct = {json.dumps(v, sort_keys=True) for v in values}
    if len(distinct) == 1:
        return {"const": values[0]}
    if all(isinstance(v, bool) for v in values):
        true_count = sum(1 for v in values if v)
        return {"type": "bool", "true": true_count, "false": len(values) - true_count}
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return {
            "type": "number",
            "n": len(values),
            "min": min(values),
            "max": max(values),
        }
    return {
        "type": "string",
        "n": len(values),
        "distinct": len(distinct),
        "example": values[0],
    }


def _columnar_json(
    data: Any,
    is_load_bearing_key: Callable[[str], bool],
) -> str | None:
    """Columnar rendering that keeps every load-bearing value verbatim.

    The schema form elides whole records, so a payload carrying identifiers --
    ``sku``, ``order_id``, ``email``, ``price`` -- fails the protected-evidence
    check and the codec correctly declines to compress it at all. Measured on a
    200-record array, ``{"id","name"}`` compressed 53% while ``{"id","sku"}``
    compressed 0%: the single difference was a key name.

    Refusing is the right answer when the only alternative destroys
    identifiers, but it is not the only alternative. A record array spends most
    of its bytes repeating key names and non-identifying fields. Emitting each
    column once, keeping load-bearing columns verbatim and summarising the
    rest, preserves exactly what the guard protects while removing what it does
    not.

    Returns None when the payload has no such shape, when nothing is
    load-bearing (the existing schema form already handles that better), or
    when the result would not be smaller.
    """
    found = _record_array(data)
    if found is None:
        return None
    container_key, rows = found

    keys = list(rows[0])
    kept = [k for k in keys if is_load_bearing_key(str(k))]
    if not kept:
        # Nothing to protect: the schema representation compresses harder.
        return None
    elided = [k for k in keys if k not in kept]

    payload: dict[str, Any] = {
        "_format": "columnar",
        "_records": len(rows),
        "_verbatim_columns": kept,
        "_summarised_columns": elided,
    }
    if container_key is not None:
        payload["_container"] = container_key
    for key in kept:
        payload[str(key)] = [row[key] for row in rows]
    payload["_summaries"] = {
        str(key): _column_summary([row[key] for row in rows]) for key in elided
    }

    return json.dumps(payload, indent=2, ensure_ascii=True)


def _columnar_preserves_columns(
    columnar: str,
    data: Any,
    is_load_bearing_key: Callable[[str], bool],
) -> bool:
    """Verify every load-bearing column survived value-for-value, in order.

    Structural, not substring-based. Re-reading the rendered text and comparing
    the reconstructed column against the source catches a truncated, reordered
    or coerced column, none of which a substring scan would notice -- and it
    does so without enumerating every value, which is what overflows on the
    payloads this form exists to serve.
    """
    found = _record_array(data)
    if found is None:
        return False
    _, rows = found

    try:
        rendered = json.loads(columnar)
    except ValueError:
        return False

    for key in rows[0]:
        if not is_load_bearing_key(str(key)):
            continue
        if rendered.get(str(key)) != [row[key] for row in rows]:
            return False
    return True


def _columnar_evidence_sample(
    data: Any,
    is_load_bearing_key: Callable[[str], bool],
    limit: int = 8,
) -> tuple[str, ...]:
    """A bounded, serialized sample of preserved values, for the receipt.

    Deliberately small. The preservation guarantee is
    :func:`_columnar_preserves_columns`; this exists so a reader can spot-check
    it without the receipt carrying thousands of identifiers.
    """
    found = _record_array(data)
    if found is None:
        return ()
    _, rows = found

    out: list[str] = []
    for key in rows[0]:
        if not is_load_bearing_key(str(key)):
            continue
        for row in rows[: max(1, limit // 2)]:
            out.append(json.dumps(row[key], sort_keys=True))
            if len(out) >= limit:
                return tuple(out)
    return tuple(out)


def _count_elided_json_records(obj: Any) -> int:
    count = 0

    def walk(node: Any, depth: int) -> None:
        nonlocal count
        if depth > 16:
            return
        if isinstance(node, dict):
            for value in node.values():
                walk(value, depth + 1)
        elif isinstance(node, list):
            count += max(0, len(node) - 1)
            if node:
                walk(node[0], depth + 1)

    walk(obj, 0)
    return count


class LogCodec:
    name = "log"
    version = "4"
    _LOG_SHAPE = re.compile(
        r"^\d{4}[-/]\d{2}|^\[?\d{2}:\d{2}|^(DEBUG|INFO|WARN|ERROR|TRACE|FATAL)",
        re.MULTILINE,
    )
    _CRITICAL = re.compile(
        r"\b(ERROR|FATAL|Traceback|Exception|panic|exit_code|exit status|"
        r"status(?:_code)?\s*[:=]\s*[45]\d\d)\b",
        re.IGNORECASE,
    )

    def __init__(self, store: RecoveryStore | None = None) -> None:
        self.store = store if store is not None else RecoveryStore()

    def supports(self, text: str, content_type: str = "") -> SupportDecision:
        if content_type == "log":
            return SupportDecision(True, 1.0, "declared content type")
        if self._LOG_SHAPE.search(text[:2000]):
            return SupportDecision(True, 0.8, "timestamped or levelled lines")
        return SupportDecision(False, 0.0, "no log line shape found")

    def representations(
        self, text: str, source_id: str = "", **options: Any
    ) -> list[Representation]:
        from .universal_compress import (
            _compress_log_universal,
            _log_template,
            _strip_log_prefix,
        )

        full = _full_representation(
            text=text,
            source_id=source_id,
            content_type="log",
            codec=self.name,
            version=self.version,
        )
        reps = [full]
        collapsed = _compress_log_universal(text)
        if not collapsed or len(collapsed) >= len(text):
            return reps

        protected = _protected_log_lines(
            text,
            _log_template,
            _strip_log_prefix,
            self._CRITICAL,
        )
        if any(value not in collapsed for value in protected):
            return reps

        recovery = self.store.put(
            text,
            item_count=_log_omitted_count(text, _log_template, _strip_log_prefix),
            note=f"complete original log for {source_id or 'log'}",
        )
        reps.append(
            Representation(
                representation_id=f"{source_id}#log.collapsed",
                source_id=source_id,
                content_type="log",
                text=collapsed,
                token_cost=estimate_tokens(collapsed),
                codec=self.name,
                codec_version=self.version,
                source_sha256=content_digest(text),
                protected_evidence=protected,
                distortion_risk=1.0 - (len(collapsed) / max(len(text), 1)),
                recovery=recovery,
            )
        )
        return reps


def _protected_log_lines(
    text: str,
    template: Callable[[str], str],
    strip_prefix: Callable[[str], str],
    critical: re.Pattern[str],
) -> tuple[str, ...]:
    seen: set[str] = set()
    protected: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        key = template(strip_prefix(stripped))
        if key in seen:
            continue
        seen.add(key)
        if critical.search(stripped):
            protected.append(stripped)
    return _unique(protected)


def _log_omitted_count(
    text: str,
    template: Callable[[str], str],
    strip_prefix: Callable[[str], str],
) -> int:
    counts: Counter[str] = Counter()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            counts[template(strip_prefix(stripped))] += 1
    return sum(max(0, count - 1) for count in counts.values())


class ShellCodec:
    name = "shell"
    version = "2"
    _SHELL_SHAPE = re.compile(
        r"^\s*[$>]\s+\S|^(?:PASS|FAIL|ok|error|warning|Error|Warning)\b"
        r"|\b(?:exit(?:ed)? (?:code|status)|Traceback|npm ERR!|error\[E\d+\])"
        r"|^\s*\d+ (?:passed|failed|error)",
        re.MULTILINE,
    )
    _OUTCOME = re.compile(
        r"exit(?:ed)?[ _](?:code|status)[ =:]*\d+"
        r"|\b\d+ (?:passed|failed|errors?|warnings?)\b"
        r"|error\[E\d+\]|npm ERR!",
        re.IGNORECASE,
    )
    _FAILURE_LINE = re.compile(
        r"\b(FAILED|ERROR|FATAL|Traceback|AssertionError|Exception|panic)\b",
        re.IGNORECASE,
    )

    def __init__(self, store: RecoveryStore | None = None) -> None:
        self.store = store if store is not None else RecoveryStore()

    def supports(self, text: str, content_type: str = "") -> SupportDecision:
        if content_type in {"shell", "tool_output"}:
            return SupportDecision(True, 1.0, "declared content type")
        if self._SHELL_SHAPE.search(text[:4000]):
            return SupportDecision(True, 0.7, "prompt, status or tool-error shape")
        return SupportDecision(False, 0.0, "no shell-output shape found")

    def representations(
        self, text: str, source_id: str = "", **options: Any
    ) -> list[Representation]:
        from .shell_codec import esc_compress

        full = _full_representation(
            text=text,
            source_id=source_id,
            content_type="shell",
            codec=self.name,
            version=self.version,
        )
        reps = [full]
        budget = int(options.get("budget", 1000))
        try:
            compressed = esc_compress(text, budget=budget).compressed
        except Exception:
            return reps
        if not compressed or len(compressed) >= len(text):
            return reps

        protected = self._protected_from_source(text)
        if any(value not in compressed for value in protected):
            return reps

        original_nonempty = sum(1 for line in text.splitlines() if line.strip())
        compressed_nonempty = sum(1 for line in compressed.splitlines() if line.strip())
        recovery = self.store.put(
            text,
            item_count=max(0, original_nonempty - compressed_nonempty),
            note=f"complete original shell output for {source_id or 'command'}",
        )
        reps.append(
            Representation(
                representation_id=f"{source_id}#shell.esc",
                source_id=source_id,
                content_type="shell",
                text=compressed,
                token_cost=estimate_tokens(compressed),
                codec=self.name,
                codec_version=self.version,
                source_sha256=content_digest(text),
                protected_evidence=protected,
                distortion_risk=1.0 - (len(compressed) / max(len(text), 1)),
                recovery=recovery,
            )
        )
        return reps

    def _protected_from_source(self, text: str) -> tuple[str, ...]:
        protected: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if not protected and re.match(r"^[$#>]\s+\S", stripped):
                protected.append(stripped)
            if self._FAILURE_LINE.search(stripped):
                protected.append(stripped)
            protected.extend(match.group(0) for match in self._OUTCOME.finditer(stripped))
        return _unique(protected)


def default_registry(store: RecoveryStore | None = None):
    from .codec import CodecRegistry

    shared = store if store is not None else RecoveryStore()
    registry = CodecRegistry()
    from .codecs_content import (
        CodeCodec,
        ConversationCodec,
        DocumentCodec,
        SchemaCodec,
    )
    from .codecs_table import TableCodec

    # Order does not decide the winner -- `select` takes the highest support
    # confidence -- but SchemaCodec deliberately outbids JsonCodec (0.95 vs
    # 0.90) on payloads that carry schema markers, because a schema compressed
    # as generic JSON loses its contract.
    registry.register(JsonCodec(shared))
    registry.register(LogCodec(shared))
    registry.register(ShellCodec(shared))
    registry.register(SchemaCodec(shared))
    registry.register(CodeCodec(shared))
    registry.register(DocumentCodec(shared))
    registry.register(ConversationCodec(shared))
    registry.register(TableCodec(shared))
    return registry
