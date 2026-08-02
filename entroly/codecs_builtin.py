"""Built-in codecs on the `entroly.codec` contract.

These wrap the compression that already existed in ``universal_compress`` and
``shell_codec`` rather than reimplementing it. What they add is the part the
contract requires and the originals could not provide: provenance, an explicit
statement of what was protected, and a recovery reference for what was dropped.

Both codecs previously elided real content and reported only a count --
``"... (40 items)"``, ``"[x200]"`` -- with no way to get the originals back.
Now the omitted content goes to a content-addressed store and the reference
travels with the representation.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .codec import (
    Representation,
    RecoveryStore,
    SupportDecision,
    content_digest,
    estimate_tokens,
)


class JsonCodec:
    """JSON / structured payloads.

    Offers two representations: the payload verbatim, and a schema-elided form
    that keeps values of record (identifiers, codes, amounts, timestamps) while
    collapsing repeated records to one exemplar. The elided records are stored
    for exact recovery.
    """

    name = "json"
    version = "2"

    def __init__(self, store: RecoveryStore | None = None) -> None:
        # `store or RecoveryStore()` silently discarded the caller's store:
        # RecoveryStore defines __len__, so an EMPTY one is falsy, and every
        # codec built with a fresh shared store got its own instead. The
        # reference then pointed into a store the caller could not read.
        self.store = store if store is not None else RecoveryStore()

    def supports(self, text: str, content_type: str = "") -> SupportDecision:
        if content_type == "json":
            return SupportDecision(True, 1.0, "declared content type")
        stripped = text.strip()
        if not stripped.startswith(("{", "[")):
            return SupportDecision(False, 0.0, "does not open as a JSON value")
        try:
            json.loads(stripped)
        except ValueError:
            # Malformed JSON must not be rewritten by a codec that assumed it
            # could parse it. Decline and let the caller keep the original.
            return SupportDecision(False, 0.0, "opens like JSON but does not parse")
        return SupportDecision(True, 0.9, "parses as JSON")

    def representations(
        self, text: str, source_id: str = "", **options: Any
    ) -> list[Representation]:
        from .universal_compress import _json_to_schema

        src_digest = content_digest(text)
        reps = [
            Representation(
                representation_id=f"{source_id}#json.full",
                source_id=source_id,
                content_type="json",
                text=text,
                token_cost=estimate_tokens(text),
                codec=self.name,
                codec_version=self.version,
                source_sha256=src_digest,
                distortion_risk=0.0,
            )
        ]

        try:
            data = json.loads(text)
        except ValueError:
            return reps

        elided = json.dumps(_json_to_schema(data, depth=0, max_depth=4), indent=2)
        if len(elided) >= len(text):
            # Anti-inflation: an "elided" form that is not smaller is not an
            # option, it is a worse copy.
            return reps

        omitted, count = _omitted_json_records(data)
        recovery = None
        if omitted:
            recovery = self.store.put(
                omitted,
                item_count=count,
                note=f"records elided from {source_id or 'json payload'}",
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
                source_sha256=src_digest,
                protected_evidence=_scalar_values(data),
                distortion_risk=1.0 - (len(elided) / max(len(text), 1)),
                recovery=recovery,
            )
        )
        return reps


def _scalar_values(obj: Any, out: list[str] | None = None, depth: int = 0) -> tuple[str, ...]:
    """Scalars a caller should be able to find in the elided form.

    Top-level and near-top-level only: values inside a collapsed array are
    intentionally not claimed, because only the exemplar survives.
    """
    out = [] if out is None else out
    if depth > 2:
        return tuple(out)
    if isinstance(obj, dict):
        for value in obj.values():
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                text = str(value)
                if text and len(text) <= 64:
                    out.append(text)
            elif isinstance(value, dict):
                _scalar_values(value, out, depth + 1)
    return tuple(dict.fromkeys(out))


def _omitted_json_records(obj: Any, depth: int = 0) -> tuple[str, int]:
    """Serialise the array records the schema form drops (all but the first)."""
    dropped: list[Any] = []

    def walk(node: Any, d: int) -> None:
        if d > 4:
            return
        if isinstance(node, dict):
            for value in node.values():
                walk(value, d + 1)
        elif isinstance(node, list) and len(node) > 1:
            dropped.extend(node[1:])
            walk(node[0], d + 1)

    walk(obj, depth)
    if not dropped:
        return "", 0
    return json.dumps(dropped, indent=2), len(dropped)


class LogCodec:
    """Log output: collapse repeated events, keep the first of each verbatim."""

    name = "log"
    version = "2"
    _LOG_SHAPE = re.compile(
        r"^\d{4}[-/]\d{2}|^\[?\d{2}:\d{2}|^(DEBUG|INFO|WARN|ERROR|TRACE|FATAL)",
        re.MULTILINE,
    )

    def __init__(self, store: RecoveryStore | None = None) -> None:
        # `store or RecoveryStore()` silently discarded the caller's store:
        # RecoveryStore defines __len__, so an EMPTY one is falsy, and every
        # codec built with a fresh shared store got its own instead. The
        # reference then pointed into a store the caller could not read.
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
        from .universal_compress import _compress_log_universal, _log_template

        src_digest = content_digest(text)
        reps = [
            Representation(
                representation_id=f"{source_id}#log.full",
                source_id=source_id,
                content_type="log",
                text=text,
                token_cost=estimate_tokens(text),
                codec=self.name,
                codec_version=self.version,
                source_sha256=src_digest,
                distortion_risk=0.0,
            )
        ]

        collapsed = _compress_log_universal(text)
        if len(collapsed) >= len(text):
            return reps

        # Everything after the first occurrence of each event template is what
        # the collapsed form no longer contains.
        seen: set[str] = set()
        omitted_lines: list[str] = []
        first_error = ""
        ts_strip = re.compile(r"^\S+\s+\S+\s+")
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            key = _log_template(ts_strip.sub("", stripped))[:100]
            if key in seen:
                omitted_lines.append(line)
            else:
                seen.add(key)
                if not first_error and re.search(r"\b(ERROR|FATAL|Traceback)\b", line):
                    first_error = stripped

        recovery = None
        if omitted_lines:
            recovery = self.store.put(
                "\n".join(omitted_lines),
                item_count=len(omitted_lines),
                note=f"repeat occurrences collapsed from {source_id or 'log'}",
            )

        protected = tuple(x for x in (first_error,) if x)
        reps.append(
            Representation(
                representation_id=f"{source_id}#log.collapsed",
                source_id=source_id,
                content_type="log",
                text=collapsed,
                token_cost=estimate_tokens(collapsed),
                codec=self.name,
                codec_version=self.version,
                source_sha256=src_digest,
                protected_evidence=protected,
                distortion_risk=1.0 - (len(collapsed) / max(len(text), 1)),
                recovery=recovery,
            )
        )
        return reps


def default_registry(store: RecoveryStore | None = None):
    """Registry with the built-in codecs. Unknown content selects nothing."""
    from .codec import CodecRegistry

    shared = store if store is not None else RecoveryStore()
    registry = CodecRegistry()
    registry.register(JsonCodec(shared))
    registry.register(LogCodec(shared))
    return registry
