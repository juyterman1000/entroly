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


class JsonCodec:
    name = "json"
    version = "3"

    def __init__(self, store: RecoveryStore | None = None) -> None:
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
            return SupportDecision(False, 0.0, "opens like JSON but does not parse")
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
            data = json.loads(text)
        except ValueError:
            return reps

        elided = json.dumps(_json_to_schema(data, depth=0, max_depth=4), indent=2)
        if len(elided) >= len(text):
            return reps

        protected = _protected_json_values(data, _is_load_bearing_key)
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
    """Derive mandatory values from the source, not the compressed output."""
    out: list[str] = []

    def walk(node: Any, *, depth: int) -> None:
        if isinstance(node, dict):
            for child_key, value in node.items():
                if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                    if depth == 0 or is_load_bearing_key(str(child_key)):
                        rendered = str(value)
                        if rendered:
                            out.append(rendered)
                else:
                    walk(value, depth=depth + 1)
        elif isinstance(node, list) and node:
            walk(node[0], depth=depth + 1)

    walk(obj, depth=0)
    return _unique(out)


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
    version = "3"
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
        from .universal_compress import _compress_log_universal, _log_template

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

        protected = _protected_log_lines(text, _log_template, self._CRITICAL)
        if any(value not in collapsed for value in protected):
            return reps

        recovery = self.store.put(
            text,
            item_count=_log_omitted_count(text, _log_template),
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
    critical: re.Pattern[str],
) -> tuple[str, ...]:
    ts_strip = re.compile(r"^\S+\s+\S+\s+")
    seen: set[str] = set()
    protected: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        key = template(ts_strip.sub("", stripped))
        if key in seen:
            continue
        seen.add(key)
        if critical.search(stripped):
            protected.append(stripped)
    return _unique(protected)


def _log_omitted_count(text: str, template: Callable[[str], str]) -> int:
    ts_strip = re.compile(r"^\S+\s+\S+\s+")
    counts: Counter[str] = Counter()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            counts[template(ts_strip.sub("", stripped))] += 1
    return sum(max(0, count - 1) for count in counts.values())


class ShellCodec:
    name = "shell"
    version = "2"
    _SHELL_SHAPE = re.compile(
        r"^\s*[$#>]\s+\S|^(?:PASS|FAIL|ok|error|warning|Error|Warning)\b"
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
    registry.register(JsonCodec(shared))
    registry.register(LogCodec(shared))
    registry.register(ShellCodec(shared))
    return registry
