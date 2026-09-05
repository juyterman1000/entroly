"""Content-blind audits of local AI-agent session histories.

The auditor treats every history file as untrusted input and emits aggregate
measurements only.  Provider usage observations are interpreted by explicit
per-agent adapters; fields with unknown cumulative/additive semantics are
reported separately instead of being silently summed into a billing claim.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator


SCHEMA_VERSION = "entroly.history-audit.v2"
_INPUT_KEYS = ("input_tokens", "prompt_tokens", "promptTokenCount", "inputTokenCount")
_OUTPUT_KEYS = ("output_tokens", "completion_tokens", "candidatesTokenCount", "outputTokenCount")
_CACHE_READ_KEYS = ("cache_read_input_tokens", "cached_tokens", "cached_input_tokens", "cacheReadInputTokens")
_CACHE_WRITE_KEYS = ("cache_creation_input_tokens", "cache_write_input_tokens", "cacheWriteInputTokens")
_TOTAL_KEYS = ("total_tokens", "totalTokenCount")
_USAGE_KEYS = {"usage", "tokenusage", "usagemetadata"}
_TEXT_KEYS = {"content", "text", "input", "output", "message"}
_SUPPORTED_SUFFIXES = {".json", ".jsonl"}


def default_history_roots(home: Path | None = None) -> dict[str, tuple[Path, ...]]:
    """Return known local history roots without creating or resolving them."""
    root = home or Path.home()
    codex_home = Path(os.environ.get("CODEX_HOME") or (root / ".codex"))
    return {
        "claude": (root / ".claude" / "projects",),
        "codex": (codex_home / "sessions", codex_home / "archived_sessions"),
        "gemini": (root / ".gemini" / "tmp",),
        "opencode": (root / ".local" / "share" / "opencode",),
    }


def custom_roots(paths: Iterable[str]) -> dict[str, tuple[Path, ...]]:
    """Label explicitly supplied roots without exposing their paths in reports."""
    return {
        f"custom-{index}": (Path(raw).expanduser(),)
        for index, raw in enumerate(paths, start=1)
    }


def _integer(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def _first_int(mapping: dict[str, Any], keys: Iterable[str]) -> int:
    return next((_integer(mapping[key]) for key in keys if key in mapping), 0)


def _usage_from(mapping: Any) -> dict[str, int] | None:
    if not isinstance(mapping, dict):
        return None
    values = {
        "input_tokens": _first_int(mapping, _INPUT_KEYS),
        "output_tokens": _first_int(mapping, _OUTPUT_KEYS),
        "cache_read_tokens": _first_int(mapping, _CACHE_READ_KEYS),
        "cache_write_tokens": _first_int(mapping, _CACHE_WRITE_KEYS),
        "total_tokens": _first_int(mapping, _TOTAL_KEYS),
    }
    if not any(values.values()):
        return None
    if values["total_tokens"] == 0:
        values["total_tokens"] = sum(
            values[key]
            for key in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
        )
    return values


def _generic_usage_blocks(value: Any, parent_key: str = "") -> Iterator[dict[str, int]]:
    if isinstance(value, dict):
        normalized = parent_key.replace("-", "").replace("_", "").lower()
        direct = _usage_from(value) if normalized in _USAGE_KEYS else None
        if direct is not None:
            yield direct
            return
        for key, child in value.items():
            yield from _generic_usage_blocks(child, str(key))
    elif isinstance(value, list):
        for child in value:
            yield from _generic_usage_blocks(child, parent_key)


def _usage_observations(record: Any, agent: str) -> Iterator[tuple[str, dict[str, int]]]:
    """Yield ``(semantics, usage)`` where semantics is additive/cumulative/unknown."""
    if not isinstance(record, dict):
        return
    if agent == "codex" and record.get("type") == "event_msg":
        payload = record.get("payload")
        if isinstance(payload, dict) and payload.get("type") == "token_count":
            info = payload.get("info")
            total = info.get("total_token_usage") if isinstance(info, dict) else None
            parsed = _usage_from(total)
            if parsed:
                yield "cumulative", parsed
            return
    if agent == "claude":
        # Claude Code session rows attach per-message usage to the assistant
        # message.  Those rows are additive across the file.
        message = record.get("message")
        usage = message.get("usage") if isinstance(message, dict) else record.get("usage")
        parsed = _usage_from(usage)
        if parsed:
            yield "additive", parsed
            return
    yield from (("unknown", block) for block in _generic_usage_blocks(record))


def _text_size(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(_text_size(item) for item in value)
    if isinstance(value, dict):
        return sum(
            _text_size(child)
            for key, child in value.items()
            if str(key).lower() in _TEXT_KEYS
        )
    return 0


def _classify_event(record: Any, agent: str) -> tuple[str, int] | None:
    if not isinstance(record, dict):
        return "other", _text_size(record)
    if agent == "codex":
        # Codex duplicates completed response items into event messages.  Count
        # response_item once and reserve event_msg for cumulative usage above.
        if record.get("type") != "response_item":
            return None
        payload = record.get("payload")
        if not isinstance(payload, dict):
            return "other", 0
        kind = str(payload.get("type") or "").lower()
        role = str(payload.get("role") or "").lower()
        size = _text_size(payload.get("content", payload.get("output", payload)))
        if role == "system":
            return "system_instructions", size
        if role == "user":
            return "user_input", size
        if role == "assistant":
            return "assistant_output", size
        if "tool" in kind or "function" in kind or kind == "command_execution_output":
            return "tool_output", size
        return "other", size

    role = str(record.get("role") or "").lower()
    kind = str(record.get("type") or record.get("event_type") or "").lower()
    content = record.get("content", record.get("message", record))
    size = _text_size(content)
    if role == "system" or "system" in kind:
        return "system_instructions", size
    if role == "user" or kind in {"user", "user_message", "human"}:
        return "user_input", size
    if role == "assistant" or kind in {"assistant", "assistant_message"}:
        return "assistant_output", size
    if "tool" in role or "tool" in kind or "function" in kind:
        return "tool_output", size
    return "other", size


def _records(path: Path) -> Iterator[Any]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except (json.JSONDecodeError, RecursionError):
                    continue
        return
    try:
        parsed = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError, RecursionError):
        return
    if isinstance(parsed, list):
        yield from parsed
    else:
        yield parsed


def _candidate_files(
    roots: dict[str, tuple[Path, ...]], max_files: int
) -> tuple[list[tuple[str, Path, int]], int]:
    found: list[tuple[str, Path, int, float]] = []
    skipped_symlinks = 0
    for agent, agent_roots in roots.items():
        for root in agent_roots:
            if not root.is_dir() or root.is_symlink():
                continue
            for path in root.rglob("*"):
                try:
                    if path.is_symlink():
                        skipped_symlinks += 1
                        continue
                    if not path.is_file() or path.suffix.lower() not in _SUPPORTED_SUFFIXES:
                        continue
                    stat = path.stat()
                except OSError:
                    continue
                found.append((agent, path, stat.st_size, stat.st_mtime))
    found.sort(key=lambda item: item[3], reverse=True)
    return [(agent, path, size) for agent, path, size, _ in found[:max_files]], skipped_symlinks


def _recommendations(sinks: Counter[str], estimated_tokens: int) -> list[dict[str, Any]]:
    total_chars = max(1, sum(sinks.values()))
    candidates = [
        (
            "command-envelope",
            "tool_output",
            0.20,
            "Use recoverable command envelopes for noisy commands.",
            "entroly shrink -- <command>",
        ),
        (
            "response-contract",
            "assistant_output",
            0.30,
            "Enable an explicit concise response contract for a measured trial.",
            "entroly response set concise --scope project",
        ),
        (
            "instruction-deduplication",
            "system_instructions",
            0.20,
            "Audit repeated agent instructions before consolidating them.",
            "entroly response show --json",
        ),
    ]
    recommendations: list[dict[str, Any]] = []
    for identifier, category, threshold, summary, command in candidates:
        share = sinks[category] / total_chars
        if share < threshold:
            continue
        recommendations.append(
            {
                "id": identifier,
                "summary": summary,
                "basis": {
                    "category": category,
                    "estimated_tokens": sinks[category] // 4,
                    "share_pct": round(share * 100, 1),
                    "history_estimated_tokens": estimated_tokens,
                },
                "proposed_action": command,
                "automatic_apply": False,
                "reversible": True,
                "evidence_gate": "paired baseline/optimized task with task success and usage receipts",
            }
        )
    return recommendations


def audit_histories(
    roots: dict[str, tuple[Path, ...]] | None = None,
    *,
    max_files: int = 200,
    max_bytes: int = 64 * 1024 * 1024,
    max_file_bytes: int = 8 * 1024 * 1024,
) -> dict[str, Any]:
    """Audit local histories and return privacy-preserving aggregate evidence."""
    selected_roots = roots or default_history_roots()
    candidates, skipped_symlinks = _candidate_files(selected_roots, max(1, max_files))
    sinks: Counter[str] = Counter()
    agents: Counter[str] = Counter()
    known_usage: Counter[str] = Counter()
    unknown_usage: Counter[str] = Counter()
    files_read = records_read = bytes_read = usage_blocks = 0
    skipped_for_total_cap = skipped_for_file_cap = parse_failures = 0

    for agent, path, size in candidates:
        if size > max_file_bytes:
            skipped_for_file_cap += 1
            continue
        if bytes_read + size > max_bytes:
            skipped_for_total_cap += 1
            continue
        bytes_read += size
        files_read += 1
        agents[agent] += 1
        cumulative_peak: Counter[str] = Counter()
        try:
            for record in _records(path):
                records_read += 1
                classified = _classify_event(record, agent)
                if classified is not None:
                    sink, chars = classified
                    sinks[sink] += chars
                for semantics, block in _usage_observations(record, agent):
                    usage_blocks += 1
                    if semantics == "cumulative":
                        for key, value in block.items():
                            cumulative_peak[key] = max(cumulative_peak[key], value)
                    elif semantics == "additive":
                        known_usage.update(block)
                    else:
                        unknown_usage.update(block)
        except (OSError, UnicodeError, RecursionError):
            parse_failures += 1
            continue
        known_usage.update(cumulative_peak)

    total_chars = sum(sinks.values())
    estimated_tokens = total_chars // 4
    sink_report = [
        {
            "category": category,
            "estimated_tokens": chars // 4,
            "share_pct": round(100 * chars / max(1, total_chars), 1),
        }
        for category, chars in sinks.most_common()
        if chars
    ]
    scope_fingerprint = hashlib.sha256(
        "\n".join(sorted(selected_roots)).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "schema_version": SCHEMA_VERSION,
        "privacy": "aggregate-only; prompts, responses, commands, URLs, and paths are not emitted",
        "scope": {
            "fingerprint": scope_fingerprint,
            "agents": dict(sorted(agents.items())),
            "files_read": files_read,
            "records_read": records_read,
            "bytes_read": bytes_read,
            "max_files": max_files,
            "max_bytes": max_bytes,
            "max_file_bytes": max_file_bytes,
            "skipped_for_total_byte_cap": skipped_for_total_cap,
            "skipped_for_file_byte_cap": skipped_for_file_cap,
            "skipped_symlinks": skipped_symlinks,
            "parse_failures": parse_failures,
        },
        "provider_reported": {
            "provenance": "provider/session fields interpreted only by adapters with known semantics",
            "usage_blocks_observed": usage_blocks,
            "known_semantics": {key: known_usage[key] for key in (
                "input_tokens", "output_tokens", "cache_read_tokens",
                "cache_write_tokens", "total_tokens",
            )},
            "unknown_semantics_observed_sum": {key: unknown_usage[key] for key in (
                "input_tokens", "output_tokens", "cache_read_tokens",
                "cache_write_tokens", "total_tokens",
            )},
            "claim_boundary": (
                "Unknown-semantics fields may be cumulative or additive and are not included "
                "in comparable totals. Session exports are not billing statements."
            ),
        },
        "structural_estimate": {
            "provenance": "estimated at 4 characters per token; not billing or savings",
            "tokens": estimated_tokens,
            "sinks": sink_report,
        },
        "recommendations": _recommendations(sinks, estimated_tokens),
        "limitations": [
            "Only recognized JSON and JSONL records are inspected.",
            "Structural estimates are useful for ranking pressure, not pricing.",
            "Recommendations require an explicit, reversible action and paired validation.",
            "No task-success or answer-quality claim is inferred from token counts.",
        ],
    }


__all__ = ["SCHEMA_VERSION", "audit_histories", "custom_roots", "default_history_roots"]
