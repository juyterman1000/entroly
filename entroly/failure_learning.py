"""Evidence-backed failure-to-success learning proposals.

This module never edits instruction files while analyzing transcripts.  It
produces a proposal containing source hashes and line-level evidence. Applying
that proposal is a separate explicit operation which re-verifies every source,
backs up the target, and appends one marker-managed block.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "entroly.failure-learning.v1"
_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"\b(?:sk|ghp|xox[baprs])[-_A-Za-z0-9]{12,}\b"),
)
_FAILURE_WORDS = re.compile(
    r"(?i)\b(error|failed|failure|traceback|exception|permission denied|not found)\b"
)
_SUCCESS_WORDS = re.compile(r"(?i)\b(passed|success|succeeded|completed|fixed)\b")


class LearningError(RuntimeError):
    """A proposal could not be created or applied safely."""


@dataclass(frozen=True)
class TranscriptEvidence:
    source_path: str
    line_number: int
    event_index: int
    kind: str
    operation: str
    content_sha256: str
    excerpt: str


@dataclass(frozen=True)
class LearningCorrection:
    correction_id: str
    operation: str
    instruction: str
    confidence: str
    failure: TranscriptEvidence
    success: TranscriptEvidence


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _redact(value: str) -> str:
    cleaned = " ".join(value.replace("\x00", " ").split())
    cleaned = _SECRET_PATTERNS[0].sub(
        lambda match: re.split(r"[:=]", match.group(0), maxsplit=1)[0]
        + "=[REDACTED]",
        cleaned,
    )
    cleaned = _SECRET_PATTERNS[1].sub("[REDACTED_TOKEN]", cleaned)
    return cleaned[:280]


def _strings(value: Any, *, key: str = "") -> Iterable[str]:
    if isinstance(value, str):
        if key.casefold() in {
            "content", "text", "message", "output", "stdout", "stderr",
            "result", "error", "exception", "command", "cmd",
        }:
            yield value
        return
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            yield from _strings(child, key=str(child_key))
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child, key=key)


def _first_string(record: Mapping[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        value = record.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _operation(record: Mapping[str, Any]) -> str:
    tool = _first_string(record, ("tool_name", "tool", "name", "function"))
    command = _first_string(record, ("command", "cmd"))
    if not command:
        args = record.get("arguments") or record.get("input")
        if isinstance(args, Mapping):
            command = _first_string(args, ("command", "cmd"))
    executable = ""
    if command:
        match = re.search(r"(?:^|[;&|]\s*)([A-Za-z0-9_.\\/-]+)", command.strip())
        if match:
            executable = Path(match.group(1)).name.casefold()
    parts = [part.casefold() for part in (tool, executable) if part]
    return ":".join(parts)


def _classification(record: Mapping[str, Any], text: str) -> str:
    exit_code = record.get("exit_code", record.get("returncode"))
    if isinstance(exit_code, int) and not isinstance(exit_code, bool):
        return "success" if exit_code == 0 else "failure"
    if record.get("is_error") is True or record.get("success") is False:
        return "failure"
    status = str(record.get("status", "")).casefold()
    if status in {"error", "failed", "failure", "cancelled"}:
        return "failure"
    if status in {"ok", "success", "succeeded", "passed", "completed"}:
        return "success"
    if record.get("is_error") is False or record.get("success") is True:
        return "success"
    if _FAILURE_WORDS.search(text):
        return "failure"
    if _SUCCESS_WORDS.search(text):
        return "success"
    return "unknown"


def _read_events(path: Path) -> tuple[list[TranscriptEvidence], str]:
    raw = path.read_bytes()
    source_sha = _sha256_bytes(raw)
    evidence: list[TranscriptEvidence] = []
    event_index = 0
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(record, Mapping):
            continue
        event_index += 1
        text = "\n".join(_strings(record))
        operation = _operation(record)
        kind = _classification(record, text)
        if not operation or kind == "unknown":
            continue
        safe_excerpt = _redact(text or json.dumps(record, sort_keys=True))
        evidence.append(
            TranscriptEvidence(
                source_path=str(path.resolve()),
                line_number=line_number,
                event_index=event_index,
                kind=kind,
                operation=operation,
                content_sha256=_sha256_bytes(raw_line),
                excerpt=safe_excerpt,
            )
        )
    return evidence, source_sha


def _correlate(events: list[TranscriptEvidence], *, max_distance: int = 50) -> list[LearningCorrection]:
    corrections: list[LearningCorrection] = []
    consumed_successes: set[tuple[str, int]] = set()
    for failure in events:
        if failure.kind != "failure":
            continue
        success = next(
            (
                candidate
                for candidate in events
                if candidate.kind == "success"
                and candidate.source_path == failure.source_path
                and candidate.operation == failure.operation
                and 0 < candidate.event_index - failure.event_index <= max_distance
                and (candidate.source_path, candidate.event_index) not in consumed_successes
            ),
            None,
        )
        if success is None:
            continue
        consumed_successes.add((success.source_path, success.event_index))
        identity = _sha256_bytes(
            f"{failure.content_sha256}:{success.content_sha256}".encode("ascii")
        )[:16]
        instruction = (
            f"When `{failure.operation}` fails, inspect the recorded failure before retrying; "
            f"the later successful event is the verified correction evidence "
            f"({Path(success.source_path).name}:{success.line_number}, sha256:{success.content_sha256[:12]})."
        )
        corrections.append(
            LearningCorrection(
                correction_id=f"fl-{identity}",
                operation=failure.operation,
                instruction=instruction,
                confidence="observed_same_operation_failure_then_success",
                failure=failure,
                success=success,
            )
        )
    return corrections


def build_learning_proposal(paths: Iterable[Path]) -> dict[str, Any]:
    resolved = [Path(path).expanduser().resolve() for path in paths]
    if not resolved:
        raise LearningError("at least one transcript path is required")
    all_events: list[TranscriptEvidence] = []
    sources: list[dict[str, Any]] = []
    for path in resolved:
        if not path.is_file():
            raise LearningError(f"transcript is not a readable file: {path}")
        events, source_sha = _read_events(path)
        all_events.extend(events)
        sources.append(
            {
                "path": str(path),
                "sha256": source_sha,
                "evidence_events": len(events),
            }
        )
    corrections = _correlate(all_events)
    proposal_id = "flp-" + _sha256_bytes(
        json.dumps(
            {
                "sources": sources,
                "corrections": [asdict(item) for item in corrections],
            },
            sort_keys=True,
        ).encode("utf-8")
    )[:16]
    return {
        "schema_version": SCHEMA_VERSION,
        "proposal_id": proposal_id,
        "created_at_unix": int(time.time()),
        "mode": "dry_run_proposal_only",
        "sources": sources,
        "corrections": [asdict(item) for item in corrections],
        "limitations": [
            "Correlation proves sequence for the same normalized operation, not causality.",
            "No instruction file is changed until explicit proposal application.",
            "Excerpts are bounded and secret-pattern redacted; source transcripts remain local.",
        ],
    }


def write_learning_proposal(proposal: Mapping[str, Any], path: Path) -> Path:
    target = Path(path).expanduser().resolve()
    _atomic_write(
        target,
        (json.dumps(dict(proposal), indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return target


def _verify_proposal_sources(proposal: Mapping[str, Any]) -> None:
    for source in proposal.get("sources", []):
        path = Path(str(source.get("path", ""))).resolve()
        try:
            current = _sha256_bytes(path.read_bytes())
        except OSError as exc:
            raise LearningError(f"cannot verify transcript source {path}: {exc}") from exc
        if current != source.get("sha256"):
            raise LearningError(f"transcript source changed after proposal creation: {path}")


def render_learning_block(proposal: Mapping[str, Any]) -> str:
    proposal_id = str(proposal["proposal_id"])
    lines = [
        f"<!-- entroly-learning:{proposal_id}:start -->",
        "## Entroly verified learnings",
        "",
        "These bounded instructions cite observed local transcript evidence; sequence is not causality.",
        "",
    ]
    corrections = proposal.get("corrections", [])
    if not corrections:
        raise LearningError("proposal contains no verified failure-to-success correction")
    for item in corrections:
        failure = item["failure"]
        success = item["success"]
        lines.append(f"- {item['instruction']}")
        failure_name = Path(str(failure["source_path"])).name
        success_name = Path(str(success["source_path"])).name
        lines.append(
            f"  Evidence: failure `{failure_name}:{failure['line_number']}` "
            f"-> success `{success_name}:{success['line_number']}`."
        )
    lines.extend(["", f"<!-- entroly-learning:{proposal_id}:end -->", ""])
    return "\n".join(lines)


def apply_learning_proposal(proposal_path: Path, target_path: Path) -> dict[str, str]:
    proposal_file = Path(proposal_path).expanduser().resolve()
    target = Path(target_path).expanduser().resolve()
    try:
        proposal = json.loads(proposal_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LearningError(f"cannot read learning proposal: {exc}") from exc
    if proposal.get("schema_version") != SCHEMA_VERSION:
        raise LearningError("learning proposal has an unsupported schema")
    _verify_proposal_sources(proposal)
    try:
        existing = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise LearningError(f"target instruction file is not readable: {target}") from exc
    marker = f"<!-- entroly-learning:{proposal['proposal_id']}:start -->"
    if marker in existing:
        raise LearningError("learning proposal is already present in the target")
    block = render_learning_block(proposal)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = target.with_name(f"{target.name}.entroly-backup-{stamp}")
    _atomic_write(backup, existing.encode("utf-8"))
    separator = "" if existing.endswith("\n") else "\n"
    _atomic_write(target, (existing + separator + "\n" + block).encode("utf-8"))
    return {
        "proposal_id": str(proposal["proposal_id"]),
        "target": str(target),
        "backup": str(backup),
    }


__all__ = [
    "LearningError",
    "SCHEMA_VERSION",
    "apply_learning_proposal",
    "build_learning_proposal",
    "render_learning_block",
    "write_learning_proposal",
]
