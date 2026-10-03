"""Avoidable rediscovery, scored against the predecessor's recorded state.

The previous metric summed duplicate reads, duplicate commands, duplicate edits
and *every failed command*. That is too broad and it punishes productive
debugging: running a failing test to diagnose a new implementation is progress,
not rediscovery, and re-running a test after an edit is the correct loop.

So failures are no longer part of the metric. They are reported separately and
classified. What counts here is narrower and defensible: work the successor did
that re-establishes something already established before the handoff, or that
repeats itself with nothing having changed in between.

Two components are computed automatically from an ordered event timeline:

    duplicate_read        the same file read again with no intervening edit to it
    duplicate_diagnostic  the same test/diagnostic repeated with no intervening
                          file change anywhere

Both are "with nothing having changed in between", which is what makes them
avoidable rather than iterative. Ordering is essential and is why this works on
the raw timeline rather than on the per-arm counters.

Two further components are *detected but not scored automatically*, because
deciding them correctly needs judgement that a regex should not pretend to have:

    repeated_rejected_approach   the successor tries something the predecessor
                                 recorded as tried and rejected
    lost_state_action            an action whose trace contradicts recorded
                                 predecessor state

These are emitted as candidates with the predecessor fact that makes each one a
candidate, for blinded annotation. Reporting them as confirmed counts would
inflate the metric in whichever direction the keyword list happened to favour.

Every scored item carries the predecessor fact it duplicates, so the score is
auditable rather than asserted.
"""

from __future__ import annotations

import json
import pathlib
import re

_DIAGNOSTIC = re.compile(
    r"\b(pytest|unittest|tox|nox|mypy|pyright|ruff|flake8|cargo\s+test)\b", re.I
)
_READ = re.compile(
    r"(?:Get-Content|\bcat\b|\btype\b|sed\s+-n|\bhead\b|\btail\b|\bnl\b)", re.I
)
_FILE_TOKEN = re.compile(r"[\w./\\-]+\.(?:py|toml|cfg|ini|md|json|txt|rs)\b")
_STOP = {"and", "the", "was", "tried", "rejected", "it", "is", "that", "for",
         "with", "but", "not", "this", "which", "because", "instead", "still",
         "fails", "failing", "test", "tests", "case", "would", "does"}


def _normalise_command(command: str) -> str:
    """Strip the shell wrapper so the same logical command compares equal."""
    text = command.split("-Command", 1)[-1]
    return " ".join(text.replace("'", " ").replace('"', " ").split()).lower()


def _files_in(text: str) -> set[str]:
    return {pathlib.PurePath(m.group(0)).name.lower()
            for m in _FILE_TOKEN.finditer(text)}


def _signature(phrase: str) -> set[str]:
    """Content words of a recorded rejected approach, for candidate matching."""
    words = re.findall(r"[a-z_][a-z_0-9]{3,}", phrase.lower())
    return {w for w in words if w not in _STOP}


def ordered_timeline(jsonl: str) -> list[dict]:
    """Reads, edits and commands in the order they happened.

    Codex emits ``item.completed`` with ``item.type`` in ``command_execution`` /
    ``file_change`` / ``agent_message``; ordering comes from stream position.
    """
    timeline: list[dict] = []
    for line in jsonl.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "item.completed":
            continue
        item = event.get("item") or {}
        kind = item.get("type")
        if kind == "command_execution":
            command = str(item.get("command") or "")
            normalised = _normalise_command(command)
            timeline.append({
                "kind": "command",
                "command": command,
                "normalised": normalised,
                "exit_code": item.get("exit_code"),
                "is_diagnostic": bool(_DIAGNOSTIC.search(normalised)),
                "is_read": bool(_READ.search(command)),
                "files": sorted(_files_in(command)),
            })
        elif kind == "file_change":
            for change in item.get("changes") or []:
                path = str(change.get("path") or "")
                if path:
                    timeline.append({
                        "kind": "edit",
                        "file": pathlib.PurePath(path).name.lower(),
                    })
    return timeline


def score(jsonl: str, recorded: dict | None = None) -> dict:
    """Avoidable rediscovery for one successor run."""
    timeline = ordered_timeline(jsonl)
    recorded = recorded or {}

    duplicate_reads: list[dict] = []
    duplicate_diagnostics: list[dict] = []
    read_since_change: dict[str, int] = {}     # file -> index of last read
    diagnostic_since_change: dict[str, int] = {}
    edits: list[str] = []

    for index, item in enumerate(timeline):
        if item["kind"] == "edit":
            edits.append(item["file"])
            # Any edit invalidates prior reads of that file and every prior
            # diagnostic: re-reading or re-running after a change is the correct
            # loop, not rediscovery.
            read_since_change.pop(item["file"], None)
            diagnostic_since_change.clear()
            continue

        if item["is_read"]:
            for name in item["files"]:
                previous = read_since_change.get(name)
                if previous is not None:
                    duplicate_reads.append({
                        "file": name,
                        "first_at": previous,
                        "again_at": index,
                        "why_avoidable": "no edit to this file in between",
                    })
                read_since_change[name] = index

        if item["is_diagnostic"]:
            key = item["normalised"]
            previous = diagnostic_since_change.get(key)
            if previous is not None:
                duplicate_diagnostics.append({
                    "command": item["command"][:120],
                    "first_at": previous,
                    "again_at": index,
                    "why_avoidable": "no file changed in between",
                })
            diagnostic_since_change[key] = index

    # ── candidates needing blinded annotation ──────────────────────────
    rejected_candidates: list[dict] = []
    for phrase in recorded.get("rejected", ()) or ():
        signature = _signature(phrase)
        if len(signature) < 2:
            continue
        for index, item in enumerate(timeline):
            if item["kind"] != "command":
                continue
            overlap = signature & _signature(item["normalised"])
            if len(overlap) >= 2:
                rejected_candidates.append({
                    "at": index,
                    "command": item["command"][:120],
                    "predecessor_fact": phrase,
                    "matched_terms": sorted(overlap),
                    "status": "CANDIDATE_NEEDS_BLINDED_ANNOTATION",
                })

    failures = [i for i in timeline
                if i["kind"] == "command" and isinstance(i.get("exit_code"), int)
                and i["exit_code"] != 0]

    return {
        "avoidable_rediscovery_operations": (
            len(duplicate_reads) + len(duplicate_diagnostics)
        ),
        "duplicate_reads": len(duplicate_reads),
        "duplicate_diagnostics": len(duplicate_diagnostics),
        "duplicate_read_detail": duplicate_reads,
        "duplicate_diagnostic_detail": duplicate_diagnostics,
        # Separate diagnostic, deliberately NOT in the metric (see docstring).
        "failed_commands_total": len(failures),
        "repeated_rejected_candidates": rejected_candidates,
        "timeline_length": len(timeline),
        "edit_count": len(edits),
        "duplicate_edit_paths": len(edits) - len(set(edits)),
    }


def first_progress_action(jsonl: str, recorded: dict | None = None) -> dict:
    """Index and position of the first action that advances the work.

    Frozen rubric, applied before any arm label is known: an action counts as
    progress when it edits a file, or runs the known-failing diagnostic for the
    first time, or greps for a term drawn from the recorded outstanding work.
    Pure orientation -- listing files, reading a file for the first time,
    locating an interpreter -- does not count.
    """
    recorded = recorded or {}
    wanted: set[str] = set()
    for item in recorded.get("remaining_work", ()) or ():
        wanted |= _signature(item)

    timeline = ordered_timeline(jsonl)
    seen_diagnostic = False
    for index, item in enumerate(timeline):
        if item["kind"] == "edit":
            return {"index": index, "reason": "edited a file", "of": len(timeline)}
        if item["is_diagnostic"] and not seen_diagnostic:
            seen_diagnostic = True
            return {"index": index, "reason": "ran the failing diagnostic",
                    "of": len(timeline)}
        if wanted and len(wanted & _signature(item["normalised"])) >= 2:
            return {"index": index, "reason": "searched for recorded outstanding work",
                    "of": len(timeline)}
    return {"index": None, "reason": "no progress action found",
            "of": len(timeline)}


__all__ = ["first_progress_action", "ordered_timeline", "score"]
