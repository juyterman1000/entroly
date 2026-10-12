"""Which tools a project actually uses, counted rather than guessed.

Tool-schema deferral is the largest measured context lever available: filtering
inactive schemas saves far more per turn than compressing their content. It is
built, wired into the outbound request path, and instrumented -- and inert,
because by design it withholds a schema only when the caller supplies an
``X-Entroly-Active-Tools`` allowlist. *The proxy never guesses which tools an
agent needs*, and that posture is correct: withholding a needed schema is a
correctness failure, not a cost one.

So the open question was never how to filter. It was **what evidence could
justify an allowlist**, and the answer turned out to be: none that exists today.

The ``PostToolUse`` hook fires on every ``Bash|Read|Grep|Glob|Edit|Write|
TodoWrite`` call and receives the harness tool name. It records neither. It
reads ``tool_input.command``, classifies it against CLI verifier patterns, and
returns early for anything unclassifiable -- which is every non-Bash call. On
this machine that produced 5,252 recorded outcomes across 19 "tools", all of
them verifiers (``pytest``, ``ruff``, ``cargo_test``), and zero records of the
tool calls an allowlist would need to rank.

That is what this module supplies, deliberately as a **bounded counter rather
than an event log**:

* An allowlist needs frequencies, not an audit trail. Storage is O(distinct
  tools), not O(calls), so it stays small no matter how long a session runs.
* It records a tool *name* and nothing else -- no command text, no file paths,
  no arguments. Counting that an ``Edit`` happened introduces no record of
  *what* was edited, so this adds no new privacy surface to a local-first
  product. A per-call journal would; that is a separate decision with its own
  deletion semantics, and is not taken here.
* It fails closed in the only direction that matters. No counter, too little
  data, or an unrecognised project means no allowlist, which means every schema
  is sent. A caller must still opt in; this produces a *recommendation*, never
  an applied filter.

One safety property worth stating because it is not obvious. An allowlist
derived from observed use cannot be shrunk by injected content: text in a web
page or file can at most cause an extra tool to be called, which *adds* a name
to the set and sends more schemas. The failure direction is fail-open, so this
evidence base cannot be poisoned into withholding a tool the agent needs.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "ToolActivity",
    "activity_path",
    "journal_path",
    "compact_activity",
    "record_tool_use",
    "load_activity",
    "recommend_active_tools",
]

# Matches the allowlist header's own grammar, so a counted name is always a
# name the applier would accept.
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}")

# A ranked allowlist is only worth proposing once enough turns have been seen.
# Below this the empirical frequencies are noise and the honest answer is "send
# everything".
MIN_OBSERVATIONS = 50

# Bound the table so a pathological caller cannot grow it without limit. The
# applier caps an allowlist at 64 names; counting more than this serves no
# purpose.
MAX_TRACKED_TOOLS = 256

# Fold the journal once it reaches this size. Each append is ~12 bytes, so
# this amortises one ~1 ms compaction across roughly 2,700 appends.
JOURNAL_COMPACT_BYTES = 32 * 1024

_lock = threading.Lock()


@dataclass(frozen=True)
class ToolActivity:
    """Observed tool-call counts for one project."""

    counts: dict[str, int]
    total: int

    @property
    def distinct(self) -> int:
        return len(self.counts)

    def ranked(self) -> list[tuple[str, int]]:
        """Names by descending count, ties broken by name for determinism."""
        return sorted(self.counts.items(), key=lambda item: (-item[1], item[0]))


def activity_path(entroly_dir: str | Path | None = None) -> Path:
    root = Path(entroly_dir or os.environ.get("ENTROLY_DIR") or (Path.home() / ".entroly"))
    return root / "tool_activity.json"


def journal_path(entroly_dir: str | Path | None = None) -> Path:
    """Where increments land before they are folded into the table.

    Counting is split from storing because read-modify-write was measured at
    **16.6 ms per call** -- unusable in a hook that runs on every tool call.
    Profiling put 14.2 ms of that inside ``load_activity``, and almost all of
    it in a single ``io.open``.

    The cause is that the write invalidated its own cache. Replacing the table
    creates a new inode every call, so the next read is a cold open of a
    just-created file, which on Windows is also a file an antivirus scanner has
    never seen. Reading an unchanging file measured 0.12 ms; reading one that
    was replaced a moment ago measured 14 ms.

    Appending a line to an existing file measured 0.196 ms and creates no new
    inode, so the increment never pays that cost. The table is folded in
    occasionally, amortising one expensive write across many cheap appends.
    """
    return activity_path(entroly_dir).with_suffix(".log")


def _normalize(name: object) -> str:
    if not isinstance(name, str):
        return ""
    candidate = name.strip()
    if not candidate or not _SAFE_NAME.fullmatch(candidate):
        return ""
    return candidate


def _load_table(entroly_dir: str | Path | None) -> dict[str, int]:
    """The compacted table, treating any damage as absence of evidence."""
    try:
        payload: Any = json.loads(
            activity_path(entroly_dir).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    raw = payload.get("counts")
    if not isinstance(raw, Mapping):
        return {}

    counts: dict[str, int] = {}
    for name, value in raw.items():
        clean = _normalize(name)
        if not clean:
            continue
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            counts[clean] = count
    return counts


def _load_journal(entroly_dir: str | Path | None) -> tuple[dict[str, int], int]:
    """Tally the un-compacted increments. Returns (counts, lines_read)."""
    counts: dict[str, int] = {}
    lines = 0
    try:
        with journal_path(entroly_dir).open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                lines += 1
                name = _normalize(line.strip())
                if name:
                    counts[name] = counts.get(name, 0) + 1
    except OSError:
        return {}, 0
    return counts, lines


def load_activity(entroly_dir: str | Path | None = None) -> ToolActivity:
    """Read the counter, treating any damage as absence of evidence.

    A torn append costs one count, which is a rounding error against a
    frequency table. That is why the increment can afford to be unsynchronised:
    the failure mode is a slightly low number, never a wrong allowlist, because
    a missing count can only make the recommendation *wider*.
    """
    counts = _load_table(entroly_dir)
    journal, _lines = _load_journal(entroly_dir)
    for name, count in journal.items():
        counts[name] = counts.get(name, 0) + count
    if len(counts) > MAX_TRACKED_TOOLS:
        counts = dict(
            sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
                :MAX_TRACKED_TOOLS
            ]
        )
    return ToolActivity(counts, sum(counts.values()))


def record_tool_use(
    tool_name: object, *, entroly_dir: str | Path | None = None
) -> bool:
    """Append one observation. Best-effort; never raises.

    One append to an existing file, which is the whole point: no read, no new
    inode, nothing for the next read to pay for. See :func:`journal_path` for
    the measurement that forced this shape.

    This runs inside a host hook on every tool call, so it must not be able to
    fail the call it is observing. An unwritable directory loses a count, which
    can only widen a future recommendation, never narrow one.
    """
    name = _normalize(tool_name)
    if not name:
        return False

    journal = journal_path(entroly_dir)
    try:
        journal.parent.mkdir(parents=True, exist_ok=True)
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(f"{name}\n")
    except (OSError, ValueError):
        return False

    # Fold occasionally so the journal cannot grow without bound. The stat is
    # cheap; the expensive write is amortised across JOURNAL_COMPACT_LINES
    # appends, so the per-call cost stays flat.
    try:
        if journal.stat().st_size >= JOURNAL_COMPACT_BYTES:
            compact_activity(entroly_dir)
    except OSError:
        pass
    return True


def compact_activity(entroly_dir: str | Path | None = None) -> bool:
    """Fold the journal into the table and truncate it.

    Ordering is deliberate: the table is written first and the journal is only
    cleared once that write succeeded. A crash between the two re-counts the
    journal on the next read, which double-counts some observations -- a
    frequency table tolerates that. The reverse order would *lose* them, and a
    lost count silently narrows nothing but weakens the evidence the
    recommendation rests on.
    """
    table = activity_path(entroly_dir)
    journal = journal_path(entroly_dir)
    try:
        with _lock:
            merged = load_activity(entroly_dir).counts
            if not merged:
                return False
            table.parent.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(
                dir=str(table.parent), prefix=".tool_activity-", suffix=".json"
            )
            try:
                with os.fdopen(handle, "w", encoding="utf-8") as stream:
                    json.dump(
                        {"schema": "entroly.tool-activity.v1", "counts": merged},
                        stream,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                os.replace(temporary, table)
            except BaseException:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
                raise
            try:
                journal.unlink()
            except OSError:
                pass
    except (OSError, ValueError):
        return False
    return True


def recommend_active_tools(
    declared: list[str] | tuple[str, ...],
    *,
    coverage: float = 0.99,
    entroly_dir: str | Path | None = None,
) -> tuple[list[str], str]:
    """Recommend an allowlist covering ``coverage`` of observed calls.

    Returns ``(names, reason)``. An empty list means "send every schema", and
    the reason says why -- the caller is entitled to know whether it got a
    recommendation or a refusal.

    The rule is the smallest prefix of the frequency ranking whose counts sum
    to at least ``coverage`` of all observations, plus every declared tool that
    has never been seen. That last clause is the fail-open half and it is not
    optional: a tool with no history is not evidence of disuse, it is absence
    of evidence, and withholding it would turn a cost optimisation into a
    capability regression. It also means the recommendation only ever narrows
    on tools this project has actually exercised.

    This is a frequency rule, not a risk bound. It does not certify a miss rate,
    and it must not be described as doing so: that needs held-out turns and a
    one-sided bound, which the counter alone cannot supply.
    """
    names = [name for name in (_normalize(item) for item in declared) if name]
    if not names:
        return [], "no_declared_tools"

    activity = load_activity(entroly_dir)
    if activity.total < MIN_OBSERVATIONS:
        return [], "insufficient_observations"

    seen = {name: count for name, count in activity.counts.items() if name in names}
    if not seen:
        return [], "no_overlap_with_declared_tools"

    unseen = [name for name in names if name not in seen]

    target = max(0.0, min(1.0, coverage)) * sum(seen.values())
    chosen: list[str] = []
    running = 0
    for name, count in sorted(seen.items(), key=lambda item: (-item[1], item[0])):
        chosen.append(name)
        running += count
        if running >= target:
            break

    recommended = sorted(set(chosen) | set(unseen))
    if len(recommended) >= len(names):
        return [], "no_safe_reduction"
    return recommended, "observed_frequency"
