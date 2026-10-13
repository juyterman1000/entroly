"""What the last session actually did, retrievable without re-reading it.

Compaction loses the *frontier*, not the objective. Measured elsewhere: an
agent's own summaries omitted key state 3.1% of the time and the **next action
8.2%** -- and independent reports describe the same shape, where the broad goal
survives while "what was already complete, what remained, and the exact next
action" degrades. So this records the frontier, not prose.

Nothing here is new infrastructure. The ``PostToolUse`` hook already fires on
every tool call and already receives the tool name, its input, its output and
the exit code; it previously read only the command and discarded the rest. The
retrieval, retention and path-classification primitives already existed too.
What was missing was a payload worth keeping and a verb to read it back.

Three decisions worth stating, because each could have gone the lazy way.

**On by default.** An event log behind an environment variable has no users, so
there is no point building one. The cost of being on is bounded instead: one
append per tool call, a retention window, and real deletion.

**Paths are classified before they are recorded.** This is the one component
here that writes filenames and command text to disk, which is a durable record
of the user's work that did not exist before. ``work_graph_path_policy``
already decides which paths may become recoverable evidence, so a secret-shaped
filename is replaced by a stable opaque id rather than stored. Command text is
truncated to its head, which is the part that identifies the action.

**Retrieval reports what it withheld.** A resume that silently returns a subset
is the same defect as a compaction that silently drops one. ``recall`` returns
the ranked events *and* a count of what it did not return, so the caller can
tell a focused answer from a truncated one.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "JournalEvent",
    "Recall",
    "journal_path",
    "record_event",
    "recall",
    "forget_session",
    "prune_journal",
]

SCHEMA = "entroly.session-journal.v1"

# One line per tool call. Bound the parts that a hostile or merely verbose
# payload could inflate; the identifying head is what retrieval needs.
MAX_TARGET_CHARS = 200
MAX_DETAIL_CHARS = 280

# Read bound. A journal is append-only, so retrieval must not grow without
# limit as a session runs long.
MAX_EVENTS_SCANNED = 20_000

# Apply retention from the write path once the file reaches this size. A
# retention window nothing invokes is not a retention window.
PRUNE_EVERY_BYTES = 256 * 1024

# How close in time two identical adjacent events must be to be treated as one
# call recorded twice rather than the same work done twice. See
# _collapse_redelivered for why this cannot be an equality test on ts.
DEDUP_WINDOW_SECONDS = 2.0

# Longest run of identical adjacent events that redelivery can explain. A host
# fans a call out once per matching hook registration, and that count is small:
# the measured case was two, reaching four only because the call was retried.
# A longer run is not fan-out, so it is left alone -- an agent that really did
# touch one file twenty times in a burst keeps all twenty. The bound errs
# toward keeping data, because collapsing real work is the worse failure.
MAX_REDELIVERY_RUN = 8

_KINDS = frozenset({"edit", "command", "error", "decision", "read", "search", "other"})

# Paths and commands carry their meaning in their parts. See JournalEvent.searchable.
_IDENTIFIER_SPLIT = re.compile(r"[^A-Za-z0-9]+")

# Shell setup is not what the call did. A multi-line invocation flattens into
# one target string, so a leading `cd <repo>` and a few `export`s push the
# actual command past MAX_TARGET_CHARS: measured on a real session, 83% of
# targets hit the cap and 61 of 69 began with the same `cd` prefix, leaving the
# ranker scoring boilerplate that every event shares. See _salient_command.
_SEGMENT_SPLIT = re.compile(r"\n|&&|\|\||;")
_PREAMBLE_VERBS = frozenset(
    {"cd", "export", "set", "unset", "source", ".", "pushd", "popd"}
)
_ENV_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")


@dataclass(frozen=True)
class JournalEvent:
    """One recorded action. ``detail`` is a head, never a full payload."""

    ts: float
    kind: str
    tool: str
    target: str
    status: str
    detail: str = ""
    session: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "ts": round(self.ts, 3),
            "kind": self.kind,
            "tool": self.tool,
            "target": self.target,
            "status": self.status,
            "detail": self.detail,
            "session": self.session,
        }

    def searchable(self) -> str:
        """Text the ranker sees, with identifiers split into their parts.

        The shared tokenizer matches a whole path as one token, so
        ``entroly/checkpoint_relevance.py`` is a single term and a query for
        "checkpoint" scores zero against it. Measured: a journal containing
        that exact file returned ``matched=0`` for "checkpoint".

        Splitting is done here rather than in the tokenizer because that
        tokenizer is shared with evidence location, where a path *is* the unit
        and splitting it would change ranking for every caller. The raw form is
        kept alongside the parts so an exact-path query still matches exactly.
        """
        parts = [self.tool, self.status]
        for token in (self.target, self.detail):
            if token:
                parts.append(_expand_identifiers(token))
        return " ".join(part for part in parts if part)


@dataclass(frozen=True)
class Recall:
    """Ranked events, plus what was left out.

    ``withheld`` is not decoration. A caller that receives five events cannot
    otherwise tell whether five matched or five hundred did.
    """

    events: tuple[JournalEvent, ...] = ()
    matched: int = 0
    withheld: int = 0
    scanned: int = 0
    truncated_scan: bool = False
    reason: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)
    redelivered_collapsed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "events": [event.as_dict() for event in self.events],
            "matched": self.matched,
            "withheld": self.withheld,
            "scanned": self.scanned,
            "truncated_scan": self.truncated_scan,
            "reason": self.reason,
            "notes": list(self.notes),
            "redelivered_collapsed": self.redelivered_collapsed,
        }


def journal_path(entroly_dir: str | Path | None = None) -> Path:
    root = Path(
        entroly_dir or os.environ.get("ENTROLY_DIR") or (Path.home() / ".entroly")
    )
    return root / "session_journal.jsonl"


def _bounded(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit] + "..."


def _expand_identifiers(text: str) -> str:
    """Return ``text`` plus the parts of any compound identifier in it.

    The shared tokenizer treats ``entroly/checkpoint_relevance.py`` as one
    term, so both sides of the comparison have to be expanded the same way.
    The raw form is kept so an exact-path query still scores highest.
    """
    parts = [text]
    for piece in _IDENTIFIER_SPLIT.split(text):
        if len(piece) > 2:
            parts.append(piece)
    return " ".join(parts)


def _salient_command(raw: object) -> str:
    """Return the part of a shell invocation that says what it did.

    ``cd``, ``export`` and bare ``VAR=value`` segments position a command; they
    are not the command. Dropping them is not cosmetic: the target is bounded
    to MAX_TARGET_CHARS, so a long preamble truncates away the verb that
    retrieval needs, and every event in the session then shares a long common
    prefix that BM25 scores as if it were signal.

    If a call is *only* setup -- ``cd somewhere`` on its own -- the original is
    kept, because an empty target is worse than a boilerplate one.
    """
    text = str(raw or "")
    if not text.strip():
        return ""
    segments = [segment.strip() for segment in _SEGMENT_SPLIT.split(text)]
    segments = [segment for segment in segments if segment]
    salient = []
    for segment in segments:
        words = segment.split()
        # `VAR=value cmd args` runs cmd. Strip the assignments, keep the
        # command -- dropping the whole segment would lose the verb.
        while words and _ENV_ASSIGNMENT.match(words[0]):
            words = words[1:]
        if not words:
            continue
        if words[0] in _PREAMBLE_VERBS:
            continue
        salient.append(" ".join(words))
    return " ".join(salient or segments)


def _safe_target(raw: object) -> str:
    """Classify a path before it becomes a durable record.

    A filename can itself be the secret -- ``.env.production``,
    ``id_rsa``, ``credentials.json``. The Work Graph already owns that
    judgement, so this defers to it rather than inventing a second rule, and
    substitutes a stable opaque id when a path must not be written down.
    """
    text = _bounded(raw, MAX_TARGET_CHARS)
    if not text:
        return ""
    try:
        from .work_graph_path_policy import classify, sensitive_id

        # The vocabulary is ordinary / sensitive / generated, not a boolean. An
        # earlier draft tested against "allowed", which matches none of them, so
        # every path was redacted -- including ordinary source files. Retrieval
        # then had no target text left to rank on and matched only error
        # details. Only "sensitive" is withheld; sensitive_id already carries
        # its own prefix, so it is returned as-is rather than wrapped again.
        disposition, _reason = classify(text)
        if disposition == "sensitive":
            return f"[{sensitive_id(text)}]"
    except Exception:
        # If the policy cannot be consulted, do not guess that a path is safe.
        return "[unclassified]"
    return text


def _classify_call(
    tool: str, status: str, target: str, detail: str
) -> str:
    lowered = tool.casefold()
    # Any non-ok status is an error. An earlier draft compared against the
    # literal "error" while record_from_hook_payload writes "exit:1", so the
    # single most valuable event kind -- a failing command, which is the
    # clearest statement of where the frontier is -- was never labelled and
    # kinds=["error"] returned nothing.
    if status and status != "ok":
        return "error"
    if lowered in {"edit", "write", "notebookedit", "multiedit"}:
        return "edit"
    if lowered in {"bash", "shell", "run"}:
        return "command"
    if lowered in {"read", "smart_read", "read_source_file"}:
        return "read"
    if lowered in {"grep", "glob", "search"}:
        return "search"
    if lowered in {"todowrite", "checkpoint_state"}:
        return "decision"
    return "other"


def record_event(
    *,
    tool: object,
    target: object = "",
    status: object = "ok",
    detail: object = "",
    session: object = "",
    kind: object = "",
    entroly_dir: str | Path | None = None,
) -> bool:
    """Append one event. Best-effort; never raises.

    This runs inside a host hook on every tool call and must not be able to
    fail the call it is observing. A lost event weakens retrieval slightly; a
    raised exception would break the user's workflow.
    """
    tool_name = _bounded(tool, 64)
    if not tool_name:
        return False

    safe_target = _safe_target(target)
    safe_status = _bounded(status, 32) or "ok"
    safe_detail = _bounded(detail, MAX_DETAIL_CHARS)
    resolved_kind = _bounded(kind, 16)
    if resolved_kind not in _KINDS:
        resolved_kind = _classify_call(
            tool_name, safe_status, safe_target, safe_detail
        )

    event = JournalEvent(
        ts=time.time(),
        kind=resolved_kind,
        tool=tool_name,
        target=safe_target,
        status=safe_status,
        detail=safe_detail,
        session=_bounded(session, 64),
    )
    path = journal_path(entroly_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {"schema": SCHEMA, **event.as_dict()},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
    except (OSError, ValueError):
        return False

    # Apply retention from the write path, because nothing else runs.
    #
    # A retention window that no caller invokes is not a retention window, and
    # a durable record of the user's work is the wrong place to leave that
    # unenforced. The check is a stat; the rewrite is amortised across roughly
    # PRUNE_EVERY_BYTES of appends, so the per-call cost stays flat.
    try:
        if path.stat().st_size >= PRUNE_EVERY_BYTES:
            prune_journal(entroly_dir)
    except OSError:
        pass
    return True


def _collapse_redelivered(
    rows: list[JournalEvent],
) -> tuple[list[JournalEvent], int]:
    """Collapse one tool call that was recorded more than once.

    A host fans a single tool call out to every matching hook registration, and
    those registrations can overlap across scopes. Measured on a real session:
    Entroly's own onboarding writes a user-scope hook matching ``Bash`` while a
    repository that ships its own matches ``Bash|Read|Edit|...``, so both fire
    and every Bash call was appended twice -- 12 byte-identical lines out of 84,
    one call recorded four times after a retry. Eight of the eight slots a
    caller asked for held three distinct commands.

    The journal stays faithful to what it received; retrieval reports distinct
    work. Identity is content plus adjacency plus a short window, deliberately
    *not* ``ts``: ``time.time()`` returned 8 distinct values across 200,000
    calls on the measured host, so a timestamp cannot tell two deliveries of
    one call from two calls. Requiring adjacency and a 2-second window keeps a
    command deliberately re-run later as two events, which it is.
    """
    def identity(event: JournalEvent) -> tuple[str, ...]:
        return (event.kind, event.tool, event.target, event.status, event.detail)

    kept: list[JournalEvent] = []
    collapsed = 0
    index = 0
    while index < len(rows):
        head = rows[index]
        run = index + 1
        while (
            run < len(rows)
            and identity(rows[run]) == identity(head)
            and abs(rows[run].ts - rows[run - 1].ts) <= DEDUP_WINDOW_SECONDS
        ):
            run += 1
        length = run - index
        if 1 < length <= MAX_REDELIVERY_RUN:
            kept.append(head)
            collapsed += length - 1
        else:
            # Either a single event, or a run too long for fan-out to explain.
            kept.extend(rows[index:run])
        index = run
    return kept, collapsed


def _read_events(
    entroly_dir: str | Path | None,
) -> tuple[list[JournalEvent], bool, int]:
    """Load events newest-last. Returns (events, scan_was_truncated, collapsed)."""
    path = journal_path(entroly_dir)
    rows: list[JournalEvent] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as stream:
            lines = stream.readlines()
    except OSError:
        return [], False, 0

    truncated = len(lines) > MAX_EVENTS_SCANNED
    for line in lines[-MAX_EVENTS_SCANNED:]:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue  # a torn append costs one event, not the journal
        if not isinstance(payload, Mapping):
            continue
        try:
            rows.append(
                JournalEvent(
                    ts=float(payload.get("ts") or 0.0),
                    kind=str(payload.get("kind") or "other"),
                    tool=str(payload.get("tool") or ""),
                    target=str(payload.get("target") or ""),
                    status=str(payload.get("status") or ""),
                    detail=str(payload.get("detail") or ""),
                    session=str(payload.get("session") or ""),
                )
            )
        except (TypeError, ValueError):
            continue
    rows, collapsed = _collapse_redelivered(rows)
    return rows, truncated, collapsed


def recall(
    query: str = "",
    *,
    limit: int = 12,
    kinds: Sequence[str] | None = None,
    entroly_dir: str | Path | None = None,
) -> Recall:
    """Return the events most relevant to ``query``, and what was withheld.

    With no query this returns the most recent events, which is the null
    hypothesis a ranker has to beat and the right answer when there is nothing
    to rank against. Ranking is lexical: BM25 over the event text, reusing the
    scorer the evidence locator already uses rather than adding a second
    definition of relevance.
    """
    events, truncated, collapsed = _read_events(entroly_dir)
    if not events:
        return Recall(reason="journal_empty", scanned=0)

    wanted = {str(kind) for kind in kinds} if kinds else None
    pool = [event for event in events if not wanted or event.kind in wanted]
    if not pool:
        return Recall(
            reason="no_events_of_requested_kind",
            scanned=len(events),
            truncated_scan=truncated,
            redelivered_collapsed=collapsed,
        )

    bound = max(1, int(limit))
    cleaned = " ".join(str(query or "").split())
    if not cleaned:
        chosen = sorted(pool, key=lambda event: event.ts)[-bound:]
        return Recall(
            events=tuple(chosen),
            matched=len(pool),
            withheld=max(0, len(pool) - len(chosen)),
            scanned=len(events),
            truncated_scan=truncated,
            redelivered_collapsed=collapsed,
            reason="most_recent",
            notes=("no query supplied, so events were ranked by recency",),
        )

    try:
        from .evidence_locator import _bm25_scores

        # Split the query the same way the events are split. Doing one side
        # only is worse than doing neither: a query like
        # "after_the_tear" or "entroly/ccr.py" tokenizes whole, so it would
        # match none of the split document terms and fail silently while
        # single-word queries kept working.
        scores = _bm25_scores(
            _expand_identifiers(cleaned), [event.searchable() for event in pool]
        )
    except Exception:
        chosen = sorted(pool, key=lambda event: event.ts)[-bound:]
        return Recall(
            events=tuple(chosen),
            matched=len(pool),
            withheld=max(0, len(pool) - len(chosen)),
            scanned=len(events),
            truncated_scan=truncated,
            redelivered_collapsed=collapsed,
            reason="ranker_unavailable_fell_back_to_recency",
        )

    ranked = sorted(
        zip(pool, scores), key=lambda pair: (-pair[1], -pair[0].ts)
    )
    hits = [event for event, score in ranked if score > 0.0]
    if not hits:
        chosen = sorted(pool, key=lambda event: event.ts)[-bound:]
        return Recall(
            events=tuple(chosen),
            matched=0,
            withheld=max(0, len(pool) - len(chosen)),
            scanned=len(events),
            truncated_scan=truncated,
            redelivered_collapsed=collapsed,
            reason="no_lexical_match",
            notes=(
                "nothing matched the query, so the most recent events were "
                "returned instead; this is recency, not relevance",
            ),
        )

    chosen = hits[:bound]
    notes: list[str] = []

    # Reserve a slot for the frontier.
    #
    # BM25 normalises by document length, and an error event is longer than a
    # plain one precisely because it carries the failure text. Measured: for the
    # query "ccr", a routine edit of entroly/ccr.py scores 1.0000 while an error
    # on the same file scores 0.5413 and ranks last of 21. So the ranker buries
    # the events this journal exists to preserve, and only surfaces them when
    # the query already names the failure -- which the resuming agent does not
    # yet know.
    #
    # The fix is composition, not scoring. Inflating an error's score would
    # claim it is more lexically relevant, which is false. Reserving a slot says
    # something true instead: whatever else matched, a failure among the matches
    # is worth one of the k, because the frontier is what was lost.
    if bound > 1 and not any(event.kind == "error" for event in chosen):
        first_error = next((event for event in hits if event.kind == "error"), None)
        if first_error is not None:
            chosen = chosen[: bound - 1] + [first_error]
            notes.append(
                "a matching error was promoted into the result; length "
                "normalisation ranks failures below shorter events that share "
                "their terms"
            )

    return Recall(
        events=tuple(chosen),
        matched=len(hits),
        withheld=max(0, len(hits) - len(chosen)),
        scanned=len(events),
        truncated_scan=truncated,
        redelivered_collapsed=collapsed,
        reason="lexical_rank",
        notes=tuple(notes),
    )


def forget_session(entroly_dir: str | Path | None = None) -> bool:
    """Delete the journal outright.

    This is the no-continue path, and it has to be real deletion rather than a
    tombstone or a filtered read. A journal of every edit and command is a
    durable record of the user's work; if the product says a fresh session
    starts clean, the bytes have to be gone.
    """
    path = journal_path(entroly_dir)
    try:
        path.unlink()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return not path.exists()


def prune_journal(entroly_dir: str | Path | None = None) -> dict[str, Any]:
    """Drop events older than the configured retention window."""
    try:
        from .privacy import prune_jsonl_by_ts, retention_cutoff_ts

        cutoff = retention_cutoff_ts()
        if cutoff is None:
            return {"status": "retention_disabled"}
        result = prune_jsonl_by_ts(
            journal_path(entroly_dir), ts_key="ts", cutoff_ts=cutoff
        )
        return {"status": "pruned", "detail": getattr(result, "__dict__", {}) or {}}
    except Exception as error:  # pragma: no cover - retention is best-effort
        return {"status": "unavailable", "reason": str(error)[:120]}


def record_from_hook_payload(
    data: Mapping[str, Any], *, entroly_dir: str | Path | None = None
) -> bool:
    """Record one ``PostToolUse`` payload.

    The hook already receives everything needed and previously kept only the
    command. Errors are the highest-value events here -- the measured loss in
    compaction is the frontier, and a failing command is the clearest statement
    of where the frontier is.
    """
    tool = data.get("tool_name") or ""
    raw_input: Any = data.get("tool_input")
    target: Any = ""
    if isinstance(raw_input, Mapping):
        target = raw_input.get("file_path") or raw_input.get("path") or ""
        if not target:
            # Only a command is rewritten. A path is already the unit retrieval
            # ranks on, and stripping segments out of one would corrupt it.
            command = raw_input.get("command")
            if command:
                target = _salient_command(command)
            else:
                target = raw_input.get("pattern") or ""
    elif isinstance(raw_input, str):
        target = raw_input

    try:
        exit_code = int(data.get("exit_code") or 0)
    except (TypeError, ValueError):
        exit_code = 0
    status = "ok" if exit_code == 0 else f"exit:{exit_code}"

    detail = ""
    if exit_code != 0:
        output = data.get("tool_output") or data.get("stderr") or ""
        detail = _bounded(str(output)[-MAX_DETAIL_CHARS * 2 :], MAX_DETAIL_CHARS)

    return record_event(
        tool=tool,
        target=target,
        status=status,
        detail=detail,
        session=data.get("session_id") or "",
        entroly_dir=entroly_dir,
    )


def _iter_dicts(events: Iterable[JournalEvent]) -> list[dict[str, Any]]:
    return [event.as_dict() for event in events]
