"""
RAVS event schema, append-only log, derived-label reducer.

Four design invariants from the v1 spec:

  1. **Honest signals only.** ``include_in_default_training=False`` flags
     events whose value is unverified self-report (the canonical example
     is the legacy ``record_outcome(success: bool)`` MCP call, where the
     agent reports its own success). These events are recorded for
     completeness but excluded by default labeling rules.

  2. **Final label is DERIVED, not stored.** The append-only log holds
     ``OutcomeEvent`` ground truth. ``derive_label`` is a pure function
     of (TraceEvent, list[OutcomeEvent], rule); offline evaluation can
     re-materialize labels under different rules without re-collecting
     data. No code path writes a stored ``final_label`` field.

  3. **Decomposition evidence is structured, not free-form.** The
     ``DecompositionEvidence`` schema is locked at v1 time so that the
     v2 compiler can promote evidence directly into plan nodes without a
     translation step. Adding a new ``kind``/``executor_candidate`` is
     a deliberate, schema-versioned change.

  4. **Passive evidence is a lower bound, not truth.** Decomposition
     evidence is collected from cheap signals only — tool calls the
     agent already emitted, retrievals it already requested, retries
     and escalations that already happened. We never resample or run
     extra inference calls in v1 to detect decomposability. The
     measured "decomposable share" is therefore a *floor* on the real
     opportunity surface — anything we miss in v1 only adds upside
     when v2 actively probes.

Storage format: JSONL, one event per line. ``TraceEvent`` and
``OutcomeEvent`` are discriminated by the ``"kind"`` field
(``"trace"`` / ``"outcome"``). Append-only — never rewritten in place.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

logger = logging.getLogger(__name__)


# ── Enumerations (deliberately string-typed for forward compatibility) ─────

# Strength levels. The reducer uses these to filter under each labeling rule.
#   "strong"  — machine-verified or non-self-reported user action
#   "medium"  — reliable behavioral signal (retry, escalation)
#   "weak"    — self-reported by the agent (ignore by default)
EVENT_STRENGTHS = ("strong", "medium", "weak")

# Honest signals (trainable by default). Anything outside this set is
# either weak or experimental.
HONEST_OUTCOME_TYPES = frozenset({
    # Strong: the world told us
    "test_result",          # user/agent ran tests; pass/fail observed
    "command_exit",         # generated command was executed; exit code observed
    "ci_result",            # CI pipeline produced a pass/fail
    "user_acceptance",      # IDE plugin reported diff accepted/rejected
    # Medium: reliable behavioral inference
    "retry_event",          # same query re-issued (suggests prior was wrong)
    "topic_change",         # session moved on (suggests prior was good enough)
    "escalation_event",     # same query sent to a stronger model after first attempt
})

# Self-report (excluded by default). Recorded for historical compat only.
WEAK_OUTCOME_TYPES = frozenset({
    "agent_self_report",    # the legacy record_outcome(success: bool) signal
})

# File extensions used by the trace-conditional ``code_change`` reducer
# rule to decide whether a trace touched code. Tuple form so it can be
# passed directly to ``str.endswith``. Kept conservative — adding a new
# language is a one-line change but every addition broadens the metric.
_CODE_FILE_EXTENSIONS = (
    ".py", ".pyi", ".pyx",
    ".rs",
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".go",
    ".java", ".kt", ".scala",
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp",
    ".rb", ".php", ".swift",
    ".cs", ".fs",
    ".sql",
    ".sh", ".bash", ".zsh",
)


# ── DecompositionEvidence — locked schema for v1, promotable to v2 plan nodes ──
#
# Five enums, each closed at v1 time. Adding a value is a deliberate,
# schema-versioned change — never silent. The v2 compiler's plan-node
# types map 1:1 onto ``kind``; ``executor_candidate`` and
# ``verifier_candidate`` map directly into the compiler's executor /
# verifier registries.

DECOMPOSITION_KINDS = frozenset({
    "computation",       # arithmetic / algebra / numerical step (SymPy-amenable)
    "retrieval",         # information lookup (file/doc/memory)
    "code_inspection",   # AST / signature / dep-graph reading
    "execution",         # run a command / test / script
    "claim_check",       # verify a factual claim against a source
    "constraint_check",  # verify a generated artifact satisfies a stated constraint
    "judgment",          # irreducible LLM judgment (Opus-bound)
    "synthesis",         # final answer assembly
})

DECOMPOSITION_SOURCES = frozenset({
    # How the evidence was observed. All v1 sources are PASSIVE:
    # we never invoke an extra LLM call to detect decomposability.
    "tool_call",             # agent emitted a tool call (strong evidence)
    "response_pattern",      # response shape matched a known pattern
    "retry",                 # retry detected by RetryCollector
    "escalation",            # escalation detected by EscalationCollector
    "test_result",           # a test result arrived (implies execution-step happened)
    "ci_result",             # CI pipeline status arrived
    "explicit_user_request", # user explicitly asked for one of the kinds
    "manual",                # human reviewer annotated the trace post-hoc
})

DECOMPOSITION_EXECUTORS = frozenset({
    "sympy",         # symbolic math
    "python",        # python sandbox
    "ast",           # python AST or equivalent for other langs
    "retrieval",     # entroly's existing knapsack retrieval
    "test_runner",   # pytest / cargo test / npm test
    "small_llm",     # cheap model adequate for the substep
    "large_llm",     # expensive model (escalation point)
    "none",          # no executor known for this kind — escalation fallback
})

DECOMPOSITION_VERIFIERS = frozenset({
    "exact",   # deterministic, single right answer (e.g. SymPy equality)
    "strong",  # tests pass / CI pass / executable confirmation
    "medium",  # retrieval citation matches; consistency across resamples
    "weak",    # another LLM agrees
    "none",    # no verifier exists for this output type
})


@dataclass
class DecompositionEvidence:
    """Structured evidence that a request had a decomposable substep.

    Recorded passively into ``TraceEvent.decomposition_evidence`` —
    never via an extra inference call. The measured rate is therefore
    a *lower bound* on the true decomposable share of a workload; v2
    can only do better than this baseline.

    Promotion to v2 plan nodes: ``kind`` becomes the node type;
    ``executor_candidate`` becomes the chosen executor; ``verifier_candidate``
    becomes the chosen verifier. The 1:1 mapping is intentional — the
    schema is the v2 IR's contract surface.
    """

    kind: str                                  # one of DECOMPOSITION_KINDS
    source: str                                # one of DECOMPOSITION_SOURCES
    executor_candidate: str = "none"           # one of DECOMPOSITION_EXECUTORS
    verifier_candidate: str = "none"           # one of DECOMPOSITION_VERIFIERS
    confidence: float = 0.5                    # collector's confidence in [0, 1]
    span_hash: str | None = None               # opaque ref to the input span (e.g. SimHash of substring)
    notes: str | None = None                   # human-readable diagnostic; never load-bearing


# ── Schemas ────────────────────────────────────────────────────────────────


@dataclass
class TraceEvent:
    """One request/decision record. Written at request boundary.

    Fields cover what the future controller (v2/v3/v4) will need without
    requiring those layers to re-instrument anything. Read carefully:
    ``shadow_recommendations`` records what each shadow policy WOULD
    have chosen, captured at decision time. Production traffic is routed
    by ``policy_decision`` only — shadows do not act on the request.
    """

    kind: str = "trace"

    # Identity
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)

    # The query (text truncated; full text only when caller chooses)
    query_text: str = ""              # truncated to ~512 chars at write time
    query_features: dict[str, Any] = field(default_factory=dict)
    # ``query_embedding_ref`` is a key/path to a separately-stored
    # embedding (we don't inline embeddings in JSONL — wastes disk).
    query_embedding_ref: str | None = None

    # The model + context that ran
    model: str = ""
    context_size_tokens: int = 0
    retrieved_fragments: list[str] = field(default_factory=list)
    tools_available: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)

    # Cost + latency (zero / -1 if unmeasured)
    latency_ms: float = -1.0
    cost_usd: float = -1.0

    # Routing decision actually used in production
    policy_decision: str = "current_heuristic"

    # Shadow policy recommendations recorded at decision time.
    # Keys are policy names; values are {model, p_success, reason}.
    # Counterfactual regret is intentionally NOT derivable from this
    # alone — shadows recommended; we don't know what they'd have done.
    shadow_recommendations: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Decomposition evidence: passively-collected hints about which
    # substeps of this request were tool-amenable / verifiable / etc.
    # Used by v1 offline analysis to estimate the decomposable-share
    # of a workload BEFORE v2 invests in the compiler runtime.
    # Each entry is a DecompositionEvidence (serialized to dict for
    # JSONL friendliness). Empty list = no decomposable substeps
    # observed (which is the lower-bound interpretation, not "none").
    decomposition_evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class OutcomeEvent:
    """An honest (or labelled-as-weak) signal that arrived AFTER a request.

    Multiple OutcomeEvents can attach to one TraceEvent via request_id.
    Order matters for the reducer (some rules use most-recent only).
    """

    kind: str = "outcome"

    request_id: str = ""             # links to TraceEvent.request_id
    timestamp: float = field(default_factory=time.time)

    event_type: str = ""             # one of HONEST_OUTCOME_TYPES | WEAK_OUTCOME_TYPES
    value: str = ""                  # "success" | "failure" | "rejected" | …
    strength: str = "weak"           # weak | medium | strong
    source: str = ""                 # which collector emitted this
    include_in_default_training: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Append-only log ────────────────────────────────────────────────────────


class AppendOnlyEventLog:
    """JSONL append-only log of TraceEvents and OutcomeEvents.

    Thread-safe. Never rewrites in place — only appends. Read paths
    return iterators so callers don't have to load the whole log.
    """

    QUERY_TRUNCATE = 512  # bytes; long queries get tail-truncated to keep lines parseable

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def write_trace(self, evt: TraceEvent) -> None:
        # Truncate query_text to keep JSONL lines well-bounded. Full
        # query is the user's concern via query_embedding_ref.
        if len(evt.query_text) > self.QUERY_TRUNCATE:
            evt = TraceEvent(**{**asdict(evt), "query_text": evt.query_text[: self.QUERY_TRUNCATE]})
        self._append(asdict(evt))

    def write_outcome(self, evt: OutcomeEvent) -> None:
        # Validate event_type against the allowed sets (warn but don't reject —
        # this is a forward-compat concern; new event types will land later
        # without code changes here).
        if (
            evt.event_type
            and evt.event_type not in HONEST_OUTCOME_TYPES
            and evt.event_type not in WEAK_OUTCOME_TYPES
        ):
            logger.debug(
                "RAVS: outcome event_type %r not in known sets; logging anyway",
                evt.event_type,
            )
        self._append(asdict(evt))

    def _append(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def read_all(self) -> Iterator[dict[str, Any]]:
        """Yield every event as a dict, in append order."""
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("RAVS: skipping malformed line in %s", self._path)
                    continue

    def traces_with_outcomes(self) -> Iterator[tuple[dict[str, Any], list[dict[str, Any]]]]:
        """Group outcomes onto their parent traces.

        Single pass over the log: build a request_id → outcomes map,
        then yield each trace with the outcomes attached. Outcomes
        without a matching trace are skipped (orphans).
        """
        traces: dict[str, dict[str, Any]] = {}
        outcomes: dict[str, list[dict[str, Any]]] = {}
        for evt in self.read_all():
            kind = evt.get("kind")
            rid = evt.get("request_id")
            if not rid:
                continue
            if kind == "trace":
                traces[rid] = evt
            elif kind == "outcome":
                outcomes.setdefault(rid, []).append(evt)
        # Preserve trace insertion order
        for rid, trace in traces.items():
            yield trace, outcomes.get(rid, [])


# ── Label reducer (the heart of the "labels are derived" invariant) ────────


def derive_label(
    trace: dict[str, Any] | TraceEvent,
    outcomes: Iterable[dict[str, Any] | OutcomeEvent],
    rule: str = "default",
) -> str:
    """Materialize a label from the (trace, outcomes) tuple under a rule.

    Rules (extend without breaking offline-eval reproducibility):

      "default"
        Any STRONG signal wins (last-write-wins among strongs). Ignores
        weak. Falls back to medium if no strong present. Returns
        "unknown" if neither.

      "strict"
        Requires at least one STRONG signal; otherwise "unknown".

      "legacy"
        Includes weak signals (the old behavior of treating
        record_outcome(success: bool) as ground truth). Provided for
        backwards compatibility comparisons; should not drive training.

      "ci_only"
        Only ci_result events count. Useful for codebase-wide eval
        runs.

      "code_change"
        Only counts signals when the trace retrieved code fragments
        AND a strong signal arrived. This is the rule v2's compiler
        will use to evaluate "did the code-change pipeline succeed?"
        without contaminating the metric with non-code traffic.

    Returns one of:
      "success" | "failure" | "rejected" | "accepted" |
      "passed"  | "failed"  | "unknown"
    """
    trace_dict: dict[str, Any] = trace if isinstance(trace, dict) else asdict(trace)  # type: ignore[arg-type]
    norm_outcomes = [
        o if isinstance(o, dict) else asdict(o)  # type: ignore[arg-type]
        for o in outcomes
    ]

    if rule == "strict":
        strongs = [o for o in norm_outcomes if o.get("strength") == "strong"]
        return _last_value(strongs) or "unknown"

    if rule == "legacy":
        # Include weak signals; preference order: strong > medium > weak
        for level in ("strong", "medium", "weak"):
            picks = [o for o in norm_outcomes if o.get("strength") == level]
            v = _last_value(picks)
            if v:
                return v
        return "unknown"

    if rule == "code_change":
        # Trace-conditional: only count if the trace touched code.
        # Heuristic: any retrieved fragment whose source path looks like
        # a code file. The code_change rule is what v2's compiler will
        # use to evaluate code-modifying pipelines without diluting
        # the metric with question-only traffic.
        retrieved = trace_dict.get("retrieved_fragments") or []
        is_code_change = any(
            isinstance(s, str) and s.endswith(_CODE_FILE_EXTENSIONS)
            for s in retrieved
        )
        if not is_code_change:
            return "unknown"
        strongs = [o for o in norm_outcomes if o.get("strength") == "strong"]
        return _last_value(strongs) or "unknown"

    if rule == "ci_only":
        ci = [o for o in norm_outcomes if o.get("event_type") == "ci_result"]
        return _last_value(ci) or "unknown"

    # default: strong > medium, weak excluded
    strongs = [o for o in norm_outcomes if o.get("strength") == "strong"]
    if strongs:
        return _last_value(strongs) or "unknown"
    mediums = [o for o in norm_outcomes if o.get("strength") == "medium"]
    return _last_value(mediums) or "unknown"


def _last_value(events: list[dict[str, Any]]) -> str:
    """Return the value of the most-recent (by timestamp) event, or empty string."""
    if not events:
        return ""
    latest = max(events, key=lambda e: float(e.get("timestamp", 0.0) or 0.0))
    return str(latest.get("value", "") or "")
