"""
RAVS v2 — Plan Compiler (Shadow Only)

Decomposes a request into a DAG of typed nodes, each mapped to the
cheapest executor + verifier that can handle it. In v2, the compiler
runs in shadow only — production traffic still goes through the normal
model path. The compiler records what it WOULD have done.

Node types (v2 — high-confidence only):

  computation     → Python/SymPy   → exact verifier
  code_inspection → AST/retrieval  → structural verifier
  test_execution  → test runner    → exit-code verifier
  retrieval_claim → retrieval      → citation/entailment check

Judgment and prose synthesis stay model-bound — not decomposed in v2.

Design invariants:
  1. Fail closed. If any executor fails, the node falls back to
     "model-bound" — the normal model handles it. Never silent failure.
  2. Shadow only. The compiler produces plans; the shadow runner
     executes them; nothing touches production output.
  3. Cost estimation is conservative. We only claim savings when the
     verifier confirms the executor's output.
  4. Every plan is serializable as JSON for offline analysis.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("entroly.ravs.compiler")


# ── Node Types ─────────────────────────────────────────────────────────


class NodeKind(str, Enum):
    """Decomposable node types — v2 high-confidence only."""
    COMPUTATION = "computation"
    CODE_INSPECTION = "code_inspection"
    TEST_EXECUTION = "test_execution"
    RETRIEVAL_CLAIM = "retrieval_claim"
    MODEL_BOUND = "model_bound"      # irreducible — stays with the model


class ExecutorType(str, Enum):
    SYMPY = "sympy"
    PYTHON = "python"
    AST = "ast"
    TEST_RUNNER = "test_runner"
    RETRIEVAL = "retrieval"
    NONE = "none"                    # model-bound, no cheap executor


class VerifierType(str, Enum):
    EXACT = "exact"
    STRUCTURAL = "structural"
    EXIT_CODE = "exit_code"
    CITATION = "citation"
    NONE = "none"


# ── Plan Schema ────────────────────────────────────────────────────────


@dataclass
class PlanNode:
    """A single typed substep in a compiled plan."""

    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    kind: str = NodeKind.MODEL_BOUND.value
    executor: str = ExecutorType.NONE.value
    verifier: str = VerifierType.NONE.value

    # The substep's input (extracted from the query)
    input_text: str = ""
    input_span: tuple[int, int] | None = None   # (start, end) in query

    # Confidence that this substep is correctly typed
    confidence: float = 0.0

    # Cost estimation (filled by shadow runner)
    estimated_cost_usd: float = -1.0
    baseline_model_cost_usd: float = -1.0

    # Shadow execution results (filled by shadow runner)
    executor_result: str | None = None
    executor_succeeded: bool = False
    verifier_passed: bool = False
    fell_back_to_model: bool = False
    execution_time_ms: float = -1.0


@dataclass
class Plan:
    """A compiled execution plan for one request."""

    plan_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    request_id: str = ""
    timestamp: float = field(default_factory=time.time)
    query_text: str = ""

    nodes: list[PlanNode] = field(default_factory=list)

    # Summary metrics (filled after shadow execution)
    total_nodes: int = 0
    decomposed_nodes: int = 0
    model_bound_nodes: int = 0
    executor_success_count: int = 0
    verifier_pass_count: int = 0
    fallback_count: int = 0
    estimated_total_cost_usd: float = -1.0
    baseline_total_cost_usd: float = -1.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True,
                          ensure_ascii=False, default=str)


# ── Pattern Detectors ──────────────────────────────────────────────────
#
# These detect decomposable substeps from query text. They are
# conservative: false negatives are fine (we miss a substep, the model
# handles it normally), false positives are bad (we claim a substep is
# computable when it isn't).


# Computation patterns: arithmetic, algebra, unit conversion, statistics
_COMPUTATION_PATTERNS = [
    # Explicit math expressions
    re.compile(r'\b(?:calculate|compute|evaluate|solve|simplify|factor|expand|integrate|differentiate|derive)\b', re.I),
    # Arithmetic in natural language
    re.compile(r'\b(?:what is|how much is|find the value of)\s+[\d\.\+\-\*\/\(\)\^\s]+', re.I),
    # Inline expressions like "2+3", "x^2-1", "100/7"
    re.compile(r'(?:^|\s)[\d]+\s*[\+\-\*\/\^]\s*[\d]+', re.I),
    # Statistical operations
    re.compile(r'\b(?:average|mean|median|mode|std|variance|sum of|product of)\b', re.I),
    # Unit conversion
    re.compile(r'\b(?:convert|how many)\s+\w+\s+(?:to|in|per)\s+\w+', re.I),
]

# Code inspection patterns: AST queries, signature lookups, dependency analysis
_CODE_INSPECTION_PATTERNS = [
    re.compile(r'\b(?:what (?:does|is)|show me|list|find)\s+(?:the\s+)?(?:function|method|class|signature|parameters|arguments|return type|imports|dependencies)', re.I),
    re.compile(r'\b(?:how many|count)\s+(?:functions|methods|classes|lines|arguments|parameters)\b', re.I),
    re.compile(r'\b(?:what (?:does|is))\s+\w+\s*\(', re.I),  # "what does foo(" — function query
    re.compile(r'\b(?:call graph|dependency graph|import graph|type of|signature of)\b', re.I),
    re.compile(r'\blist\s+(?:the\s+)?(?:functions|methods|classes|imports)', re.I),
]

# Test execution patterns: run tests, check exit codes
_TEST_EXECUTION_PATTERNS = [
    re.compile(r'\b(?:run|execute)\s+(?:the\s+)?(?:tests?|test suite|pytest|unittest|cargo test|npm test|jest|mocha)\b', re.I),
    re.compile(r'\b(?:does|do|will)\s+(?:the\s+)?tests?\s+(?:pass|fail|succeed)\b', re.I),
    re.compile(r'\b(?:does|do|will)\s+(?:the\s+)?test\s+suite\s+(?:pass|fail|succeed)\b', re.I),
    re.compile(r'\b(?:check if|verify that|make sure)\s+.*\b(?:compiles?|builds?|passes?)\b', re.I),
]

# Retrieval claim patterns: fact-checking, citation verification
_RETRIEVAL_CLAIM_PATTERNS = [
    re.compile(r'\b(?:according to|as stated in|based on|from the)\s+(?:the\s+)?(?:docs?|documentation|readme|api|spec|specification|source)\b', re.I),
    re.compile(r'\b(?:what does the (?:docs?|documentation|readme|api) say about)\b', re.I),
    re.compile(r'\b(?:is it true that|verify that|confirm that|check whether)\b', re.I),
    re.compile(r'\b(?:look up|find in|search for)\s+.*\b(?:in the|from the)\b', re.I),
]


def detect_substeps(query: str, tools_used: list[str] | None = None) -> list[PlanNode]:
    """Detect decomposable substeps from query text.

    Conservative: only emits nodes when pattern confidence is high.
    Returns at least one MODEL_BOUND node if nothing is decomposable.
    """
    nodes: list[PlanNode] = []
    tools = set(tools_used or [])

    # Check each pattern set
    for pattern in _COMPUTATION_PATTERNS:
        m = pattern.search(query)
        if m:
            nodes.append(PlanNode(
                kind=NodeKind.COMPUTATION.value,
                executor=ExecutorType.SYMPY.value,
                verifier=VerifierType.EXACT.value,
                input_text=m.group(0).strip(),
                input_span=(m.start(), m.end()),
                confidence=0.7,
            ))
            break  # one computation node per query (v2 conservative)

    for pattern in _CODE_INSPECTION_PATTERNS:
        m = pattern.search(query)
        if m:
            nodes.append(PlanNode(
                kind=NodeKind.CODE_INSPECTION.value,
                executor=ExecutorType.AST.value,
                verifier=VerifierType.STRUCTURAL.value,
                input_text=m.group(0).strip(),
                input_span=(m.start(), m.end()),
                confidence=0.6,
            ))
            break

    for pattern in _TEST_EXECUTION_PATTERNS:
        m = pattern.search(query)
        if m:
            nodes.append(PlanNode(
                kind=NodeKind.TEST_EXECUTION.value,
                executor=ExecutorType.TEST_RUNNER.value,
                verifier=VerifierType.EXIT_CODE.value,
                input_text=m.group(0).strip(),
                input_span=(m.start(), m.end()),
                confidence=0.8,
            ))
            break

    for pattern in _RETRIEVAL_CLAIM_PATTERNS:
        m = pattern.search(query)
        if m:
            nodes.append(PlanNode(
                kind=NodeKind.RETRIEVAL_CLAIM.value,
                executor=ExecutorType.RETRIEVAL.value,
                verifier=VerifierType.CITATION.value,
                input_text=m.group(0).strip(),
                input_span=(m.start(), m.end()),
                confidence=0.5,
            ))
            break

    # Tool-based detection (from V1 trace data)
    if "calculator" in tools or "wolfram" in tools:
        if not any(n.kind == NodeKind.COMPUTATION.value for n in nodes):
            nodes.append(PlanNode(
                kind=NodeKind.COMPUTATION.value,
                executor=ExecutorType.PYTHON.value,
                verifier=VerifierType.EXACT.value,
                input_text="[detected from tool usage]",
                confidence=0.85,
            ))

    # Always include a model_bound node for the remainder
    nodes.append(PlanNode(
        kind=NodeKind.MODEL_BOUND.value,
        executor=ExecutorType.NONE.value,
        verifier=VerifierType.NONE.value,
        input_text="[remainder — judgment/synthesis, model-bound]",
        confidence=1.0,
    ))

    return nodes


# ── Plan Compiler ──────────────────────────────────────────────────────


class PlanCompiler:
    """Compile a request into a typed execution plan.

    The compiler is stateless — it takes a query and returns a Plan.
    Plans are data objects; execution is the ShadowRunner's job.
    """

    def compile(
        self,
        query: str,
        *,
        request_id: str = "",
        tools_used: list[str] | None = None,
        retrieved_fragments: list[str] | None = None,
    ) -> Plan:
        """Compile a query into an execution plan.

        Args:
            query: The user's request text.
            request_id: Trace ID for correlation with V1 events.
            tools_used: Tool names the agent used (from trace).
            retrieved_fragments: File paths retrieved (from trace).

        Returns:
            A Plan with typed nodes. At minimum, one MODEL_BOUND node.
        """
        nodes = detect_substeps(query, tools_used)

        # Boost code_inspection confidence if retrieved fragments are code files
        if retrieved_fragments:
            code_exts = ('.py', '.rs', '.js', '.ts', '.go', '.java', '.c', '.cpp')
            has_code = any(
                f.endswith(code_exts) for f in retrieved_fragments
                if isinstance(f, str)
            )
            if has_code:
                for node in nodes:
                    if node.kind == NodeKind.CODE_INSPECTION.value:
                        node.confidence = min(1.0, node.confidence + 0.15)

        plan = Plan(
            request_id=request_id,
            query_text=query[:512],
            nodes=nodes,
            total_nodes=len(nodes),
            decomposed_nodes=sum(
                1 for n in nodes if n.kind != NodeKind.MODEL_BOUND.value
            ),
            model_bound_nodes=sum(
                1 for n in nodes if n.kind == NodeKind.MODEL_BOUND.value
            ),
        )

        logger.debug(
            "PlanCompiler: %d nodes (%d decomposed, %d model-bound) for %s",
            plan.total_nodes, plan.decomposed_nodes, plan.model_bound_nodes,
            request_id,
        )

        return plan
