"""
RAVS — Reasoning Amplification via Verified Scaffolds.

Production-ready subsystem with five tiers:

  V1: Instrumentation — Honest outcome signals, offline eval, PRISM bridge.
  V2: Shadow Compiler — Decompose & execute cheap paths (SymPy, AST, TF-IDF retrieval).
  V3: Guarded Router — Four learned policies (heuristic, Thompson, KNN, logistic).
  V4: Sequential Controller — Budget-bounded, escalation-aware step execution.
  V5: Epistemic Cascade Engine — Mathematical uncertainty routing (Fisher, Rényi, Lyapunov).
  V6: Entropy Production Rate — Logprob-based hallucination detection (HALT/EPR, 2025-2026).
  V7: Spectral Consistency — EigenScore-inspired entity cross-similarity SVD (EMNLP 2025, ICLR 2024).
"""

from .events import (
    DECOMPOSITION_EXECUTORS,
    DECOMPOSITION_KINDS,
    DECOMPOSITION_SOURCES,
    DECOMPOSITION_VERIFIERS,
    EVENT_STRENGTHS,
    HONEST_OUTCOME_TYPES,
    AppendOnlyEventLog,
    DecompositionEvidence,
    OutcomeEvent,
    TraceEvent,
    derive_label,
)
from .report import generate_report, format_report_text
from .outcome_bridge import OutcomeBridge
from .compiler import PlanCompiler, Plan, PlanNode, NodeKind, detect_substeps
from .executors import (
    ExecutorRegistry, SymPyExecutor, PythonExecutor, ASTExecutor,
    TestRunnerExecutor, RetrievalExecutor,
)
from .verifiers import (
    VerifierRegistry, ExactVerifier, StructuralVerifier, CitationVerifier,
)
from .shadow import ShadowEvaluator
from .shadow_runner import ShadowRunner
from .router import GuardedRouter, GateStatus, compute_gate_status, classify_risk
from .controller import SequentialController, ControllerResult, EscalationPolicy
from .ece import (
    EpistemicCascadeEngine,
    UncertaintySignal,
    LyapunovThresholdController,
    compute_fisher_curvature,
    compute_renyi_entropy,
    select_renyi_alpha,
    cluster_by_simhash,
)
from .epr import (
    compute_epr,
    compute_fused_risk,
    EPRSignal,
    FusedHallucinationSignal,
)
from .spectral import (
    compute_spectral_consistency,
    SpectralSignal,
)
from .world_model import (
    anytime_hoeffding_radius,
    DreamRollout,
    EbbiforgeWorldModelAdapter,
    EmpiricalWorldModel,
    InsufficientWorldModelData,
    PromotionDecision,
    TransitionIntegrityError,
    TransitionLedger,
    TransitionReceipt,
    transition_from_ravs,
    VerifiedDreamController,
    VerifiedTransition,
    WorldModelPrediction,
)

__all__ = [
    # V1 — Instrumentation
    "DECOMPOSITION_EXECUTORS",
    "DECOMPOSITION_KINDS",
    "DECOMPOSITION_SOURCES",
    "DECOMPOSITION_VERIFIERS",
    "EVENT_STRENGTHS",
    "HONEST_OUTCOME_TYPES",
    "AppendOnlyEventLog",
    "DecompositionEvidence",
    "OutcomeEvent",
    "TraceEvent",
    "derive_label",
    "generate_report",
    "format_report_text",
    # V1+ — PRISM Bridge
    "OutcomeBridge",
    # V2 — Shadow Compiler
    "PlanCompiler",
    "Plan",
    "PlanNode",
    "NodeKind",
    "detect_substeps",
    "ExecutorRegistry",
    "SymPyExecutor",
    "PythonExecutor",
    "ASTExecutor",
    "TestRunnerExecutor",
    "RetrievalExecutor",
    "VerifierRegistry",
    "ExactVerifier",
    "StructuralVerifier",
    "CitationVerifier",
    "ShadowRunner",
    "ShadowEvaluator",
    # V3 — Guarded Router
    "GuardedRouter",
    "GateStatus",
    "compute_gate_status",
    "classify_risk",
    # V4 — Sequential Controller
    "SequentialController",
    "ControllerResult",
    "EscalationPolicy",
    # V5 — Epistemic Cascade Engine
    "EpistemicCascadeEngine",
    "UncertaintySignal",
    "LyapunovThresholdController",
    "compute_fisher_curvature",
    "compute_renyi_entropy",
    "select_renyi_alpha",
    "cluster_by_simhash",
    # V6 — Entropy Production Rate
    "compute_epr",
    "compute_fused_risk",
    "EPRSignal",
    "FusedHallucinationSignal",
    # Verified model-based learning — real transitions + bounded dreams
    "anytime_hoeffding_radius",
    "DreamRollout",
    "EbbiforgeWorldModelAdapter",
    "EmpiricalWorldModel",
    "InsufficientWorldModelData",
    "PromotionDecision",
    "TransitionIntegrityError",
    "TransitionLedger",
    "TransitionReceipt",
    "transition_from_ravs",
    "VerifiedDreamController",
    "VerifiedTransition",
    "WorldModelPrediction",
]

