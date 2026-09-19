"""RELATE-X research primitives.

Experimental only: these modules do not wire into Entroly production paths.
"""
from .types import (
    AuthorityEnvelope,
    ConstraintSpan,
    QueryContract,
    RelationVector,
    EvidenceCandidate,
    CollisionReport,
    DifferentialSpan,
    OmissionWitness,
)
from .constraints import compile_query_contract
from .collision import detect_semantic_collision
from .differential import extract_differential_spans
from .actions import normalize_action
from .info_residual import InfoResidual, compute_residual
from .omission import verify_omission_safety

__all__ = [
    "AuthorityEnvelope",
    "ConstraintSpan",
    "QueryContract",
    "RelationVector",
    "EvidenceCandidate",
    "CollisionReport",
    "DifferentialSpan",
    "OmissionWitness",
    "compile_query_contract",
    "detect_semantic_collision",
    "extract_differential_spans",
    "normalize_action",
    "InfoResidual",
    "compute_residual",
    "verify_omission_safety",
]
