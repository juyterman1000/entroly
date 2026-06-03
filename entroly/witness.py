"""WITNESS: proof-carrying factuality and hallucination suppression.

The core rule is intentionally stricter than ordinary hallucination detection:
neutral evidence is not negative evidence, and weak retrieval cannot convict a
claim. WITNESS returns proof certificates first, then a workload profile decides
whether unsupported or unknown claims should be suppressed or only warned.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

from .witness_calibration import (
    Action,
    ThresholdSet,
    default_thresholds,
)

if TYPE_CHECKING:
    from .witness_risk_model import RiskModel

logger = logging.getLogger(__name__)


_STOPWORDS = frozenset(
    """
    about after again against also although among another before being between
    both could does doing done each either every first from have into just last
    like made make many more most much only other over should some such than
    that their them then there these they this those very were what when where
    which while with would yes sure okay actually believe
    """.split()
)

_QUESTION_STARTERS = frozenset(
    "who what when where why how which whose is are was were do does did can could should would".split()
)

_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

_NLI_CACHE: dict[tuple[str, str], "NLIVerdict"] = {}

# Lazy module-level risk model used by the continuous-risk path in
# WitnessAnalyzer._certify_claim. Constructed on first access via
# `_get_default_risk_model()` so importing this module remains cheap.
_DEFAULT_RISK_MODEL: "RiskModel | None" = None  # noqa: F821


def _get_default_risk_model() -> "RiskModel":  # noqa: F821
    """Return the process-wide default RiskModel, constructing on demand."""
    global _DEFAULT_RISK_MODEL
    if _DEFAULT_RISK_MODEL is None:
        from .witness_risk_model import RiskModel
        _DEFAULT_RISK_MODEL = RiskModel()
    return _DEFAULT_RISK_MODEL

try:
    from entroly_core import py_witness_analyze as _rust_witness_analyze  # type: ignore
    from entroly_core import py_witness_claims as _rust_witness_claims  # type: ignore
except Exception:
    _rust_witness_analyze = None
    _rust_witness_claims = None

# ── Single source of truth ───────────────────────────────────────────
# WITNESS had two implementations (Python here, Rust in entroly-core).
# The proxy/MCP hot path silently preferred Rust when the native engine
# was installed — and the Rust path diverged badly from the calibrated,
# benchmarked Python behaviour (shipped QA retention ~0.11 vs validated
# 0.74), so native-engine users got a different, broken WITNESS than
# every report measured. The user-facing product must be ONE behaviour,
# not "depends which binary you have".
#
# Resolution: the validated Python implementation is the one shipped
# behaviour for every surface (proxy, MCP, SDK, CLI). The Rust fast-path
# is OFF unless explicitly opted in AND it has passed the conformance
# gate (tests/test_witness_parity.py). Nulling the handles here is the
# single chokepoint — every call site falls through to Python with no
# per-caller changes. Re-enable only once conformant:
#     ENTROLY_WITNESS_RUST=1
if os.environ.get("ENTROLY_WITNESS_RUST", "0") != "1":
    _rust_witness_analyze = None
    _rust_witness_claims = None


@dataclass(frozen=True)
class Claim:
    id: int
    text: str
    start: int
    end: int
    kind: str = "proposition"


@dataclass(frozen=True)
class EvidenceWindow:
    text: str
    score: float
    adequacy: float
    entity_coverage: float
    number_coverage: float
    token_coverage: float


@dataclass(frozen=True)
class NLIVerdict:
    label: str  # entailment | contradiction | neutral
    confidence: float
    evidence: str
    adequacy: float


@dataclass(frozen=True)
class ProofStep:
    operator: str
    evidence: str
    strength: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "evidence": self.evidence,
            "strength": round(self.strength, 4),
        }


@dataclass(frozen=True)
class Certificate:
    claim_id: int
    claim_text: str
    label: str  # grounded | contradicted | unsupported | unknown
    support_strength: float
    contradiction_strength: float
    evidence_adequacy: float
    risk: float
    proof_path: list[ProofStep]

    @property
    def is_actionable(self) -> bool:
        return self.label != "grounded"

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "label": self.label,
            "support_strength": round(self.support_strength, 4),
            "contradiction_strength": round(self.contradiction_strength, 4),
            "evidence_adequacy": round(self.evidence_adequacy, 4),
            "risk": round(self.risk, 4),
            "proof_path": [step.as_dict() for step in self.proof_path],
        }


@dataclass(frozen=True)
class WitnessResult:
    output: str
    certificates: list[Certificate]
    summary_score: float
    n_grounded: int
    n_unsupported: int
    n_contradicted: int
    n_unknown: int
    latency_ms: float

    def flagged(self) -> list[Certificate]:
        return [cert for cert in self.certificates if cert.is_actionable]

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary_score": round(self.summary_score, 4),
            "n_claims": len(self.certificates),
            "n_grounded": self.n_grounded,
            "n_unsupported": self.n_unsupported,
            "n_contradicted": self.n_contradicted,
            "n_unknown": self.n_unknown,
            "latency_ms": round(self.latency_ms, 1),
            "certificates": [cert.as_dict() for cert in self.certificates],
        }


@dataclass(frozen=True)
class WitnessRewrite:
    """Result of applying a product policy to a WITNESS analysis."""

    output: str
    changed: bool
    mode: str
    flagged_count: int
    profile: str = "auto"
    suppressed_count: int = 0
    warned_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "changed": self.changed,
            "mode": self.mode,
            "profile": self.profile,
            "flagged_count": self.flagged_count,
            "suppressed_count": self.suppressed_count,
            "warned_count": self.warned_count,
        }


def extract_claims(output: str, *, force_python: bool = False) -> list[Claim]:
    """Extract independently checkable factual claims from model output.

    Dialogue-fallback rule: if no sentence-level claims survive filtering
    AND the output has any factual content at all (entities, numbers, or
    conversational content longer than a stub), treat the whole response
    as a single proposition-class claim. This is what fixes the
    "Dialogue F1 = 0.06" failure mode — without it, conversational
    responses produce zero certificates and bypass the policy entirely.

    Args:
        output: Model output to extract from.
        force_python: When True, skip the Rust fast-path and run the
            Python extractor. Used by the benchmark to validate Python
            improvements end-to-end.
    """
    if _rust_witness_claims is not None and not force_python:
        try:
            raw_claims = json.loads(_rust_witness_claims(output))
            claims = [
                Claim(
                    id=int(item.get("id", idx)),
                    text=str(item.get("text", "")),
                    start=int(item.get("start", 0)),
                    end=int(item.get("end", 0)),
                    kind=str(item.get("kind", "proposition")),
                )
                for idx, item in enumerate(raw_claims)
                if isinstance(item, dict) and str(item.get("text", "")).strip()
            ]
            # If Rust returned zero, still apply the dialogue fallback so
            # the policy has *something* to act on.
            if not claims:
                fb = _dialogue_fallback_claim(output)
                if fb is not None:
                    claims = [fb]
            return claims
        except Exception as e:
            logger.debug("Rust WITNESS claim extraction failed; falling back to Python: %s", e)

    claims: list[Claim] = []
    for start, segment in _candidate_claim_segments(output):
        text = _clean_claim_segment(segment)
        if not _is_claim_like(text):
            continue
        kind = _classify_claim(text)
        claims.append(Claim(id=len(claims), text=text, start=start, end=start + len(text), kind=kind))

    if not claims:
        fb = _dialogue_fallback_claim(output)
        if fb is not None:
            claims = [fb]
    return claims


def _dialogue_fallback_claim(output: str) -> Claim | None:
    """When sentence-level extraction yields nothing, treat the whole
    response as a single claim — if it has any substantive content.

    Empty / pure-acknowledgment responses ("ok", "thanks", "yes") still
    return None because there is no factual content to verify.

    This fixes a critical gap: HaluEval-Dialogue responses are short and
    often lack the strict declarative-sentence markers that
    `_is_claim_like` requires (it demands a claim verb, entity, number,
    or code marker). For dialogue we want a more permissive policy:
    if there's substance, verify it.
    """
    cleaned = (output or "").strip()
    if not cleaned:
        return None
    # Pure backchannels — no fact content possible
    pure_ack = {"ok", "okay", "sure", "yes", "no", "thanks", "thank you",
                "got it", "alright", "sounds good", "right", "yeah"}
    if cleaned.lower().strip(".!? ") in pure_ack:
        return None
    # Minimum substantive length (avoids "hi" being verified)
    if len(cleaned) < 15:
        return None
    return Claim(
        id=0,
        text=cleaned[:1000],
        start=0,
        end=min(len(cleaned), 1000),
        kind="dialogue_response",
    )


def apply_witness_policy(
    output: str,
    result: WitnessResult,
    *,
    mode: str = "audit",
    profile: str = "auto",
    max_items: int = 6,
    thresholds: ThresholdSet | None = None,
) -> WitnessRewrite:
    """Apply a product policy to an already-analyzed output.

    Modes:
      - audit: no visible output change; callers use certificates/headers.
      - annotate: keep the output and append compact verification warnings.
      - strict: 4-action graduated policy (pass / hedge / warn / suppress)
                with thresholds from `thresholds` or per-profile defaults.

    The graduated policy replaces the old binary {pass, suppress} by
    branching on the continuous risk score ρ per certificate via a
    ThresholdSet. Mathematical specification and the conformal
    calibration procedure for computing the thresholds is documented in
    entroly/witness_calibration.py.

    Per-action behavior in strict mode:
        PASS     — no rewrite
        HEDGE    — keep claim, append "[unverified]" note for the run
        WARN     — keep claim, surface as a visible warning line
        SUPPRESS — remove claim from the output

    `unknown`-labeled certificates are now driven by ρ instead of by
    profile — eliminating the old "summary profile leaks everything"
    and "qa profile suppresses everything" failure modes observed on
    HaluEval.
    """
    normalized_mode = (mode or "audit").strip().lower()
    normalized_profile = _normalize_profile(profile, output)
    if normalized_mode in {"off", "audit"}:
        return WitnessRewrite(
            output=output,
            changed=False,
            mode=normalized_mode,
            profile=normalized_profile,
            flagged_count=len(result.flagged()),
        )

    flagged = result.flagged()
    if not flagged:
        return WitnessRewrite(
            output=output,
            changed=False,
            mode=normalized_mode,
            profile=normalized_profile,
            flagged_count=0,
        )

    if normalized_mode == "annotate":
        lines = [
            "",
            "",
            "[Entroly WITNESS] Verification warnings:",
        ]
        for cert in flagged[:max_items]:
            lines.append(f"- {cert.label}: {cert.claim_text}")
        if len(flagged) > max_items:
            lines.append(f"- ... {len(flagged) - max_items} more")
        return WitnessRewrite(
            output=output.rstrip() + "\n".join(lines),
            changed=True,
            mode=normalized_mode,
            profile=normalized_profile,
            flagged_count=len(flagged),
            suppressed_count=0,
            warned_count=len(flagged),
        )

    if normalized_mode == "strict":
        # Resolve thresholds: explicit override, then calibrated defaults,
        # then risk-graduated per-profile defaults.
        ts = thresholds if thresholds is not None else default_thresholds(normalized_profile)

        # Action assignment per certificate via continuous ρ.
        # Contradicted claims always SUPPRESS (cert.risk = 0.92 ≥ τ_warn for all profiles).
        # Grounded claims always PASS (cert.risk ≤ 0.35 < τ_pass for all profiles).
        # The interesting cases are unsupported (risk=0.80) and unknown
        # (risk=0.55), which previously hard-mapped per profile but now
        # route through the calibrated ThresholdSet.
        actions: list[tuple[Certificate, str]] = []
        for cert in flagged:
            action = ts.action(cert.risk)
            # Escalation: zero support AND zero adequacy means the claim
            # has NO evidence at all — strictly stronger than "weak evidence".
            # The bucket-risk path from Rust may assign risk=0.55 (unknown),
            # which falls in the WARN band, but a claim with literally no
            # supporting evidence should be suppressed in strict mode.
            if (
                cert.support_strength <= 0.0
                and cert.evidence_adequacy <= 0.01
                and action in (Action.HEDGE, Action.WARN)
            ):
                action = Action.SUPPRESS
            if (
                normalized_profile in {"summary", "summarization"}
                and cert.contradiction_strength < 0.70
                and action == Action.SUPPRESS
            ):
                action = Action.WARN
            actions.append((cert, action))

        suppressed = [c for c, a in actions if a == Action.SUPPRESS]
        warned = [c for c, a in actions if a == Action.WARN]
        hedged = [c for c, a in actions if a == Action.HEDGE]

        rewritten = output
        for cert in sorted(suppressed, key=lambda c: len(c.claim_text), reverse=True):
            rewritten = _remove_claim_text(rewritten, cert.claim_text)
        rewritten = _cleanup_rewritten_output(rewritten)
        if suppressed and not rewritten:
            rewritten = "Entroly WITNESS suppressed this answer because its factual claims lacked support."
        elif suppressed and rewritten != output:
            rewritten = (
                rewritten.rstrip()
                + f"\n\n[Entroly WITNESS suppressed {len(suppressed)} unsupported factual claim(s).]"
            )
        if warned:
            lines = ["", "", "[Entroly WITNESS warning: some factual claims could not be fully verified:"]
            for cert in warned[:max_items]:
                lines.append(f"- {cert.label}: {cert.claim_text}")
            if len(warned) > max_items:
                lines.append(f"- ... {len(warned) - max_items} more")
            lines.append("]")
            rewritten = rewritten.rstrip() + "\n".join(lines)
        if hedged:
            lines = ["", "", "[Entroly WITNESS notes (unverified — present but not fully grounded):"]
            for cert in hedged[:max_items]:
                lines.append(f"- {cert.claim_text}")
            if len(hedged) > max_items:
                lines.append(f"- ... {len(hedged) - max_items} more")
            lines.append("]")
            rewritten = rewritten.rstrip() + "\n".join(lines)

        # `warned_count` semantics preserved for backward compat with
        # callers that read it (proxy stats, dashboards). Treat HEDGE as
        # a non-blocking warning at the count level so existing CI
        # gates don't break.
        return WitnessRewrite(
            output=rewritten,
            changed=rewritten != output,
            mode=normalized_mode,
            profile=normalized_profile,
            flagged_count=len(flagged),
            suppressed_count=len(suppressed),
            warned_count=len(warned) + len(hedged),
        )

    raise ValueError(f"unknown WITNESS mode {mode!r}; expected off, audit, annotate, or strict")


def format_witness_report(result: WitnessResult, *, max_items: int = 12) -> str:
    """Return a compact human-readable certificate report."""
    lines = [
        f"WITNESS score: {result.summary_score:.3f}",
        (
            "claims: "
            f"{len(result.certificates)} total, {result.n_grounded} grounded, "
            f"{result.n_contradicted} contradicted, {result.n_unsupported} unsupported, "
            f"{result.n_unknown} unknown"
        ),
    ]
    for cert in result.certificates[:max_items]:
        lines.append(
            f"- [{cert.label}] risk={cert.risk:.2f} support={cert.support_strength:.2f} "
            f"contradiction={cert.contradiction_strength:.2f}: {cert.claim_text}"
        )
    if len(result.certificates) > max_items:
        lines.append(f"- ... {len(result.certificates) - max_items} more")
    return "\n".join(lines)


def _witness_result_from_dict(data: dict[str, Any]) -> WitnessResult:
    certificates = [
        Certificate(
            claim_id=int(cert.get("claim_id", 0)),
            claim_text=str(cert.get("claim_text", "")),
            label=str(cert.get("label", "unknown")),
            support_strength=float(cert.get("support_strength", 0.0)),
            contradiction_strength=float(cert.get("contradiction_strength", 0.0)),
            evidence_adequacy=float(cert.get("evidence_adequacy", 0.0)),
            risk=float(cert.get("risk", 0.55)),
            proof_path=[
                ProofStep(
                    operator=str(step.get("operator", "")),
                    evidence=str(step.get("evidence", "")),
                    strength=float(step.get("strength", 0.0)),
                )
                for step in cert.get("proof_path", [])
                if isinstance(step, dict)
            ],
        )
        for cert in data.get("certificates", [])
        if isinstance(cert, dict)
    ]
    return WitnessResult(
        output=str(data.get("output", "")),
        certificates=certificates,
        summary_score=float(data.get("summary_score", 1.0)),
        n_grounded=int(data.get("n_grounded", 0)),
        n_unsupported=int(data.get("n_unsupported", 0)),
        n_contradicted=int(data.get("n_contradicted", 0)),
        n_unknown=int(data.get("n_unknown", 0)),
        latency_ms=float(data.get("latency_ms", 0.0)),
    )


def _witness_rewrite_from_dict(data: dict[str, Any]) -> WitnessRewrite:
    return WitnessRewrite(
        output=str(data.get("output", "")),
        changed=bool(data.get("changed", False)),
        mode=str(data.get("mode", "audit")),
        profile=str(data.get("profile", "auto")),
        flagged_count=int(data.get("flagged_count", 0)),
        suppressed_count=int(data.get("suppressed_count", 0)),
        warned_count=int(data.get("warned_count", 0)),
    )


# ── Novel-entity risk floor ───────────────────────────────────────────────
# A claim that introduces content words entirely absent from the evidence is a
# strong hallucination signal the learned risk model can under-weight when the
# claim's SUBJECT is grounded but its PREDICATE is fabricated (e.g. "AutoGPT
# uses quantum bananas" — "AutoGPT" is in evidence, "quantum bananas" invented).
# STAVE catches wrong relations between KNOWN entities, not a freshly-invented
# noun phrase. This floor is deliberately conservative — it fires only when a
# MAJORITY of the claim's content words (and >=2 of them) are novel — so
# faithful paraphrases, which share most vocabulary with the evidence, are
# unaffected. Validated to not regress HaluEval-QA AUROC.
_NOVEL_STOPWORDS = frozenset("""
the a an of to in on for with by and or but not is are was were be been being
this that these those it its as at from into over under up down out off about
he she they we you i me him her them us my your his their our then than so if
has have had do does did can could should would will may might must shall
uses use used using made make makes get gets got also more most very such
which who whom whose what when where why how all any both each few many some
""".split())
_NOVEL_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")


def _novel_content_tokens(text: str) -> set[str]:
    return {
        w.lower() for w in _NOVEL_WORD_RE.findall(text or "")
        if w.lower() not in _NOVEL_STOPWORDS
    }


def _novel_content_floor(claim: str, evidence: str, risk: float, floor: float = 0.45) -> float:
    """Raise ``risk`` to at least ``floor`` when the claim contains a fabricated
    noun phrase: two or more *adjacent* content words absent from the evidence
    (e.g. "quantum bananas"). Detecting an adjacent novel run — rather than a
    bag-of-words ratio — keeps this conservative: faithful paraphrases swap a
    single synonym at a time and seldom introduce two brand-new neighbouring
    content words, whereas a hallucinated entity arrives as a fresh phrase.
    Never lowers risk."""
    claim_seq = [
        w.lower() for w in _NOVEL_WORD_RE.findall(claim or "")
        if w.lower() not in _NOVEL_STOPWORDS
    ]
    if len(claim_seq) < 2:
        return risk
    evidence_words = _novel_content_tokens(evidence)
    run = 0
    for w in claim_seq:
        run = run + 1 if w not in evidence_words else 0
        if run >= 2:
            return max(risk, floor)
    return risk


class WitnessAnalyzer:
    """Proof-carrying factuality analyzer.

    Args:
        use_nli: When true, call OpenAI for entailment/contradiction. If the
            API is unavailable, WITNESS falls back to deterministic local PAV.
        model: NLI model name.
        support_threshold: minimum support strength for a grounded certificate.
        contradiction_threshold: minimum contradiction strength for a hard block.
        adequacy_threshold: evidence adequacy needed before "unsupported" is
            allowed. Below this, the label is "unknown".
    """

    def __init__(
        self,
        *,
        use_nli: bool = False,
        use_local_nli: bool | None = None,
        use_stave: bool = True,
        model: str = "gpt-4o-mini",
        profile: str = "auto",
        support_threshold: float = 0.62,
        contradiction_threshold: float = 0.70,
        adequacy_threshold: float = 0.72,
        nli_timeout_s: float = 6.0,
        nli_max_claims: int = 8,
        force_python: bool = False,
        thresholds: "ThresholdSet | None" = None,  # noqa: F821
    ) -> None:
        self.use_nli = use_nli
        # Local NLI: default = env var ENTROLY_LOCAL_NLI, else False.
        # Set True to use DeBERTa-v3-small (~80 MB) for zero-API entailment.
        if use_local_nli is None:
            self.use_local_nli = os.getenv("ENTROLY_LOCAL_NLI", "").lower() in ("1", "true", "yes")
        else:
            self.use_local_nli = use_local_nli
        # STAVE: binary-relational slot verifier. Default on; disable with
        # ENTROLY_STAVE=0 or use_stave=False.
        _stave_env = os.getenv("ENTROLY_STAVE", "1").lower()
        self.use_stave = use_stave and _stave_env not in ("0", "false", "no")
        self.model = model
        self.profile = profile
        self.support_threshold = support_threshold
        self.contradiction_threshold = contradiction_threshold
        self.adequacy_threshold = adequacy_threshold
        self.nli_timeout_s = nli_timeout_s
        self.nli_max_claims = nli_max_claims
        self.force_python = force_python
        # ThresholdSet for the 4-action graduated policy. When None we
        # resolve defaults from the profile at policy time.
        self.thresholds = thresholds
        self.nli_calls = 0
        self.nli_timeouts = 0
        self.nli_fallbacks = 0
        self.nli_input_tokens_est = 0
        self.nli_output_tokens_est = 0

    def analyze(self, context: str, output: str, retrieved: list[str] | None = None) -> WitnessResult:
        t0 = time.perf_counter()
        evidence_context = context
        if retrieved:
            evidence_context = context + "\n\n" + "\n\n".join(retrieved)

        if not self.use_nli and not self.force_python and _rust_witness_analyze is not None:
            try:
                payload = json.loads(_rust_witness_analyze(
                    evidence_context,
                    output,
                    "audit",
                    self.profile,
                    self.support_threshold,
                    self.contradiction_threshold,
                    self.adequacy_threshold,
                ))
                return _witness_result_from_dict(payload["witness"])
            except Exception as e:
                logger.debug("Rust WITNESS analyze failed; falling back to Python: %s", e)

        claims = extract_claims(output, force_python=self.force_python)
        if self.use_nli and os.getenv("OPENAI_API_KEY"):
            certificates = self._certify_claims_with_batch_nli(claims, evidence_context)
        else:
            certificates = [
                self._certify_claim(claim, evidence_context)
                for claim in claims
            ]
        counts = Counter(cert.label for cert in certificates)
        if certificates:
            summary_score = 1.0 - sum(cert.risk for cert in certificates) / len(certificates)
        else:
            summary_score = 1.0

        # ── STAVE enhancement ────────────────────────────────────────────
        # Blend Relational Slot Fidelity into summary_score when STAVE has a
        # non-neutral signal (coverage ~35% of QA decisions). stave_risk is a
        # *risk*: 0.0 = grounded, 1.0 = wrong-slot/hallucinated, 0.5 = no signal.
        # A high risk (wrong-slot gate, capped >=0.75) is hard-penalised; any
        # other relational signal is soft-blended at 25% on the grounding
        # scale. Wrong-slot precision is 100% on HaluEval-QA, so the hard cap
        # does not introduce false positives. Disable with use_stave=False or
        # ENTROLY_STAVE=0.
        if self.use_stave:
            try:
                from .verifiers.stave import stave_risk as _stave_risk
                # Extract knowledge-only (strip trailing "Question: ...")
                _knowledge = (
                    evidence_context.split("\n\nQuestion:")[0].strip()
                    if "\n\nQuestion:" in evidence_context
                    else evidence_context
                )
                _sr = _stave_risk(output, _knowledge)
                if _sr != 0.5:  # STAVE has relational signal
                    if _sr >= 0.75:  # wrong-slot gate fired: hard cap risk high
                        summary_score = min(summary_score, 0.10)
                    else:  # soft blend: 75% WITNESS + 25% STAVE grounding
                        summary_score = 0.75 * summary_score + 0.25 * (1.0 - _sr)
            except Exception as _e:
                logger.debug("STAVE blend failed: %s", _e)

        return WitnessResult(
            output=output,
            certificates=certificates,
            summary_score=max(0.0, min(1.0, summary_score)),
            n_grounded=counts.get("grounded", 0),
            n_unsupported=counts.get("unsupported", 0),
            n_contradicted=counts.get("contradicted", 0),
            n_unknown=counts.get("unknown", 0),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def analyze_and_rewrite(
        self,
        context: str,
        output: str,
        *,
        retrieved: list[str] | None = None,
        mode: str = "audit",
    ) -> tuple[WitnessResult, WitnessRewrite]:
        """Analyze output and apply policy in one call.

        The deterministic path delegates both analysis and policy execution to
        Rust. The Python path remains for optional OpenAI NLI.
        """
        evidence_context = context
        if retrieved:
            evidence_context = context + "\n\n" + "\n\n".join(retrieved)

        if not self.use_nli and not self.force_python and _rust_witness_analyze is not None:
            try:
                payload = json.loads(_rust_witness_analyze(
                    evidence_context,
                    output,
                    mode,
                    self.profile,
                    self.support_threshold,
                    self.contradiction_threshold,
                    self.adequacy_threshold,
                ))
                return (
                    _witness_result_from_dict(payload["witness"]),
                    _witness_rewrite_from_dict(payload["policy"]),
                )
            except Exception as e:
                logger.debug("Rust WITNESS policy failed; falling back to Python: %s", e)

        result = self.analyze(evidence_context, output)
        return result, apply_witness_policy(
            output, result,
            mode=mode,
            profile=self.profile,
            thresholds=self.thresholds,
        )

    def _certify_claims_with_batch_nli(self, claims: list[Claim], context: str) -> list[Certificate]:
        prepared: list[tuple[Claim, list[EvidenceWindow], float]] = []
        for claim in claims:
            windows, adequacy = select_evidence_windows(context, claim.text)
            prepared.append((claim, windows, adequacy))

        verdicts = self._openai_nli_batch_check(context, prepared[: self.nli_max_claims])
        certificates: list[Certificate] = []
        for claim, _, _ in prepared:
            certificates.append(self._certify_claim(claim, context, nli_verdict=verdicts.get(claim.id)))
        return certificates

    def _openai_nli_batch_check(
        self,
        context: str,
        prepared: list[tuple[Claim, list[EvidenceWindow], float]],
    ) -> dict[int, NLIVerdict]:
        if not prepared:
            return {}

        items: list[dict[str, Any]] = []
        evidence_chunks: list[str] = []
        for claim, windows, adequacy in prepared:
            evidence = "\n".join(f"- {w.text}" for w in windows)[:1800]
            evidence_chunks.append(evidence)
            items.append(
                {
                    "id": claim.id,
                    "claim": claim.text,
                    "evidence": evidence,
                    "adequacy": round(adequacy, 4),
                }
            )
        prompt = (
            "Verify each claim using only its evidence. Return strict JSON with this schema: "
            '{"verdicts":[{"id":0,"verdict":"supported|contradicted|insufficient","confidence":0.0}]}'
            "\n\nItems:\n"
            + json.dumps(items, ensure_ascii=False)
        )
        self.nli_input_tokens_est += max(1, len(prompt) // 4)

        try:
            import openai

            client = openai.OpenAI(timeout=self.nli_timeout_s)
            started = time.perf_counter()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a conservative evidence verifier. Use only the provided evidence. "
                            "Choose supported only when the evidence directly entails the claim; choose "
                            "contradicted when the evidence directly conflicts; otherwise choose insufficient."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=min(900, 120 + 90 * len(items)),
                temperature=0.0,
            )
            self.nli_calls += 1
            raw = response.choices[0].message.content or "{}"
            self.nli_output_tokens_est += max(1, len(raw) // 4)
            if (time.perf_counter() - started) > self.nli_timeout_s:
                self.nli_timeouts += 1
            parsed = json.loads(raw)
            raw_verdicts = parsed.get("verdicts", [])
        except Exception as e:
            logger.debug("OpenAI WITNESS batch NLI failed; falling back to local PAV: %s", e)
            self.nli_fallbacks += 1
            return {
                claim.id: _local_pav(context, claim.text, windows, adequacy)
                for claim, windows, adequacy in prepared
            }

        verdicts: dict[int, NLIVerdict] = {}
        by_id = {claim.id: (claim, windows, adequacy) for claim, windows, adequacy in prepared}
        for item in (raw_verdicts if isinstance(raw_verdicts, list) else []):
            if not isinstance(item, dict):
                continue
            claim_id = int(item.get("id", -1))
            if claim_id not in by_id:
                continue
            _, windows, adequacy = by_id[claim_id]
            raw_label = str(item.get("verdict", "insufficient")).lower()
            confidence = float(item.get("confidence", 0.0) or 0.0)
            evidence = "\n".join(w.text for w in windows)
            if "contradict" in raw_label:
                verdicts[claim_id] = NLIVerdict("contradiction", max(0.82, confidence), evidence, adequacy)
            elif "support" in raw_label:
                verdicts[claim_id] = NLIVerdict("entailment", max(0.84, confidence), evidence, adequacy)
            else:
                verdicts[claim_id] = NLIVerdict("neutral", max(0.50, confidence), evidence, adequacy)

        for claim, windows, adequacy in prepared:
            verdicts.setdefault(claim.id, _local_pav(context, claim.text, windows, adequacy))
        return verdicts

    def nli_usage(self) -> dict[str, Any]:
        """Return bounded-cost NLI telemetry for proxy stats and audits."""
        input_cost = self.nli_input_tokens_est / 1000 * 0.00015
        output_cost = self.nli_output_tokens_est / 1000 * 0.00060
        return {
            "calls": self.nli_calls,
            "timeouts": self.nli_timeouts,
            "fallbacks": self.nli_fallbacks,
            "input_tokens_est": self.nli_input_tokens_est,
            "output_tokens_est": self.nli_output_tokens_est,
            "cost_usd_est": round(input_cost + output_cost, 6),
            "timeout_s": self.nli_timeout_s,
            "max_claims_per_batch": self.nli_max_claims,
        }

    def _certify_claim(
        self,
        claim: Claim,
        context: str,
        *,
        nli_verdict: NLIVerdict | None = None,
    ) -> Certificate:
        windows, adequacy = select_evidence_windows(context, claim.text)
        support_steps: list[ProofStep] = []
        contradiction_steps: list[ProofStep] = []
        question = _question_from_context(context)

        quote = _quote_support(context, claim.text)
        if quote > 0.0 and not question:
            support_steps.append(ProofStep("quote_support", f"quote_strength={quote:.2f}", quote))

        slot_support = _slot_support(context, claim.text, adequacy)
        if slot_support > 0.0 and not question:
            support_steps.append(ProofStep("slot_support", f"adequacy={adequacy:.2f}", slot_support))

        local = _local_pav(context, claim.text, windows, adequacy)
        if local.label == "entailment":
            support_steps.append(ProofStep("local_entailment", local.evidence[:180], local.confidence))
        elif local.label == "contradiction":
            contradiction_steps.append(ProofStep("local_contradiction", local.evidence[:180], local.confidence))

        if nli_verdict is not None:
            nli = nli_verdict
        elif self.use_nli and os.getenv("OPENAI_API_KEY"):
            nli = _openai_nli_check(context, claim.text, windows, adequacy, self.model)
        elif self.use_local_nli:
            # Local DeBERTa-v3-small NLI — zero API cost, ~30-80ms/claim on CPU.
            # Only run when local_pav verdict is uncertain (neutral) to keep
            # the average latency overhead acceptable.
            try:
                from .verifiers.local_nli import nli_score as _local_nli_score
                _evidence = "\n".join(w.text for w in windows) if windows else context[:1000]
                _label, _conf = _local_nli_score(_evidence, claim.text)
                nli = NLIVerdict(_label, _conf, _evidence[:180], adequacy)
            except Exception as _nli_e:
                logger.debug("Local NLI failed: %s", _nli_e)
                nli = None
        else:
            nli = None
        if nli is not None:
            if nli.label == "entailment":
                support_steps.append(ProofStep("nli_entailment", nli.evidence[:180], nli.confidence))
            elif nli.label == "contradiction":
                contradiction_steps.append(ProofStep("nli_contradiction", nli.evidence[:180], nli.confidence))

        support = max((step.strength for step in support_steps), default=0.0)
        contradiction = max((step.strength for step in contradiction_steps), default=0.0)

        # ── Continuous risk path (default for Python-only mode) ──
        # When NLI didn't provide a confident verdict, route through the
        # featurized risk model. This replaces the 4-bucket labels with
        # a continuous ρ ∈ [0,1] from a logistic on φ(c, C). The downstream
        # ThresholdSet (witness_calibration.py) maps ρ → action.
        use_continuous = nli is None or nli.label == "neutral"
        if use_continuous:
            from .witness_features import extract_features
            from .witness_risk_model import label_from_features_and_risk

            risk_model = _get_default_risk_model()
            # Build the evidence text for feature extraction. Critical
            # detail for QA: the context is often KNOWLEDGE + "\n\nQuestion:
            # ...", and verifying an answer-claim against the *question*
            # inflates lex overlap (the answer often shares keywords with
            # the question), letting hallucinated QA answers slip past.
            # Strip the trailing "Question: ..." block so feature
            # extraction only sees the knowledge — and pass the extracted
            # question into the QA-alignment feature (φ₉).
            evidence_text = (
                "\n".join(w.text for w in windows)
                if windows else context
            )
            question_text: str | None = None
            if "\n\nQuestion:" in context:
                parts = context.split("\n\nQuestion:", 1)
                knowledge_only = parts[0].strip()
                question_text = parts[1].strip() if len(parts) == 2 else None
                if knowledge_only:
                    evidence_text = knowledge_only
            features = extract_features(
                claim.text,
                evidence_text,
                adequacy=adequacy,
                question=question_text,
            )
            risk = risk_model.predict(features)
            # Novel-entity floor: raise risk when the claim introduces a
            # fabricated noun phrase (adjacent content words absent from the
            # evidence) on an otherwise-grounded subject. Conservative; never
            # lowers risk. Validated to not regress HaluEval-QA AUROC.
            risk = _novel_content_floor(claim.text, evidence_text, float(risk))
            if contradiction >= self.contradiction_threshold:
                label = "contradicted"
                risk = max(float(risk), 0.92)
            else:
                label = label_from_features_and_risk(features, risk)
            proof = (contradiction_steps + support_steps)[:6]
            if not proof:
                proof = [ProofStep(
                    "continuous_risk",
                    f"phi={features.as_dict()}",
                    float(risk),
                )]
            return Certificate(
                claim_id=claim.id,
                claim_text=claim.text,
                label=label,
                support_strength=support,
                contradiction_strength=contradiction,
                evidence_adequacy=adequacy,
                risk=float(risk),
                proof_path=proof,
            )

        # ── Discrete-bucket path (kept for NLI verdicts that are
        #     definitive: entailment or contradiction). Continuous risk
        #     wouldn't add information when NLI gave a confident answer. ──
        if contradiction >= self.contradiction_threshold:
            label = "contradicted"
            risk = 0.92
            proof = contradiction_steps + support_steps[:1]
        elif support >= self.support_threshold:
            label = "grounded"
            risk = max(0.05, 0.35 * (1.0 - support))
            proof = support_steps
        elif adequacy >= self.adequacy_threshold:
            label = "unsupported"
            risk = 0.80
            proof = support_steps + [ProofStep("evidence_adequacy", _top_window_text(windows), adequacy)]
        else:
            label = "unknown"
            risk = 0.55
            proof = [ProofStep("weak_evidence", _top_window_text(windows), adequacy)]

        return Certificate(
            claim_id=claim.id,
            claim_text=claim.text,
            label=label,
            support_strength=support,
            contradiction_strength=contradiction,
            evidence_adequacy=adequacy,
            risk=risk,
            proof_path=proof,
        )


def select_evidence_windows(context: str, claim: str, *, top_k: int = 4) -> tuple[list[EvidenceWindow], float]:
    claim_words = _content_words(claim)
    entities = _extract_entities(claim)
    numbers = _extract_numbers(claim)
    windows: list[EvidenceWindow] = []

    for text in _sentence_windows(context):
        lower = text.lower()
        words = _content_words(text)
        entity_cov = _coverage_fraction(entities, lower)
        number_cov = _coverage_fraction(numbers, lower)
        token_cov = len(claim_words & words) / max(len(claim_words), 1)
        adequacy = _weighted_adequacy(
            entity_cov=entity_cov,
            number_cov=number_cov,
            token_cov=token_cov,
            has_entities=bool(entities),
            has_numbers=bool(numbers),
        )
        score = 3.0 * entity_cov + 2.0 * number_cov + token_cov + adequacy
        windows.append(EvidenceWindow(text, score, adequacy, entity_cov, number_cov, token_cov))

    windows.sort(key=lambda w: (w.score, w.adequacy, len(w.text)), reverse=True)
    top = windows[:top_k]
    combined = "\n".join(w.text for w in top).lower()
    combined_words: set[str] = set()
    for window in top:
        combined_words |= _content_words(window.text)
    adequacy = _weighted_adequacy(
        entity_cov=_coverage_fraction(entities, combined),
        number_cov=_coverage_fraction(numbers, combined),
        token_cov=len(claim_words & combined_words) / max(len(claim_words), 1),
        has_entities=bool(entities),
        has_numbers=bool(numbers),
    )
    return top, max([w.adequacy for w in top] + [adequacy, 0.0])


def _openai_nli_check(
    context: str,
    claim: str,
    windows: list[EvidenceWindow],
    adequacy: float,
    model: str,
) -> NLIVerdict:
    evidence = "\n".join(f"- {w.text}" for w in windows)
    cache_key = (evidence[:1400], claim[:260])
    if cache_key in _NLI_CACHE:
        return _NLI_CACHE[cache_key]

    try:
        import openai

        client = openai.OpenAI()
        question = _question_from_context(context)
        if question:
            task = (
                f"Question:\n{question}\n\n"
                f"Candidate answer:\n{claim}\n\n"
                "Decide whether the evidence supports this candidate answer as the answer to the question."
            )
        else:
            task = f"Claim:\n{claim}"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Verify the claim using only the evidence. Reply with exactly one word: "
                        "supported, contradicted, or insufficient. Use insufficient for partial, "
                        "off-topic, or missing evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Evidence:\n{evidence[:1800]}\n\n{task}\n\nVerdict:",
                },
            ],
            max_tokens=5,
            temperature=0.0,
        )
        raw = (response.choices[0].message.content or "").strip().lower()
        if "contradict" in raw:
            verdict = NLIVerdict("contradiction", 0.86, evidence, adequacy)
        elif "support" in raw:
            verdict = NLIVerdict("entailment", 0.88, evidence, adequacy)
        else:
            verdict = NLIVerdict("neutral", 0.50, evidence, adequacy)
    except Exception:
        verdict = _local_pav(context, claim, windows, adequacy)

    _NLI_CACHE[cache_key] = verdict
    return verdict


def _local_pav(context: str, claim: str, windows: list[EvidenceWindow], adequacy: float) -> NLIVerdict:
    evidence = "\n".join(w.text for w in windows)
    comparative = _comparative_verdict(context, claim)
    if comparative:
        label, conf = comparative
        return NLIVerdict(label, conf, evidence, max(adequacy, 0.80))

    if _question_from_context(context):
        return NLIVerdict("neutral", 0.35, evidence, adequacy)

    quote = _quote_support(context, claim)
    if quote >= 0.82:
        return NLIVerdict("entailment", min(0.85, quote), evidence, max(adequacy, 0.75))

    claim_nums = [n for n in _extract_numbers(claim) if n]
    evidence_nums = [n for n in _extract_numbers(evidence) if n]
    if claim_nums and adequacy >= 0.72:
        if all(n in evidence_nums for n in claim_nums):
            return NLIVerdict("entailment", 0.68, evidence, adequacy)
        if evidence_nums:
            return NLIVerdict("contradiction", 0.66, evidence, adequacy)

    if adequacy >= 0.88 and _slot_support(context, claim, adequacy) >= 0.70:
        return NLIVerdict("entailment", 0.66, evidence, adequacy)
    return NLIVerdict("neutral", 0.35, evidence, adequacy)


def _quote_support(context: str, claim: str) -> float:
    norm_context = _normalize_text(context)
    norm_claim = _normalize_text(claim)
    if not norm_claim:
        return 0.0
    if norm_claim in norm_context:
        return 0.95 if len(norm_claim) > 80 else 0.82
    if len(norm_claim) <= 80 and re.search(rf"\b{re.escape(norm_claim)}\b", norm_context):
        return 0.78

    match = SequenceMatcher(None, norm_context, norm_claim, autojunk=True).find_longest_match(
        0, len(norm_context), 0, len(norm_claim)
    )
    contiguous = match.size / max(len(norm_claim), 1)
    if contiguous >= 0.65 and match.size >= 30:
        return min(0.80, 0.45 + contiguous * 0.45)
    return 0.0


def _slot_support(context: str, claim: str, adequacy: float) -> float:
    words = _content_words(claim)
    entities = _extract_entities(claim)
    numbers = _extract_numbers(claim)
    lower = context.lower()
    token_cov = len(words & _content_words(context)) / max(len(words), 1)
    entity_cov = _coverage_fraction(entities, lower)
    number_cov = _coverage_fraction(numbers, lower)
    if numbers and number_cov < 1.0:
        return 0.0
    if entities and entity_cov < 0.65:
        return 0.0
    return min(0.74, 0.25 + 0.35 * adequacy + 0.20 * token_cov)


def _comparative_verdict(context: str, claim: str) -> tuple[str, float] | None:
    question = _question_from_context(context)
    candidates = _candidate_entities(question)
    if len(candidates) != 2:
        return None

    clean_context = re.sub(r"(?:^|\n)Question:\s*.+", "", context, flags=re.IGNORECASE | re.DOTALL)
    values = {candidate: _numbers_near(clean_context, candidate) for candidate in candidates}
    if any(not nums for nums in values.values()):
        return None

    q_lower = question.lower()
    expected: str | None = None
    if any(term in q_lower for term in ("first", "earlier", "older", "started", "founded")):
        expected = min(candidates, key=lambda c: min(values[c]))
    elif any(term in q_lower for term in ("more", "most", "larger", "higher")):
        expected = max(candidates, key=lambda c: max(values[c]))
    if not expected:
        return None

    claim_lower = claim.lower()
    if expected.lower() in claim_lower:
        return ("entailment", 0.76)
    if any(c.lower() in claim_lower for c in candidates if c != expected):
        return ("contradiction", 0.78)
    return None


def _candidate_claim_segments(text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    seen: set[str] = set()

    def add(start: int, raw: str) -> None:
        cleaned = _clean_claim_segment(raw)
        key = _normalize_text(cleaned)
        if cleaned and key and key not in seen:
            seen.add(key)
            out.append((start, cleaned))

    for start, sentence in _split_sentences(text):
        add(start, sentence)
        for part_start, part in _split_compound_claims(start, sentence):
            add(part_start, part)

    offset = 0
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        trimmed = line.strip()
        if trimmed:
            leading = line.find(trimmed)
            start = offset + max(leading, 0)
            cleaned = _clean_claim_segment(trimmed)
            if _is_list_or_table_row(trimmed) or _looks_like_code_claim(cleaned):
                add(start, cleaned)
            for part_start, part in _split_compound_claims(start, cleaned):
                add(part_start, part)
        offset += len(raw_line)

    out.sort(key=lambda item: (item[0], -len(item[1])))
    return out


def _clean_claim_segment(segment: str) -> str:
    s = segment.strip().strip("`").strip()
    s = re.sub(r"^\s*[-*+]\s+", "", s)
    s = re.sub(r"^\s*\d+\.\s+", "", s)
    if s.startswith("|") and s.endswith("|"):
        cells = [
            cell.strip()
            for cell in s.strip("|").split("|")
            if cell.strip() and not re.fullmatch(r"[-:\s]+", cell.strip())
        ]
        if len(cells) >= 2:
            s = " | ".join(cells)
    return s.strip(" -*:")


def _is_list_or_table_row(segment: str) -> bool:
    s = segment.strip()
    return bool(
        s.startswith(("- ", "* ", "+ ", "|"))
        or re.match(r"^\d+\.\s+", s)
    )


def _split_compound_claims(start: int, segment: str) -> list[tuple[int, str]]:
    parts: list[tuple[int, str]] = []
    for match in re.finditer(r";\s+|\s+and\s+", segment, flags=re.IGNORECASE):
        left = segment[: match.start()].strip()
        right = segment[match.end() :].strip()
        if len(left) >= 8 and len(right) >= 8 and _has_fact_signal(left) and _has_fact_signal(right):
            parts.append((start, left))
            parts.append((start + match.end(), right))
    return parts


def _is_claim_like(text: str) -> bool:
    s = text.strip()
    if len(s) < 2 or _is_question_like(s) or s.startswith("```"):
        return False
    if s.lower() in {"json", "python", "rust", "javascript", "typescript"}:
        return False
    return _looks_like_code_claim(s) or _has_fact_signal(s) or _contains_claim_verb(s)


def _has_fact_signal(text: str) -> bool:
    return bool(_extract_entities(text) or _extract_numbers(text) or _looks_like_code_claim(text))


def _contains_claim_verb(text: str) -> bool:
    lower = f" {text.lower()} "
    return any(
        marker in lower
        for marker in (
            " is ", " are ", " was ", " were ", " has ", " have ", " had ", " uses ",
            " returns ", " contains ", " supports ", " requires ", " equals ", " means ",
            " runs ", " fails ",
        )
    )


def _looks_like_code_claim(text: str) -> bool:
    lower = text.lower()
    return bool(
        "()" in text
        or "_" in text
        or "::" in text
        or any(marker in lower for marker in ("function", "method", "class ", "module ", "import "))
        or any(ext in lower for ext in (".py", ".rs", ".js", ".ts"))
    )


def _classify_claim(text: str) -> str:
    if _looks_like_code_claim(text):
        return "code_ref"
    if _extract_numbers(text):
        return "quantity"
    return "proposition"


def _normalize_profile(profile: str, output: str) -> str:
    normalized = (profile or "auto").strip().lower().replace("-", "_")
    if normalized == "summarization":
        normalized = "summary"
    if normalized in {"code", "rag", "qa", "benchmark_qa", "summary", "chat", "dialogue"}:
        return normalized
    lower = output.lower()
    if "```" in output or any(ext in lower for ext in (".py", ".rs", ".js", ".ts")) or "function" in lower:
        return "code"
    if len(output) > 900 or sum(1 for line in output.splitlines() if line.strip().startswith("- ")) >= 3:
        return "summary"
    return "rag"


def _should_suppress(cert: Certificate, profile: str) -> bool:
    if cert.label == "grounded":
        return False
    if cert.label == "contradicted":
        return True
    if cert.label == "unsupported":
        return profile not in {"chat", "dialogue"}
    if cert.label == "unknown":
        return profile in {"code", "rag", "qa", "benchmark_qa"} or cert.risk >= 0.75
    return False


def _split_sentences(text: str) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    offset = 0
    for part in re.split(r"(?<=[.!?])\s+|\n+", text):
        part = part.strip()
        if not part:
            continue
        pos = text.find(part, offset)
        if pos < 0:
            pos = offset
        result.append((pos, part))
        offset = pos + len(part)
    return result


def _is_question_like(sentence: str) -> bool:
    s = sentence.strip()
    if s.endswith("?"):
        return True
    first = re.match(r"[A-Za-z]+", s)
    return bool(first and first.group(0).lower() in _QUESTION_STARTERS)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.%'-]", " ", text.lower())).strip()


def _remove_claim_text(output: str, claim: str) -> str:
    idx = output.find(claim)
    if idx >= 0:
        before = output[:idx].rstrip()
        after = output[idx + len(claim):].lstrip()
        return (before + "\n" + after).strip()

    # Fallback for distillation or provider formatting changes: remove the
    # closest full sentence if it contains most of the claim's content words.
    claim_words = _content_words(claim)
    if not claim_words:
        return output
    rewritten = output
    for start, sentence in sorted(_split_sentences(output), reverse=True):
        words = _content_words(sentence)
        overlap = len(claim_words & words) / max(len(claim_words), 1)
        if overlap >= 0.72:
            rewritten = rewritten[:start].rstrip() + "\n" + rewritten[start + len(sentence):].lstrip()
            break
    return rewritten


def _cleanup_rewritten_output(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?m)^\s*[-*]\s*$\n?", "", text)
    return text.strip()


def _content_words(text: str) -> set[str]:
    return {
        word.lower()
        for word in re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{3,}\b", text)
        if word.lower() not in _STOPWORDS
    }


def _extract_numbers(text: str) -> list[str]:
    numbers = [re.sub(r"^0+", "", m.group().replace(",", "").replace("$", "")) for m in re.finditer(r"\$?\d[\d,]*(?:\.\d+)?%?", text)]
    for word, value in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", text, flags=re.IGNORECASE):
            numbers.append(str(value))
    return [n for n in numbers if n]


def _extract_entities(text: str) -> list[str]:
    entities: list[str] = []
    for match in re.finditer(r'"([^"]{2,90})"', text):
        entities.append(match.group(1).strip())
    for match in re.finditer(r"\b[A-Z][A-Z0-9]{1,}\b", text):
        entities.append(match.group(0))
    for match in re.finditer(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*){0,4}\b", text):
        value = match.group(0).strip()
        if len(value) >= 3 and value.lower() not in _STOPWORDS:
            entities.append(value)

    seen: set[str] = set()
    unique: list[str] = []
    for entity in entities:
        key = re.sub(r"\W+", "", entity).lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(entity)
    return unique


def _sentence_windows(context: str, *, max_chars: int = 900) -> list[str]:
    raw = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", context) if len(s.strip()) > 4]
    windows: list[str] = []
    for sentence in raw:
        if len(sentence) <= max_chars:
            windows.append(sentence)
        else:
            windows.extend(sentence[i : i + max_chars].strip() for i in range(0, len(sentence), max_chars))
    return [w for w in windows if w] or [context[:max_chars]]


def _coverage_fraction(items: list[str], haystack_lower: str) -> float:
    if not items:
        return 0.0
    return sum(1 for item in items if item and item.lower() in haystack_lower) / len(items)


def _weighted_adequacy(
    *,
    entity_cov: float,
    number_cov: float,
    token_cov: float,
    has_entities: bool,
    has_numbers: bool,
) -> float:
    weights: list[tuple[float, float]] = []
    if has_entities:
        weights.append((0.45, entity_cov))
    if has_numbers:
        weights.append((0.35, number_cov))
    weights.append((0.20 if has_entities or has_numbers else 1.0, token_cov))
    denom = sum(w for w, _ in weights)
    return max(0.0, min(1.0, sum(w * value for w, value in weights) / max(denom, 1e-9)))


def _question_from_context(context: str) -> str:
    match = re.search(r"(?:^|\n)Question:\s*(.+)", context, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _candidate_entities(question: str) -> list[str]:
    if " or " not in question.lower():
        return []
    q = question.strip().rstrip("?")
    tail = q.rsplit(",", 1)[-1] if "," in q else q
    for marker in (" first ", " more ", " older ", " earlier ", " larger ", " higher "):
        idx = tail.lower().find(marker)
        if idx >= 0:
            tail = tail[idx + len(marker) :]
            break
    return [p.strip(" .?\"'") for p in re.split(r"\s+or\s+", tail, flags=re.IGNORECASE) if len(p.strip()) > 1]


def _numbers_near(context: str, entity: str, *, radius: int = 260) -> list[int]:
    nums: list[int] = []
    lower = context.lower()
    key = entity.lower()
    start = 0
    while True:
        idx = lower.find(key, start)
        if idx < 0:
            break
        window = context[max(0, idx - radius) : idx + len(entity) + radius]
        nums.extend(int(raw) for raw in re.findall(r"\b\d{1,4}\b", window))
        for word, value in _NUMBER_WORDS.items():
            if re.search(rf"\b{word}\b", window, flags=re.IGNORECASE):
                nums.append(value)
        start = idx + len(entity)
    return nums


def _top_window_text(windows: list[EvidenceWindow]) -> str:
    return windows[0].text[:200] if windows else ""


__all__ = [
    "Certificate",
    "Claim",
    "EvidenceWindow",
    "ProofStep",
    "WitnessAnalyzer",
    "WitnessRewrite",
    "WitnessResult",
    "apply_witness_policy",
    "extract_claims",
    "format_witness_report",
    "select_evidence_windows",
]
