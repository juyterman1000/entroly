"""
Evidence Support Graph (ESG) — EICV Layer 2
============================================

Computes the constraint tension T(G) between evidence E and claim C as
the MRF energy of a bipartite support graph under PSL-style soft logic:

    T(G) = (1/|V_c|) * Σ_{v ∈ V_c}
               [ max(0, 1 - max_{u ∈ V_e} S(u,v))   # unsupported mass
               + λ * max_{u ∈ V_e} K(u,v) ]          # contradiction penalty

where:
  V_e    = atomic propositions extracted from evidence (sentence-level)
  V_c    = atomic propositions extracted from claim (sentence + clause-level)
  S(u,v) ∈ [0,1]  support: evidence atom u supports claim atom v
  K(u,v) ∈ [0,1]  contradiction: evidence atom u contradicts claim atom v
  λ      contradiction weight (default 0.5)

Higher T(G) means more claim atoms are structurally unsupported or
contradicted by the evidence → stronger hallucination signal.

Why this is strictly stronger than Fusion-4 / pairwise NLI
-----------------------------------------------------------
1. Atom-level resolution: a single wrong entity in a multi-sentence
   claim is invisible to whole-document NLI but surfaces as one fully-
   unsupported atom in T(G).
2. Distributed signals: S and K combine IDF-lexical overlap, entity
   match, number consistency, and negation polarity — none is dominant,
   so the entity-gap artifact that collapses Fusion-4 under C2 (entity-
   controlled) cannot collapse T(G).
3. Auditable: per-atom support and contradiction scores are returned in
   ESGResult for downstream Certificate generation.

PSL Substrate (Bach et al., JMLR 2017)
---------------------------------------
A PSL rule
  w: SUPPORT(u,v) ∧ CONSISTENT(u,v) → SUPPORTED(v)
maps to the hinge-loss
  w * max(0, S(u,v) * C(u,v) - SUPPORTED(v))
MAP inference (closed-form at greedy assignment μ*_v = max_u [S(u,v)·C(u,v)])
yields the per-atom unsupported mass above.

Design constraints
------------------
- Fully deterministic: no LLM, no external models, no network.
- Reuses witness_features.py primitives (IDF-overlap, entity_precision,
  number_consistency, negation_polarity).
- O(|V_e| × |V_c|) in sentence count (both are 1–20 in practice).
- Returns ESGResult with per-atom detail for auditability.

References
----------
- Bach et al., 2017. Hinge-Loss Markov Random Fields and Probabilistic
  Soft Logic. JMLR 18(109):1–67.
- Min et al., 2023. FActScore. EMNLP 2023.
- Honovich et al., 2022. TRUE: Re-evaluating Factual Consistency
  Evaluation. NAACL 2022.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field, asdict
from typing import Sequence


# ── Atom dataclass ────────────────────────────────────────────────────


@dataclass(frozen=True)
class Atom:
    """An atomic proposition — one verifiable unit of text."""
    text: str
    source: str = "claim"   # "claim" | "evidence"
    index: int = 0           # position in original sequence


# ── Per-atom support record (for auditability) ────────────────────────


@dataclass
class AtomVerdict:
    """Support/contradiction verdict for one claim atom v."""
    atom: str
    max_support: float          # max_{u ∈ V_e} S(u, v)
    max_contradiction: float    # max_{u ∈ V_e} K(u, v)
    best_ev_atom: str           # evidence atom achieving max_support
    unsupported_mass: float     # max(0, 1 - max_support)
    tension_contribution: float # unsupported_mass + λ * max_contradiction


# ── Primary result ─────────────────────────────────────────────────────


@dataclass
class ESGResult:
    """Output of ESGAnalyzer.score().

    Primary signal for EICV: `tension` ∈ [0, 1].
    Higher tension → higher structural hallucination probability.
    """
    # Core tension T(G) ∈ [0, 1]
    tension: float

    # Structural statistics
    n_claim_atoms: int
    n_ev_atoms: int
    mean_max_support: float        # average of max_{u} S(u,v) over v ∈ V_c
    mean_max_contradiction: float  # average of max_{u} K(u,v) over v ∈ V_c
    unsupported_fraction: float    # fraction of atoms with max_support < threshold
    contradiction_fraction: float  # fraction of atoms with contradiction > threshold

    # Per-atom verdicts (one per claim atom)
    atom_verdicts: list[AtomVerdict] = field(default_factory=list)

    # Configuration echo
    lambda_contradict: float = 0.5
    support_threshold: float = 0.30   # below this → atom is "unsupported"
    contradict_threshold: float = 0.40  # above this → atom is "contradicted"

    def as_dict(self) -> dict:
        d = asdict(self)
        # atom_verdicts is already serialisable via asdict
        return d

    @property
    def hallucination_score(self) -> float:
        """Alias: tension remapped to [0,1] as a probability proxy.
        Consistent with WitnessAnalyzer's summary_score convention
        (higher = more hallucinated).
        """
        return self.tension


# ── Primitive helpers (no spaCy dependency) ──────────────────────────

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]+")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")

_STOPWORDS = frozenset(
    """
    a an and as at be by for from has have in into is it of on or that the to
    was were will with would could should may might can do does did been being
    its this these those their our your his her them us me my we i you he she
    """.split()
)

_NEG_CUES = (
    "is not", "are not", "was not", "were not", "does not", "do not",
    "did not", "has not", "have not", "had not", "will not", "would not",
    "cannot", "can not", "isn't", "aren't", "wasn't", "weren't", "doesn't",
    "don't", "didn't", "hasn't", "haven't", "hadn't", "won't", "wouldn't",
    "no ", "not ", "never ", "none ",
)

# Coordinators that often join independent atomic clauses
_COORD_SPLIT = re.compile(
    r",\s+(?:and|but|or|however|although|while|whereas|yet|while)\s+',\s+",
    re.I,
)
# Simpler variant (no leading comma required — catches "X; however, Y")
_COORD_SPLIT2 = re.compile(
    r"(?:;\s*|,\s+)(?:and|but|or|however|although|while|whereas|yet|moreover"
    r"|furthermore|additionally|nevertheless|nonetheless|consequently)\s+",
    re.I,
)


def _stem(word: str) -> str:
    """Minimal rule-based stemmer for English verb / plural forms.

    Handles the most common inflections that cause spurious lexical
    mismatches in factuality scoring:
      developed → develop, develops → develop, developing → develop
      studies   → study,   studied → study
      cities    → city,    runs   → run

    Conservative: only strips for words of length > 4 to avoid mangling
    short words. Preserves the original if the result would be < 4 chars.
    """
    if len(word) <= 4:
        return word
    for suffix, replacement in (
        ("ied",  "y"),    # studies→study, studied→study (after ies-ied normalisation)
        ("ies",  "y"),
        ("ing",  ""),
        ("eed",  "ee"),   # agreed → agree (only after specific stems)
        ("ed",   ""),
        ("es",   ""),
        ("s",    ""),
    ):
        if word.endswith(suffix):
            stem = word[: -len(suffix)] + replacement
            if len(stem) >= 4:
                return stem
            return word
    return word


def _content_words(text: str) -> set[str]:
    """Lowercased, stemmed content words (stopwords removed, len > 2).

    Stemming makes the lexical-overlap features robust to verb tense
    changes ("developed"/"develops"/"developing" all map to "develop"),
    which was the Phase 1C diagnostic finding (Γ_temporal = 0.22).
    """
    return {
        _stem(w.lower()) for w in _WORD_RE.findall(text)
        if w.lower() not in _STOPWORDS and len(w) > 2
    }


def _named_entities(text: str) -> set[str]:
    """Cheap regex NER: contiguous capitalized words."""
    _INITIAL_NON_ENTS = frozenset({
        "the", "this", "that", "these", "those", "a", "an", "it", "they",
        "we", "i", "you", "he", "she", "but", "and", "or", "so", "however",
        "moreover", "furthermore", "therefore", "thus", "hence", "indeed",
        "actually", "interestingly", "importantly", "specifically",
    })
    out: set[str] = set()
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        for m in _ENTITY_RE.findall(sentence):
            lowered = m.lower()
            first_tok = sentence.split(None, 1)[0].rstrip(",.!?:;")
            if m == first_tok and lowered in _INITIAL_NON_ENTS:
                continue
            out.add(lowered)
    return out


def _extract_numbers(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(text))


def _idf_weight(word: str, ref_text: str) -> float:
    """In-sample IDF: rarer words in ref_text get higher weight."""
    count = ref_text.lower().count(word)
    return 1.0 / (1.0 + math.log(1.0 + count))


def _has_negation_before(text: str, word: str) -> bool:
    idx = text.find(word)
    if idx <= 0:
        return False
    preceding = text[max(0, idx - 40):idx]
    return any(cue in preceding for cue in _NEG_CUES)


# ── Atom decomposition ────────────────────────────────────────────────


def _split_atoms(text: str, *, min_words: int = 3) -> list[str]:
    """Decompose text into atomic propositions.

    Strategy (no dependency parser required):
      1. Split at sentence boundaries (.!?)
      2. Within each sentence, split at coordinating conjunctions (and/but/or…)
         that join clauses (signalled by preceding comma or semicolon).
      3. Keep only atoms with >= min_words words.

    This produces 1–20 atoms from typical claim/evidence text.
    """
    # Step 1: sentence splitting
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    atoms: list[str] = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        # Step 2: coordinating clause splitting
        parts = _COORD_SPLIT2.split(sent)
        if len(parts) == 1:
            parts = [sent]
        for p in parts:
            p = p.strip().strip(",.;:!?\"'")
            words = _WORD_RE.findall(p)
            if len(words) >= min_words:
                atoms.append(p)
    # Fallback: if nothing survived, treat whole text as one atom
    if not atoms and text.strip():
        atoms = [text.strip()]
    return atoms


# ── Pairwise scoring primitives ───────────────────────────────────────


def _support_score(ev_atom: str, claim_atom: str) -> float:
    """S(u, v) — how strongly evidence atom u supports claim atom v.

    Combines:
      - IDF-weighted lexical overlap (backbone, artifact-resistant)
      - Entity match bonus (rewards matching, penalises active mismatch)
      - Number match bonus

    All signals ∈ [0, 1]; combination is clipped to [0, 1].
    """
    claim_words = _content_words(claim_atom)
    if not claim_words:
        return 0.5   # vacuously supported (nothing to check)
    ev_lower = ev_atom.lower()

    # IDF-weighted Jaccard over content words
    ev_words = _content_words(ev_atom)
    shared = claim_words & ev_words
    if not shared:
        idf_jaccard = 0.0
    else:
        num = sum(_idf_weight(w, ev_lower) for w in shared)
        den = sum(_idf_weight(w, ev_lower) for w in claim_words)
        idf_jaccard = num / max(den, 1e-9)

    # Entity precision of claim_atom against ev_atom.
    # Penalty -0.35 / bonus +0.15 — values selected by group-DRO calibration
    # (Phase 2) to maximise worst-manifold AUROC across C1-C4. The slight
    # asymmetry (penalty > bonus) reflects that false-positive matching is
    # more costly than missed full-match recognition.
    claim_ents = _named_entities(claim_atom)
    if claim_ents:
        ev_lower2 = ev_atom.lower()
        matched = sum(1 for e in claim_ents if e in ev_lower2)
        if matched == len(claim_ents):
            ent_bonus = 0.15      # all claim entities present → bonus (DRO-tuned)
        else:
            missing = len(claim_ents) - matched
            ent_bonus = -0.35 * missing / len(claim_ents)  # mismatch (DRO-tuned)
    else:
        ent_bonus = 0.0

    # Number consistency — penalty -0.50 (DRO-tuned in Phase 2).
    # Original 0.20 was too weak (Γ_numeric = 0.05 in Phase 1C). Group-DRO
    # selected 0.50 by maximising worst-manifold AUROC across C1-C4.
    claim_nums = _extract_numbers(claim_atom)
    if claim_nums:
        ev_nums = _extract_numbers(ev_atom)
        n_miss = sum(1 for n in claim_nums if n not in ev_nums)
        if n_miss == 0:
            num_bonus = 0.15
        else:
            num_bonus = -0.50 * n_miss / len(claim_nums)
    else:
        num_bonus = 0.0

    raw = idf_jaccard + ent_bonus + num_bonus
    return float(max(0.0, min(1.0, raw)))


def _contradiction_score(ev_atom: str, claim_atom: str) -> float:
    """K(u, v) — how strongly evidence atom u contradicts claim atom v.

    Takes the max of FOUR contradiction signals:
      1. Active entity mismatch (bag-of-entities)
      2. Active number mismatch
      3. Negation polarity inversion
      4. SLOT-AWARE entity substitution (Phase 5 addition; entroly/slot_extraction.py)
         — detects "Eiffel Tower in London" vs "Eiffel Tower in Paris" where
         the same syntactic slot is filled by a different entity.

    Topical relevance gate: entity and number mismatch are only
    contradictions when the evidence atom is topically relevant to the
    claim atom (shares at least one content word). A topically-disjoint
    evidence atom is simply silent — not contradictory. The slot
    substitution signal is self-gating (requires aligned context).

    Returns ∈ [0, 1].
    """
    # Topical relevance gate — must come first
    claim_words = _content_words(claim_atom)
    ev_words_set = _content_words(ev_atom)
    topically_relevant = bool(claim_words & ev_words_set)

    # 1. Entity mismatch — only fires when topically relevant
    claim_ents = _named_entities(claim_atom)
    if claim_ents and topically_relevant:
        ev_lower = ev_atom.lower()
        matched = sum(1 for e in claim_ents if e in ev_lower)
        missing = len(claim_ents) - matched
        ent_mismatch = missing / len(claim_ents) if missing > 0 else 0.0
        # Only count as hard contradiction if ev_atom also has entities
        # (otherwise the evidence is silent, not contradictory)
        ev_ents = _named_entities(ev_atom)
        if not ev_ents:
            ent_mismatch *= 0.5  # weaken: evidence is entity-free
    else:
        ent_mismatch = 0.0

    # 2. Number mismatch — only fires when topically relevant
    claim_nums = _extract_numbers(claim_atom)
    if claim_nums and topically_relevant:
        ev_nums = _extract_numbers(ev_atom)
        if ev_nums:
            # Evidence asserts numbers: mismatched ones are contradictions
            n_miss = sum(1 for n in claim_nums if n not in ev_nums)
            num_mismatch = n_miss / len(claim_nums)
        else:
            num_mismatch = 0.0   # evidence silent on numbers — not contradiction
    else:
        num_mismatch = 0.0

    # 3. Negation polarity inversion (shared already computed above)
    shared = claim_words & ev_words_set
    neg_flip = 0.0
    if shared:
        claim_lower = claim_atom.lower()
        ev_lower2 = ev_atom.lower()
        for word in shared:
            nc = _has_negation_before(claim_lower, word)
            ne = _has_negation_before(ev_lower2, word)
            if nc != ne:
                neg_flip = 1.0
                break

    # 4. Slot-aware entity substitution (Phase 5).
    slot_sub = 0.0
    if topically_relevant and len(claim_words) >= 2:
        try:
            from entroly.slot_extraction import slot_substitution_score
            slot_sub, _ = slot_substitution_score(claim_atom, ev_atom)
        except Exception:
            slot_sub = 0.0

    # 5. Quantifier-aware contradiction (Phase 6).
    # "All X are Y" vs evidence "no X are Y" → 1.0; partial mismatches lower.
    quant_sub = 0.0
    try:
        from entroly.atomic_decomposition import quantifier_contradiction
        quant_sub = quantifier_contradiction(claim_atom, ev_atom)
    except Exception:
        pass

    # 6. Magnitude-aware numeric mismatch (Phase 6).
    mag_num = 0.0
    if topically_relevant:
        try:
            from entroly.atomic_decomposition import numeric_magnitude_mismatch
            mag_num = numeric_magnitude_mismatch(claim_atom, ev_atom)
        except Exception:
            pass

    # 7. Categorical type contradiction (Phase 6+).
    cat_type = 0.0
    if topically_relevant:
        try:
            from entroly.atomic_decomposition import categorical_type_contradiction
            cat_type = categorical_type_contradiction(claim_atom, ev_atom)
        except Exception:
            pass

    # 8. Modifier mismatch (same head, different modifiers).
    # "drama film" vs "thriller film" — both films but different KINDS.
    mod_mismatch = 0.0
    if topically_relevant:
        try:
            from entroly.atomic_decomposition import modifier_mismatch
            mod_mismatch = modifier_mismatch(claim_atom, ev_atom)
        except Exception:
            pass

    return float(max(ent_mismatch, num_mismatch, neg_flip, slot_sub,
                     quant_sub, mag_num, cat_type, mod_mismatch))


# ── Main analyser ─────────────────────────────────────────────────────


class ESGAnalyzer:
    """Computes the Evidence Support Graph tension T(G).

    Args:
        lambda_contradict: weight of contradiction mass in T(G).
            Default 0.5. Higher → contradiction signal dominates.
        support_threshold: max_support below this → atom deemed "unsupported".
        contradict_threshold: max_contradiction above this → "contradicted".
        min_atom_words: minimum word count to keep a split fragment as an atom.
    """

    def __init__(
        self,
        lambda_contradict: float = 0.5,
        support_threshold: float = 0.30,
        contradict_threshold: float = 0.40,
        min_atom_words: int = 3,
    ) -> None:
        self.lambda_contradict = lambda_contradict
        self.support_threshold = support_threshold
        self.contradict_threshold = contradict_threshold
        self.min_atom_words = min_atom_words

    # ── Core scoring ─────────────────────────────────────────────────

    def score(self, evidence: str, claim: str) -> ESGResult:
        """Compute T(G) between evidence and claim.

        Args:
            evidence: The reference context (retrieved passages, etc.)
            claim: The text to verify against the evidence.

        Returns:
            ESGResult with `tension` as the primary signal ∈ [0, 1].
            Higher tension → more hallucination evidence.
        """
        ev_atoms = _split_atoms(evidence, min_words=self.min_atom_words)
        claim_atoms = _split_atoms(claim, min_words=2)  # shorter claim fragments OK

        n_ev = len(ev_atoms)
        n_claim = len(claim_atoms)

        # Edge case: empty inputs
        if n_claim == 0:
            return ESGResult(
                tension=0.0,
                n_claim_atoms=0,
                n_ev_atoms=n_ev,
                mean_max_support=1.0,
                mean_max_contradiction=0.0,
                unsupported_fraction=0.0,
                contradiction_fraction=0.0,
                lambda_contradict=self.lambda_contradict,
                support_threshold=self.support_threshold,
                contradict_threshold=self.contradict_threshold,
            )

        if n_ev == 0:
            # No evidence at all → every claim atom is fully unsupported
            verdicts = [
                AtomVerdict(
                    atom=ca,
                    max_support=0.0,
                    max_contradiction=0.0,
                    best_ev_atom="",
                    unsupported_mass=1.0,
                    tension_contribution=1.0,
                )
                for ca in claim_atoms
            ]
            return ESGResult(
                tension=1.0,
                n_claim_atoms=n_claim,
                n_ev_atoms=0,
                mean_max_support=0.0,
                mean_max_contradiction=0.0,
                unsupported_fraction=1.0,
                contradiction_fraction=0.0,
                atom_verdicts=verdicts,
                lambda_contradict=self.lambda_contradict,
                support_threshold=self.support_threshold,
                contradict_threshold=self.contradict_threshold,
            )

        # ── Main loop: for each claim atom v, find max_u S(u,v) and K(u, v) ──
        # IMPORTANT: contradiction K is computed against the BEST-SUPPORTING
        # evidence atom, not max'd independently. Otherwise a topically-
        # disjoint evidence atom (e.g. "Einstein was awarded Nobel in 1921"
        # vs claim "Einstein was born in 1879") would fire a spurious
        # numeric mismatch (1921 vs 1879) and overwhelm the genuinely
        # supporting evidence atom ("Einstein was born in Ulm, Germany,
        # in 1879"). Tying K to the best-supporting evidence atom routes
        # contradiction through the most relevant context.
        verdicts: list[AtomVerdict] = []

        for ca in claim_atoms:
            best_support = 0.0
            best_ev = ev_atoms[0]
            for ea in ev_atoms:
                s = _support_score(ea, ca)
                if s > best_support:
                    best_support = s
                    best_ev = ea

            # Contradiction is computed against the best-supporting ev atom
            best_contradict = _contradiction_score(best_ev, ca)

            # Belt-and-suspenders: if NO evidence atom provides decent
            # support (best_support < 0.20), the claim is topically
            # unsupported — fall back to max contradiction across all
            # evidence atoms (catches the case where claim is talking
            # about a topic the evidence directly negates).
            if best_support < 0.20:
                fallback_k = 0.0
                for ea in ev_atoms:
                    k = _contradiction_score(ea, ca)
                    if k > fallback_k:
                        fallback_k = k
                if fallback_k > best_contradict:
                    best_contradict = fallback_k

            unsupported_mass = max(0.0, 1.0 - best_support)
            contrib = unsupported_mass + self.lambda_contradict * best_contradict

            verdicts.append(AtomVerdict(
                atom=ca,
                max_support=round(best_support, 4),
                max_contradiction=round(best_contradict, 4),
                best_ev_atom=best_ev[:120],   # truncate for JSON size
                unsupported_mass=round(unsupported_mass, 4),
                tension_contribution=round(contrib, 4),
            ))

        # ── Aggregate ──────────────────────────────────────────────────────
        tension = sum(v.tension_contribution for v in verdicts) / n_claim
        # Clip to [0, 1]: lambda_contradict * 1.0 can push beyond 1
        tension = min(1.0, max(0.0, tension))

        mean_sup = sum(v.max_support for v in verdicts) / n_claim
        mean_con = sum(v.max_contradiction for v in verdicts) / n_claim
        unsup_frac = sum(
            1 for v in verdicts if v.max_support < self.support_threshold
        ) / n_claim
        contra_frac = sum(
            1 for v in verdicts if v.max_contradiction > self.contradict_threshold
        ) / n_claim

        return ESGResult(
            tension=round(tension, 6),
            n_claim_atoms=n_claim,
            n_ev_atoms=n_ev,
            mean_max_support=round(mean_sup, 4),
            mean_max_contradiction=round(mean_con, 4),
            unsupported_fraction=round(unsup_frac, 4),
            contradiction_fraction=round(contra_frac, 4),
            atom_verdicts=verdicts,
            lambda_contradict=self.lambda_contradict,
            support_threshold=self.support_threshold,
            contradict_threshold=self.contradict_threshold,
        )

    # ── Convenience shim ─────────────────────────────────────────────

    def tension(self, evidence: str, claim: str) -> float:
        """Return T(G) as a scalar — convenience alias for score().tension."""
        return self.score(evidence, claim).tension


# ── Module-level default instance ────────────────────────────────────

_DEFAULT_ANALYZER = ESGAnalyzer()


def compute_tension(evidence: str, claim: str) -> float:
    """Module-level convenience function.

    Returns T(G) ∈ [0, 1] using default hyperparameters (λ=0.5).
    Higher → more structurally unsupported / contradicted.
    """
    return _DEFAULT_ANALYZER.tension(evidence, claim)
