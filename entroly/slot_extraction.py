"""
Slot-Aware Entity Verification — EICV Layer 2.5
=================================================

The missing primitive: SLOT BINDING without a parser.

Standard lexical hallucination detectors (including the original ESG)
use BAG-OF-ENTITIES matching:
   "Is entity E from claim found anywhere in evidence?"

This fails on substitution-class hallucinations:
   claim:    "The Eiffel Tower is located in London."
   evidence: "The Eiffel Tower is located in Paris."

A bag-of-entities match returns:
   "Eiffel Tower" → present ✓
   "London"       → absent  ✗
But "absent in 90%-overlap evidence" gives only a modest penalty, because
the IDF-lexical signal scores the claim as well-supported.

The fix: detect that the SAME SYNTACTIC SLOT (left_context, _, right_context)
contains DIFFERENT entities in claim vs evidence. This is a substitution,
not a missing fact. It carries near-1.0 contradiction signal.

Algorithm
---------
Without a dependency parser, we approximate slot structure with
CONTEXT-WINDOW FINGERPRINTS:

  slot(entity)  = (left_context_words, right_context_words)
                  where context = up to W content-words on each side

  align(claim_slot, ev_slot) = jaccard(claim.left, ev.left)
                              + jaccard(claim.right, ev.right)   (averaged)

A SUBSTITUTION is detected when:
  ent_claim ≠ ent_evidence
  ∧ align(claim_slot, ev_slot) ≥ θ_align     (default 0.50)
  ∧ both entities are concrete (not "the", "this", etc.)

The substitution score is align(slot) × 1.0, which goes directly into
the ESG contradiction potential K(u, v).

Why this works
--------------
For FEVER-style refutations:
  claim:    "The Eiffel Tower is located in London."
              ↑ slot for "London" = ({eiffel, tower, located}, {})
  evidence: "The Eiffel Tower is located in Paris, France."
              ↑ slot for "Paris" = ({eiffel, tower, located}, {france})

  align = 0.5*(3/3 + 0/1) = 0.5  (left fully matched; right has "france")
  entities differ → SUBSTITUTION score = 0.5 (above threshold) → K = 0.5 → +
  (and the right-context divergence also adds support difference)

Crucially, this distinguishes:
  - SUPPORTS:   "London" in claim AND "London" in evidence → same entity, no signal
  - REFUTES:    "London" in claim, "Paris" in evidence-same-slot → 0.5 signal
  - SILENT:     "London" in claim, no entity in evidence's matching slot → 0.0

No transformer, no parser. Pure context fingerprinting.

References
----------
- Bach et al., 2017. Hinge-Loss MRFs and Probabilistic Soft Logic. JMLR.
  (Our typed-relation formulation is a discrete PSL specialization.)
- Hearst, 1992. Automatic acquisition of hyponyms from large text corpora.
  (Pattern-based relation extraction without a parser.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]+")
_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

_STOPWORDS = frozenset(
    """
    a an and as at be by for from has have in into is it of on or that the to
    was were will with would could should may might can do does did been being
    its this these those their our your his her them us me my we i you he she
    """.split()
)

_INITIAL_NON_ENTS = frozenset({
    "the", "this", "that", "these", "those", "a", "an", "it", "they",
    "we", "i", "you", "he", "she", "but", "and", "or", "so", "however",
    "moreover", "furthermore", "therefore", "thus", "hence", "indeed",
})


@dataclass(frozen=True)
class Slot:
    """One entity occurrence with its surrounding context."""
    entity: str             # lowercased
    entity_full: str        # original case
    left_context: frozenset[str]   # up to W content-words preceding
    right_context: frozenset[str]  # up to W content-words following
    position: int           # token index in source text

    def context_size(self) -> int:
        return len(self.left_context) + len(self.right_context)


def _tokenize(text: str) -> list[tuple[str, int, int]]:
    """Tokenise text into (token, start, end) triples using _WORD_RE."""
    return [(m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def extract_slots(text: str, window: int = 3) -> list[Slot]:
    """Extract one Slot per named entity occurrence.

    Args:
        text: source text.
        window: number of content-word neighbours to include on each side
            (default 3). Stopwords are NOT counted toward the window.

    Returns:
        List of Slot objects, one per entity. Multiple occurrences of the
        same entity get separate slots (different positions/contexts).
    """
    # First pass: find all entity match spans
    entity_spans: list[tuple[str, int, int]] = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        sent_start_offset = text.find(sent) if sent in text else 0
        # Sentence-relative entity extraction
        for m in _ENTITY_RE.finditer(sent):
            lowered = m.group().lower()
            # Filter sentence-initial false positives
            first_tok = sent.split(None, 1)[0].rstrip(",.!?:;") if sent.split() else ""
            if m.group() == first_tok and lowered in _INITIAL_NON_ENTS:
                continue
            # Map back to global offset
            entity_spans.append((
                m.group(),
                sent_start_offset + m.start(),
                sent_start_offset + m.end(),
            ))

    # Second pass: tokenise for context windows
    all_tokens = _tokenize(text)
    n_tokens = len(all_tokens)

    slots: list[Slot] = []
    for ent_str, ent_start, ent_end in entity_spans:
        # Find the token index for this entity's first token
        ent_tok_idx = None
        for i, (tok, ts, te) in enumerate(all_tokens):
            if ts >= ent_start and te <= ent_end + 1:
                ent_tok_idx = i
                break
        if ent_tok_idx is None:
            continue

        # Count entity span (number of tokens it spans)
        ent_tok_span = max(1, len(_WORD_RE.findall(ent_str)))

        # Walk backwards, collecting content words
        left: list[str] = []
        i = ent_tok_idx - 1
        while i >= 0 and len(left) < window:
            tok = all_tokens[i][0].lower()
            if tok not in _STOPWORDS and len(tok) > 2 and not tok[0].isupper():
                left.append(tok)
            i -= 1

        # Walk forwards
        right: list[str] = []
        i = ent_tok_idx + ent_tok_span
        while i < n_tokens and len(right) < window:
            tok = all_tokens[i][0].lower()
            if tok not in _STOPWORDS and len(tok) > 2 and not tok[0].isupper():
                right.append(tok)
            i += 1

        slots.append(Slot(
            entity=ent_str.lower(),
            entity_full=ent_str,
            left_context=frozenset(left),
            right_context=frozenset(right),
            position=ent_tok_idx,
        ))

    return slots


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0   # both empty: undefined; treat as no alignment evidence
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def slot_alignment(slot_c: Slot, slot_e: Slot) -> float:
    """Alignment score ∈ [0,1] between two slots.

    Uses Jaccard of left and right context windows. Average of the two
    sides; if one side is empty in BOTH slots, that side contributes 0
    (no evidence of alignment).
    """
    left_sim  = _jaccard(slot_c.left_context, slot_e.left_context)
    right_sim = _jaccard(slot_c.right_context, slot_e.right_context)
    # If one side has empty contexts on BOTH, we shouldn't count it.
    # Otherwise it averages.
    n_sides = 0
    total = 0.0
    if slot_c.left_context or slot_e.left_context:
        total += left_sim
        n_sides += 1
    if slot_c.right_context or slot_e.right_context:
        total += right_sim
        n_sides += 1
    return total / max(n_sides, 1)


def slot_substitution_score(
    claim_text: str,
    evidence_text: str,
    *,
    window: int = 3,
    align_threshold: float = 0.40,
) -> tuple[float, list[dict]]:
    """Detect entity substitutions between claim and evidence.

    Returns:
        (max_substitution_score, evidence_list)
        - max_substitution_score ∈ [0, 1]: strongest substitution detected.
          Goes directly into the ESG contradiction K(u,v).
        - evidence_list: per-substitution audit trail
          (claim_entity, evidence_entity, alignment, left_match, right_match)

    Algorithm:
        For each claim slot s_c, find the evidence slot s_e with best
        alignment that has a DIFFERENT entity (different lowercased form).
        If alignment ≥ align_threshold, record substitution score = alignment.
    """
    claim_slots = extract_slots(claim_text, window=window)
    ev_slots = extract_slots(evidence_text, window=window)

    if not claim_slots or not ev_slots:
        return 0.0, []

    # Critical guard: if the claim entity appears ANYWHERE in evidence
    # (not just in the aligned slot), it's supported — skip substitution
    # check. This rules out the SUPPORTS false-positive case where claim
    # entity "Paris" appears in evidence next to a different entity
    # "France", producing a spurious 0.5 alignment.
    ev_text_lower = evidence_text.lower()

    audit: list[dict] = []
    max_subst = 0.0

    for sc in claim_slots:
        # Guard: claim entity present anywhere in evidence → supported, skip
        if sc.entity in ev_text_lower:
            continue

        best_align = 0.0
        best_se: Slot | None = None
        for se in ev_slots:
            # Same entity: NOT a substitution
            if sc.entity == se.entity or sc.entity in se.entity or se.entity in sc.entity:
                continue
            align = slot_alignment(sc, se)
            if align > best_align:
                best_align = align
                best_se = se

        if best_se is not None and best_align >= align_threshold:
            audit.append({
                "claim_entity": sc.entity_full,
                "evidence_entity": best_se.entity_full,
                "alignment": round(best_align, 4),
                "claim_left": list(sc.left_context),
                "ev_left": list(best_se.left_context),
                "claim_right": list(sc.right_context),
                "ev_right": list(best_se.right_context),
            })
            if best_align > max_subst:
                max_subst = best_align

    return float(max_subst), audit
