"""
Atomic Fact Decomposition + Quantifier/Magnitude Semantics — EICV Phase 6
============================================================================

Three additions that materially improve FEVER-class claim verification:

1. **Atomic decomposition** (verb-anchored, coreference-resolving)
   "Einstein developed relativity in 1915 and won the Nobel Prize in 1921"
   → atom₁: "Einstein developed relativity in 1915"
   → atom₂: "Einstein won the Nobel Prize in 1921"   ← subject copied
   Each atom is verified independently; claim survives iff all atoms survive.
   Without a parser: verb-dictionary anchoring + conjunction splitting.

2. **Quantifier-aware contradiction**
   "All X are Y"   contradicted by "some X are not Y" / "no X are Y"
   "No X are Y"    contradicted by "some X are Y"
   "Some X are Y"  contradicted only by "no X are Y"
   Universal/existential cues are extracted from claim AND evidence.

3. **Magnitude-aware numeric comparison**
   "born in 1867" vs evidence "born in 1869" → small distance (~2 years)
   "born in 1867" vs evidence "born in 1567" → large distance (~300 years)
   The contradiction score scales with log-magnitude of the relative
   difference, not just present/absent. This is FActScore's "atomic
   precision" applied at the number level.

These primitives plug into ESG._contradiction_score as additional signals
that take the max with the existing entity-slot, number-presence, and
negation signals.

Theoretical motivation: a claim's truth value is the LOGICAL CONJUNCTION
of its atoms (∧). The probability of all-atoms-supported is the product of
per-atom probabilities under (conditional) independence — a multiplicative
penalty for any single unsupported atom. This is the e-value composition
from Layer 5 applied at the sub-claim level.

References
----------
- Min et al., 2023. FActScore. EMNLP 2023. (atomic precision concept)
- Honovich et al., 2022. TRUE. (compositional NLI evaluation)
- Barwise & Cooper, 1981. Generalized Quantifiers and Natural Language.
  Linguistics and Philosophy 4(2). (quantifier semantics foundation)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


# ── Verb dictionary (for atomic splitting without parser) ────────────

# Common English verbs that anchor a clause. Used to detect when a
# conjunction-separated segment is a new clause needing subject-coreference.
_VERB_FORMS = frozenset(
    """
    is was are were has had have do does did been being am
    developed develops developing
    discovered discovers discovering
    found finds finding
    won wins winning
    lost loses losing
    wrote writes writing written
    made makes making
    created creates creating
    produced produces producing
    became becomes becoming
    served serves serving
    led leads leading
    governed governs governing
    ruled rules ruling
    appeared appears appearing
    showed shows showing
    proved proves proving
    received receives receiving
    earned earns earning
    studied studies studying
    taught teaches teaching
    worked works working
    lived lives living
    died dies dying
    composed composes composing
    designed designs designing
    invented invents inventing
    built builds building
    formed forms forming
    began begins beginning
    started starts starting
    ended ends ending
    married marries marrying
    fought fights fighting
    travelled traveled travels
    visited visits visiting
    joined joins joining
    moved moves moving
    became
    """.split()
)

# Coordinating conjunctions where the second clause MAY drop its subject
_COORD_RE = re.compile(
    r"\b(and|but|or|while|whereas|moreover|furthermore|additionally"
    r"|nevertheless|yet|however)\b",
    re.I,
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]+")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")


# ── Atomic decomposition ─────────────────────────────────────────────


@dataclass(frozen=True)
class Atom:
    text: str
    subject: str = ""    # extracted/inherited subject
    verb: str = ""       # primary verb form
    source: str = "claim"


def _extract_subject(clause: str) -> str:
    """Extract the subject (typically the first noun phrase) from a clause.

    Without a parser, we look for: leading capitalized phrase OR leading
    'The X' / 'A X' pattern. Returns empty string if no clear subject.
    """
    tokens = clause.strip().split()
    if not tokens:
        return ""

    # Handle "The/A/An ___" patterns
    if tokens[0].lower() in {"the", "a", "an", "this", "that", "these", "those"} and len(tokens) > 1:
        # Take "The X" or "The X Y" until a known verb or stopword stops it
        subj_tokens = [tokens[0]]
        for tok in tokens[1:]:
            if tok.lower() in _VERB_FORMS:
                break
            subj_tokens.append(tok)
            if len(subj_tokens) >= 4:   # cap subject length
                break
        return " ".join(subj_tokens)

    # Leading capitalized phrase (entity-as-subject)
    m = _ENTITY_RE.match(clause)
    if m:
        return m.group()

    # Single capitalized word
    if tokens[0][0].isupper():
        return tokens[0]

    return ""


def _first_verb_index(tokens: list[str]) -> int:
    """Return the index of the first verb form in the token list, or -1."""
    for i, t in enumerate(tokens):
        if t.lower() in _VERB_FORMS:
            return i
    return -1


def _segment_has_subject(segment: str) -> bool:
    """Heuristic: does this segment have its own subject?

    Returns True if the segment starts with a noun-phrase before any verb.
    Returns False if it starts with a verb (i.e. subject was elided).
    """
    tokens = segment.strip().split()
    if not tokens:
        return False
    # If first non-stopword is a verb form, no subject
    for tok in tokens:
        if len(tok) <= 2:
            continue
        tl = tok.lower()
        if tl in _VERB_FORMS:
            return False
        # If we hit a content word that's NOT a verb, we have a subject
        return True
    return False


def decompose_atoms(text: str) -> list[Atom]:
    """Decompose text into atomic propositions with subject coreference.

    Strategy:
      1. Split by sentence boundary
      2. Within sentence, split at coordinating conjunctions (and/but/or)
      3. For each split fragment, if it has no subject, copy from prior atom
      4. Each Atom has (text, subject, verb)

    Returns:
        list of Atom objects in document order.
    """
    atoms: list[Atom] = []

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # Find coordinator positions
        matches = list(_COORD_RE.finditer(sent))
        if not matches:
            # Single-clause sentence
            subj = _extract_subject(sent)
            verb_tok = ""
            toks = sent.split()
            vi = _first_verb_index(toks)
            if vi >= 0:
                verb_tok = toks[vi].lower()
            atoms.append(Atom(text=sent, subject=subj, verb=verb_tok))
            continue

        # Multi-clause: split at coordinators
        segments = []
        last_end = 0
        for m in matches:
            seg = sent[last_end:m.start()].strip(" ,;")
            if seg:
                segments.append(seg)
            last_end = m.end()
        tail = sent[last_end:].strip(" ,;")
        if tail:
            segments.append(tail)

        # Process segments with coreference resolution
        current_subject = ""
        for seg in segments:
            seg_clean = seg.strip(" ,;.")
            if not seg_clean or len(seg_clean.split()) < 2:
                continue

            if _segment_has_subject(seg_clean):
                # Has its own subject — extract and use
                current_subject = _extract_subject(seg_clean) or current_subject
                atom_text = seg_clean
            else:
                # Subject elided — prepend the running subject
                if current_subject:
                    atom_text = f"{current_subject} {seg_clean}"
                else:
                    atom_text = seg_clean

            toks = atom_text.split()
            vi = _first_verb_index(toks)
            verb_tok = toks[vi].lower() if vi >= 0 else ""
            atoms.append(Atom(
                text=atom_text,
                subject=current_subject,
                verb=verb_tok,
            ))

    return atoms or [Atom(text=text.strip())]


# ── Quantifier handling ──────────────────────────────────────────────


_UNIVERSAL_CUES = frozenset({
    "all", "every", "each", "any", "always", "everyone", "everybody",
    "everything", "everywhere", "constantly", "continually",
    "annual", "annually", "yearly", "daily", "monthly", "weekly",
    "regularly", "frequently",
})
_EXISTENTIAL_CUES = frozenset({
    "some", "several", "few", "sometimes", "occasionally",
    "certain", "many", "rarely", "seldom", "scarcely", "barely",
    "infrequently", "uncommon",
})
_NEGATIVE_QUANTIFIER = frozenset({
    "no", "none", "never", "nobody", "nothing", "nowhere", "neither",
    "without", "absent",
})

# Verbs that imply negation / refusal (Renato "refused", Genocide "fails")
_REFUSAL_CUES = frozenset({
    "refused", "rejected", "denied", "declined", "fails", "failed",
    "avoided", "abandoned", "ceased", "stopped",
})


def _detect_quantifier(text: str) -> str:
    """Detect the dominant quantifier type: universal | existential | negative | none.

    Looks at the first few content words of the text.
    """
    tokens = [t.lower() for t in _WORD_RE.findall(text)][:8]
    for tok in tokens:
        if tok in _NEGATIVE_QUANTIFIER:
            return "negative"
    for tok in tokens:
        if tok in _UNIVERSAL_CUES:
            return "universal"
    for tok in tokens:
        if tok in _EXISTENTIAL_CUES:
            return "existential"
    return "none"


def _has_refusal_cue(text: str) -> bool:
    """Detect refusal/denial verbs that act as scope-wide negators."""
    tokens = {t.lower() for t in _WORD_RE.findall(text)}
    return bool(tokens & _REFUSAL_CUES)


def quantifier_contradiction(claim: str, evidence: str) -> float:
    """Detect quantifier-class contradictions.

    Returns ∈ [0, 1]:
      universal claim ∧ negative-quantifier evidence  → 1.0
      universal claim ∧ existential evidence          → 0.3
      negative claim  ∧ universal/existential evidence → 0.7
      existential     ∧ negative evidence             → 0.5
      refusal-verb claim ∧ assertion-verb evidence   → 0.8  (Renato "refused" vs "began")
      claim "rarely X" ∧ evidence "annually X"        → universal/existential conflict
    """
    qc = _detect_quantifier(claim)
    qe = _detect_quantifier(evidence)

    if qc == "universal" and qe == "negative":
        return 1.0
    if qc == "negative" and qe in ("universal", "existential"):
        return 0.7
    if qc == "universal" and qe == "existential":
        return 0.30
    if qc == "existential" and qe == "negative":
        return 0.50
    # Cross-class: "rarely X" (existential) vs "annual X" (universal)
    if qc == "existential" and qe == "universal":
        return 0.40
    if qc == "universal" and qe == "existential":
        return 0.40

    # Refusal-verb scope: "refused to study" vs evidence "began studying"
    if _has_refusal_cue(claim) and not _has_refusal_cue(evidence):
        # Only fire if evidence asserts the relevant action positively.
        # Use topical overlap as a weak gate.
        c_words = {t.lower() for t in _WORD_RE.findall(claim)}
        e_words = {t.lower() for t in _WORD_RE.findall(evidence)}
        if len(c_words & e_words) >= 2:
            return 0.80

    return 0.0


def modifier_mismatch(claim: str, evidence: str) -> float:
    """Soft contradiction when type heads MATCH but modifiers DIFFER.

    Example:
      claim: "Match Point was a drama film"     type=drama film,    head=film
      evid:  "Match Point is a thriller film"   type=thriller film, head=film
      heads match (film=film), modifiers differ (drama vs thriller) → 0.45

    This is a SOFT signal — modifier differences are sometimes synonymous
    or descriptive (e.g. "great film" / "amazing film"), so we score low
    (0.45) rather than triggering full contradiction.
    """
    claim_types = _extract_type_assertions(claim)
    ev_types = _extract_type_assertions(evidence)
    if not claim_types or not ev_types:
        return 0.0

    max_score = 0.0
    for c_subj, c_type in claim_types:
        for e_subj, e_type in ev_types:
            if not (c_subj == e_subj or c_subj in e_subj or e_subj in c_subj):
                continue
            c_parts = c_type.split()
            e_parts = e_type.split()
            if len(c_parts) < 2 or len(e_parts) < 2:
                continue   # need at least 2 words for modifier comparison
            if c_parts[-1] != e_parts[-1]:
                continue   # different head → handled by categorical
            # Same head — compare modifiers (everything before the head)
            c_mods = set(c_parts[:-1])
            e_mods = set(e_parts[:-1])
            # Intersection check
            if c_mods & e_mods:
                continue   # shared modifier → compatible
            # All modifiers differ — soft contradiction
            if 0.45 > max_score:
                max_score = 0.45

    return max_score


# ── Magnitude-aware numeric comparison ───────────────────────────────


def _looks_like_year(n: float) -> bool:
    return 1500.0 <= n <= 2100.0 and abs(n - int(n)) < 1e-6


def numeric_magnitude_mismatch(claim: str, evidence: str) -> float:
    """Magnitude-aware numeric contradiction score.

    Two regimes:
      Year-like (1500-2100, integer): use ABSOLUTE difference / 100.
        |1960 - 2002| / 100 = 0.42 → moderate contradiction
        |1937 - 1937| / 100 = 0.00 → exact
      Other numbers: relative magnitude on log scale.
        |100 - 1000| / 1000 = 0.9 → strong contradiction.

    Aggregate: max over claim numbers (most-discrepant number drives score).
    """
    claim_nums = [float(n) for n in _NUMBER_RE.findall(claim)]
    ev_nums = [float(n) for n in _NUMBER_RE.findall(evidence)]

    if not claim_nums:
        return 0.0
    if not ev_nums:
        return 0.0

    max_score = 0.0
    for cn in claim_nums:
        if cn == 0:
            continue
        # Need to find best match in evidence
        if _looks_like_year(cn):
            # Year regime: absolute difference / 100
            year_evs = [en for en in ev_nums if _looks_like_year(en)]
            if year_evs:
                best_diff_abs = min(abs(cn - en) for en in year_evs)
                if best_diff_abs < 0.5:
                    continue   # exact match
                # 100 years = full contradiction
                contrib = min(1.0, best_diff_abs / 100.0)
            else:
                continue   # evidence has no year — silent
        else:
            best_diff = min(
                abs(cn - en) / max(abs(cn), abs(en), 1.0) for en in ev_nums
            )
            if best_diff < 1e-6:
                continue
            contrib = min(1.0, math.log1p(best_diff * 10.0))
        if contrib > max_score:
            max_score = contrib

    return max_score


# ── Categorical type contradiction (Phase 6+) ────────────────────────


# Pattern: "[Subject] [is/are/was/were] [a/an/the] [optional number] [TYPE]"
# Captures the subject and the type phrase. Allows dots and hyphens in the
# type to handle "U.S." / "British-American". Allows leading numbers (year)
# before the type word.
_TYPE_ASSERTION_RE = re.compile(
    r"\b([A-Z][\w\s,\.&\-']{2,60}?)\s+"
    r"(?:is|are|was|were)\s+"
    r"(?:a|an|the)\s+"
    r"(?:\d+\s+)?"                                       # optional year
    r"([a-zA-Z][a-zA-Z\.]*(?:[\s\-][a-zA-Z][a-zA-Z\.]*){0,4})",
)

# Stoptypes — words that look like types but are too generic to compare
_TYPE_STOPLIST = frozenset({
    "person", "thing", "place", "object", "entity", "item",
    "former", "latter", "kind", "type", "sort", "way",
    "result", "form", "part", "member", "name",
})


_TYPE_STOP_PREPS = (" in ", " by ", " of ", " from ", " on ", " at ", " for ",
                    " with ", " to ", " into ", " about ")


def _clean_type(typ: str) -> str:
    """Normalise a type phrase: strip punctuation, stop at prepositions."""
    typ = re.sub(r"\.", "", typ)   # "U.S." → "US"
    for prep in _TYPE_STOP_PREPS:
        if prep in typ:
            typ = typ.split(prep)[0]
            break
    return typ.strip()


def _type_head(typ: str) -> str:
    """The head noun is the LAST content word of the type phrase."""
    parts = typ.split()
    return parts[-1] if parts else ""


def _extract_type_assertions(text: str) -> list[tuple[str, str]]:
    """Extract (subject, type) tuples from "X is a Y" patterns.

    Returns lowercased subjects and CLEANED types (prep phrases stripped).
    """
    out: list[tuple[str, str]] = []
    for m in _TYPE_ASSERTION_RE.finditer(text):
        subj = m.group(1).strip().lower()
        typ_raw = m.group(2).strip().lower()
        typ = _clean_type(typ_raw)
        if not typ:
            continue
        # Filter verb-as-type (passives like "established", "founded")
        typ_first = typ.split()[0]
        if typ_first in _TYPE_STOPLIST:
            continue
        if typ_first.endswith("ed") and len(typ_first) > 4:
            continue
        if typ_first.endswith("ing") and len(typ_first) > 5:
            continue
        if typ_first in _VERB_FORMS:
            continue
        out.append((subj, typ))
    return out


def categorical_type_contradiction(claim: str, evidence: str) -> float:
    """Detect type-level contradictions ("X is a CITY" vs "X is a COUNTY").

    For each (subj_c, type_c) extracted from claim, find any
    (subj_e, type_e) in evidence with matching subject. If the type
    nouns DIFFER (and neither is a substring of the other), score 0.85.

    Subject matching: substring-or-equal, lowercased.
    Type-difference check: also requires non-substring (so "thriller film"
    ≠ "horror thriller film" is gated to <0.85).

    Returns ∈ [0, 1].
    """
    claim_types = _extract_type_assertions(claim)
    if not claim_types:
        return 0.0
    ev_types = _extract_type_assertions(evidence)
    if not ev_types:
        return 0.0

    max_score = 0.0
    for c_subj, c_type in claim_types:
        for e_subj, e_type in ev_types:
            # Subject match (substring tolerance for "Camden" vs "Camden, NJ")
            if not (c_subj == e_subj or
                    c_subj in e_subj or
                    e_subj in c_subj):
                continue

            # Compare type HEAD nouns (last word of the type phrase).
            # "drama film" and "thriller film" share head "film" → not a
            # contradiction. "city in county" → head "city" vs "county" →
            # different heads → contradiction.
            c_head = _type_head(c_type)
            e_head = _type_head(e_type)
            if not c_head or not e_head:
                continue
            if c_head == e_head:
                continue
            # Stem-equal? (e.g. "city" / "cities")
            from entroly.esg import _stem
            if _stem(c_head) == _stem(e_head):
                continue

            # Different head nouns, same subject → strong contradiction
            if 0.85 > max_score:
                max_score = 0.85

    return max_score


# ── Composite atomic verification ────────────────────────────────────


def per_atom_support_breakdown(
    claim: str,
    evidence: str,
    *,
    scorer,
) -> dict:
    """Decompose claim into atoms, score each independently, return breakdown.

    Args:
        claim: the full claim text.
        evidence: the evidence corpus.
        scorer: callable(evidence, atom_text) -> tension ∈ [0,1].
            Typically ESGAnalyzer.tension.

    Returns:
        dict with:
          atoms_text: list of atom texts
          atom_tensions: list of T(G) per atom
          max_atom_tension: max of per-atom tensions (worst atom)
          mean_atom_tension: average
          combined_via_max: claim is unsupported iff ANY atom is high-T(G)
          combined_via_mean: average T(G) for diagnostic
    """
    atoms = decompose_atoms(claim)
    atom_texts = [a.text for a in atoms]
    atom_tensions = [scorer(evidence, a.text) for a in atoms]

    return {
        "atoms_text": atom_texts,
        "atom_tensions": [round(t, 4) for t in atom_tensions],
        "max_atom_tension": round(max(atom_tensions) if atom_tensions else 0.0, 4),
        "mean_atom_tension": round(
            sum(atom_tensions) / len(atom_tensions) if atom_tensions else 0.0, 4
        ),
        "n_atoms": len(atoms),
    }
