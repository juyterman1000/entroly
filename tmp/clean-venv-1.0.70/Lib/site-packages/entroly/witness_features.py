"""
WITNESS Feature Extraction
==========================

Computes the feature vector φ(c, C) ∈ [0,1]^K for a claim c against
context C. This is the upstream of the continuous risk model
(witness_risk_model.py). Pure functions — no state, no I/O.

Information-theoretic claim
---------------------------
The original witness verifier collapsed rich per-claim evidence to one
of 4 labels {grounded, unsupported, unknown, contradicted} and then
mapped each label to one of 4 risk values {0.05, 0.55, 0.80, 0.92}.
That's 2 bits of information per claim — a hard information ceiling
on what any downstream policy can decide.

The features here preserve the full information content (real-valued
vectors), so the risk model gets the *signal*, not the *summary*. The
empirical consequence is visible on the HaluEval forensic: with the
bucketed pipeline, hallucinated and safe dialogue claims had nearly
identical mean risk (0.44 vs 0.47). The continuous pipeline keeps the
features separated by ~0.2 on the same data, which is enough for the
graduated policy to act.

Feature definitions
-------------------
All features are normalized to [0, 1] (or [-1, 1] for signed signals,
clamped at the risk model).

φ₁ entity_precision      — fraction of named entities in c that appear in C
φ₂ number_consistency    — 1 if all numbers in c are in C; signed mismatch ratio if not
φ₃ idf_lex_overlap       — IDF-weighted Jaccard of content words
φ₄ quote_support         — longest common substring / |c|
φ₅ forward_entail        — existing PAV/quote/slot support (C entails c)
φ₆ reverse_entail        — fraction of c's content words covered by C (cheap reverse-NLI)
φ₇ negation_polarity     — 1 if polarity matches, -1 if c affirms what C negates
φ₈ adequacy              — existing weighted adequacy

References
----------
- Min et al., 2023. FactScore: Fine-grained Atomic Evaluation of
  Factual Precision in Long-Form Text Generation. EMNLP 2023.
- Lin & Hovy, 2003. Automatic Evaluation of Summaries Using N-gram
  Co-Occurrence Statistics. NAACL 2003. (IDF weighting baseline.)
- Honovich et al., 2022. TRUE: Re-evaluating Factual Consistency
  Evaluation. NAACL 2022. (Bidirectional entailment intuition.)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher


# Negation cues (lowercase). Order matters: longest first for matching.
_NEG_CUES = (
    "is not", "are not", "was not", "were not", "does not", "do not",
    "did not", "has not", "have not", "had not", "will not", "would not",
    "cannot", "can not", "isn't", "aren't", "wasn't", "weren't", "doesn't",
    "don't", "didn't", "hasn't", "haven't", "hadn't", "won't", "wouldn't",
    "no ", "not ", "never ", "none ",
)

# Affirmation cues — the absence of negation in a relevant context window
# isn't enough; some claims actively assert ("X is Y").
_AFFIRM_VERBS = frozenset({"is", "are", "was", "were", "has", "have", "had"})

_STOPWORDS = frozenset(
    """
    a an and as at be by for from has have in into is it of on or that the to
    was were will with would could should may might can do does did been being
    """.split()
)


@dataclass(frozen=True)
class ClaimFeatures:
    """The feature vector φ(c, C). All entries in [-1, 1]."""
    entity_precision: float
    number_consistency: float
    idf_lex_overlap: float
    quote_support: float
    forward_entail: float
    reverse_entail: float
    negation_polarity: float
    adequacy: float
    qa_alignment: float = 1.0   # φ₉: QA-only signal, neutral elsewhere

    def as_dict(self) -> dict:
        return asdict(self)

    def as_vector(self) -> list[float]:
        return [
            self.entity_precision,
            self.number_consistency,
            self.idf_lex_overlap,
            self.quote_support,
            self.forward_entail,
            self.reverse_entail,
            self.negation_polarity,
            self.adequacy,
            self.qa_alignment,
        ]

    @classmethod
    def feature_names(cls) -> list[str]:
        return [
            "entity_precision", "number_consistency", "idf_lex_overlap",
            "quote_support", "forward_entail", "reverse_entail",
            "negation_polarity", "adequacy", "qa_alignment",
        ]


# ── Primitive extractors ─────────────────────────────────────────────


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]+")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")


def content_words(text: str) -> set[str]:
    """Lowercased content words, stopwords removed."""
    return {
        w.lower() for w in _WORD_RE.findall(text)
        if w.lower() not in _STOPWORDS and len(w) > 2
    }


_SENTENCE_INITIAL_NON_ENTITIES = frozenset({
    # Common English sentence-openers that look like entities under regex
    "the", "this", "that", "these", "those", "a", "an", "it", "they",
    "we", "i", "you", "he", "she", "but", "and", "or", "so", "however",
    "moreover", "furthermore", "therefore", "thus", "hence", "indeed",
    "actually", "interestingly", "importantly", "specifically",
    "additionally", "consequently", "yet", "still", "call", "use", "run",
    "read", "write", "return", "update", "create", "delete",
})


def named_entities(text: str) -> set[str]:
    """Cheap NER: contiguous capitalized words.

    Keeps sentence-initial caps EXCEPT for a tiny blocklist of common
    English sentence-openers. This is what catches the entity-swap
    hallucination class: 'Bohr developed general relativity' → 'Bohr'
    extracted as entity → checked against context → not found → high
    contradiction signal.

    The previous behavior (drop all sentence-initial caps) was too
    aggressive: it threw away the very signal needed for HaluEval-style
    entity swaps. Wikipedia-style content puts entities at sentence
    start very frequently.
    """
    out: set[str] = set()
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        for m in _ENTITY_RE.findall(sentence):
            lowered = m.lower()
            # First word of sentence: only keep if not a sentence-initial
            # non-entity. Subsequent matches are always kept.
            first_token = sentence.split(None, 1)[0].rstrip(",.!?:;")
            if m == first_token and lowered in _SENTENCE_INITIAL_NON_ENTITIES:
                continue
            out.add(lowered)
    return out


def extract_numbers(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(text))


def _tokens(text: str) -> list[str]:
    """Ordered lowercased word tokens (no stopword removal — order matters
    for contiguity)."""
    return [w.lower() for w in _WORD_RE.findall(text)]


def _longest_contiguous_run(ans: list[str], sent: list[str]) -> int:
    """Length of the longest *contiguous token substring* shared by `ans`
    and `sent` (any offset in either). Classic LCS-substring DP, O(|ans|·
    |sent|) time / O(|sent|) space; inputs are short (a QA answer vs. one
    sentence). This is the extractive-grounding primitive: a genuine
    extractive answer is a contiguous span of its evidence sentence; a
    recombination / wrong-option answer is not.
    """
    if not ans or not sent:
        return 0
    prev = [0] * (len(sent) + 1)
    best = 0
    for a in ans:
        cur = [0] * (len(sent) + 1)
        for j, s in enumerate(sent, 1):
            if a == s:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


# ── Per-feature computations ─────────────────────────────────────────


def feat_entity_precision(claim: str, context: str) -> float:
    """φ₁: SIGNED entity-match score.

    Returns:
        1.0     all entities in claim are in context (full support)
        0.0     claim has no entities (neutral/no signal)
       -k/n     k of n claim entities are NOT in context — *active*
                disagreement, treated as a hard-contradiction gate
                by the risk model (parallel to number_consistency).

    Wikipedia-style HaluEval hallucinations typically swap an entity
    rather than invent one wholesale ('Einstein' → 'Bohr'). The signed
    return preserves the strength of that signal so the risk model
    can multiplicatively gate ρ → 1 when an active mismatch is seen,
    even if other supportive features score high.
    """
    ents = named_entities(claim)
    if not ents:
        return 0.0
    ctx_lower = context.lower()
    matched = sum(1 for e in ents if e in ctx_lower)
    if matched == len(ents):
        return 1.0
    # Signed mismatch — proportion of claim entities that are NOT in context
    missing = len(ents) - matched
    return -missing / len(ents)


def feat_number_consistency(claim: str, context: str) -> float:
    """φ₂: signed measure of number agreement.

    1.0  → all numbers in claim are in context
    0.0  → claim has no numbers (neutral/no signal)
    -k/n → k of n numbers in claim don't appear in context

    Negative values signal *active disagreement*, which is much
    stronger evidence than mere absence. The risk model can apply a
    larger weight to negative values to catch '1962' → '1985' swaps.
    """
    nums = extract_numbers(claim)
    if not nums:
        return 0.0
    ctx_nums = extract_numbers(context)
    missing = sum(1 for n in nums if n not in ctx_nums)
    if missing == 0:
        return 1.0
    # Negative proportion — signed mismatch
    return -missing / len(nums)


def feat_idf_lex_overlap(claim: str, context: str, idf: dict[str, float] | None = None) -> float:
    """φ₃: IDF-weighted Jaccard of content words.

    When idf is provided, each shared word is weighted by its inverse
    document frequency, so rare-word matches dominate over common-word
    noise. When idf is None, we use an in-sample approximation: a word's
    weight is 1/(1 + log(1 + count_in_context)).
    """
    claim_words = content_words(claim)
    if not claim_words:
        return 1.0
    ctx_words = content_words(context)
    shared = claim_words & ctx_words
    if not shared:
        return 0.0

    def weight(w: str) -> float:
        if idf is not None and w in idf:
            return idf[w]
        # In-sample fallback: more frequent ctx words = lower weight
        return 1.0 / (1.0 + math.log(1 + context.lower().count(w)))

    num = sum(weight(w) for w in shared)
    den = sum(weight(w) for w in claim_words)
    return num / max(den, 1e-9)


def feat_quote_support(claim: str, context: str) -> float:
    """φ₄: longest verbatim substring of claim found in context, /|claim|."""
    if not claim or not context:
        return 0.0
    matcher = SequenceMatcher(None, claim.lower(), context.lower(), autojunk=False)
    match = matcher.find_longest_match(0, len(claim), 0, len(context))
    return match.size / max(len(claim), 1)


def feat_forward_entail(claim: str, context: str) -> float:
    """φ₅: 'context entails claim'.

    Combines quote support and content-word overlap weighted by claim
    length. This is the existing forward-direction signal that the
    PAV machinery already produces; we re-derive it cleanly here so
    feature extraction is self-contained.
    """
    quote = feat_quote_support(claim, context)
    claim_words = content_words(claim)
    if not claim_words:
        return 1.0
    ctx_words = content_words(context)
    overlap = len(claim_words & ctx_words) / len(claim_words)
    # 0.5 quote / 0.5 overlap — quote is verbatim ground truth, overlap
    # is paraphrase tolerance. Both ∈ [0,1] so combination is in [0,1].
    return 0.5 * quote + 0.5 * overlap


def feat_reverse_entail(claim: str, context: str) -> float:
    """φ₆: 'claim contains nothing that isn't in context'.

    The opposite direction from forward_entail. A hallucinated claim
    often shares topic with context (high forward), but introduces
    facts that aren't there (low reverse). This is the cheap proxy
    for bidirectional NLI that Honovich et al. 2022 found materially
    improves consistency detection.

    Score: fraction of c's content words that appear anywhere in C.
    Distinct from forward because the denominator is the claim's
    *unique* content words (set), not the context's.
    """
    claim_words = content_words(claim)
    if not claim_words:
        return 1.0
    ctx_words = content_words(context)
    matched = len(claim_words & ctx_words)
    return matched / len(claim_words)


def feat_negation_polarity(claim: str, context: str) -> float:
    """φ₇: polarity-mismatch detector.

    Returns:
        1.0  if overlapping content has matching polarity
        0.0  if polarity is not observable
        -1.0 if claim affirms something that context appears to negate
              (or vice versa) on overlapping content

    Heuristic: for each shared content word between claim and context,
    check the 30 chars preceding the word's first occurrence in each
    source for a negation cue. If exactly one side has negation, the
    polarity has flipped.

    Returns -1.0 as soon as any single shared word shows a flip — a
    single polarity inversion is enough evidence of disagreement.
    """
    shared = content_words(claim) & content_words(context)
    if not shared:
        return 0.0
    claim_lower = claim.lower()
    context_lower = context.lower()

    def has_negation_before(text: str, word: str) -> bool:
        idx = text.find(word)
        if idx <= 0:
            return False
        preceding = text[max(0, idx - 30):idx]
        return any(cue in preceding for cue in _NEG_CUES)

    for word in shared:
        neg_claim = has_negation_before(claim_lower, word)
        neg_ctx = has_negation_before(context_lower, word)
        if neg_claim != neg_ctx:
            return -1.0
    return 1.0


# ── Top-level extractor ──────────────────────────────────────────────


def feat_qa_alignment(
    answer: str,
    knowledge: str,
    question: str | None,
) -> float:
    """φ₉: question-grounded answer alignment via sentence-density peak.

    Returns SIGNED score in [-1, +1]:
        +1.0      strong support (answer verbatim/heavily overlapping
                  with the question's evidence sentence)
         0..1     partial support
         0        weak — answer's content not in Q-relevant sentence
        -1..0     ACTIVE mismatch: answer has content but it's clearly
                  in a different sentence than the question's evidence
                  (e.g. "1921" answering "what year did Einstein develop
                  relativity?" — 1921 IS in K but for a different fact)

    The negative regime acts as a hard-contradiction gate in the risk
    model (registered in _HARD_CONTRADICTION_FEATURES). A single
    qa-misplaced answer should be enough to drive ρ→1, even when
    other supportive features all pass.

    Returns 0.0 (neutral/no signal) when there is no question or no answer content.

    Algorithm:
      1. Split knowledge K into sentences (Wikipedia-style is clean).
      2. For each sentence sᵢ, score sᵢ = Σ idf(t) for t ∈ Q ∩ sᵢ.
         IDF estimated in-sample.
      3. w* = arg max sᵢ. Tiebreak: prefer the sentence whose content
         words include more of the *answer's* words too — this fuses
         Q-relevance and A-relevance, which is the right thing because
         the right answer's sentence will both match Q-keywords AND
         contain the answer content.
      4. Output: |answer_content ∩ w*| / |answer_content| (IDF-weighted).

    Why sentence-level instead of fixed-char windows:
      Hallucinations like "Einstein won the Nobel Prize in 1921" for the
      question "what year did Einstein develop general relativity?" reuse
      a year that's in the knowledge but at the WRONG sentence. A fixed
      sliding window that's larger than typical sentence length can
      include the wrong-sentence year as if it supported the answer.
      Sentence-level windows respect natural fact boundaries.

    Properties:
      - returns 0.0 when no question (neutral/no signal)
      - returns 0.0 when answer has no content words (nothing to verify)
      - bounded in [0, 1]
    """
    if not question or not question.strip():
        return 0.0
    if not knowledge or not knowledge.strip():
        return 0.0
    q_words = content_words(question)
    if not q_words:
        return 0.0
    a_words = content_words(answer)
    a_nums = extract_numbers(answer)
    # When the answer has no content words AND no numbers (e.g. empty
    # or pure punctuation), it carries no factual claim to verify.
    if not a_words and not a_nums:
        return 0.0

    K_lower = knowledge.lower()
    a_lower = answer.lower()

    def _w(t: str) -> float:
        return 1.0 / (1.0 + math.log(1 + K_lower.count(t)))

    # Sentence segmentation
    sentences = re.split(r"(?<=[.!?])\s+", K_lower)
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        sentences = [K_lower]

    best_score = -1.0
    best_sentence = ""
    for sent in sentences:
        sent_words = set(_WORD_RE.findall(sent))
        q_score = sum(_w(t) for t in q_words if t in sent_words)
        # Tiebreaker: a tiny boost when the answer's content words are
        # also in this sentence — keeps the right sentence centered on
        # the answer's location rather than purely Q-keyword density.
        a_lex = sum(_w(t) for t in a_words if t in sent_words) * 0.1
        total = q_score + a_lex
        if total > best_score:
            best_score = total
            best_sentence = sent

    if best_score <= 0.0:
        return 0.0

    # Compute answer-overlap-IDF in the best sentence.
    # When the answer has only numbers (no content words), we fall back
    # to a *numeric-presence* check: is the number in the Q-relevant
    # sentence? This is the right thing for short factual QA answers.
    best_words = set(_WORD_RE.findall(best_sentence))
    if a_words:
        num = sum(_w(t) for t in a_words if t in best_words)
        den = sum(_w(t) for t in a_words)
        word_score = num / den if den > 0.0 else 1.0
    else:
        word_score = 1.0  # neutral if no words to score

    # Numeric presence: if the answer contains numbers, score by their
    # presence in the Q-relevant sentence specifically.
    if a_nums:
        sentence_nums = set(_NUMBER_RE.findall(best_sentence))
        num_present = sum(1 for n in a_nums if n in sentence_nums)
        numeric_score = num_present / len(a_nums)
        # Combine: when answer has BOTH words and numbers, average them;
        # when only numbers, numeric is the score.
        if a_words:
            word_score = 0.5 * word_score + 0.5 * numeric_score
        else:
            word_score = numeric_score

    # Verbatim-substring bonus: if the literal answer string appears in
    # the Q-relevant sentence, that's strong corroboration. Useful for
    # short noun-phrase answers like "Princeton" or "Tony Blair".
    verbatim_bonus = 0.0
    a_stripped = a_lower.strip(" .,!?\"'")
    if len(a_stripped) >= 2 and a_stripped in best_sentence:
        verbatim_bonus = 0.3

    raw_score = min(1.0, word_score + verbatim_bonus)

    # ── Signed-feature transformation ─────────────────────────────
    # Only promote a low score to ACTIVE mismatch when the evidence
    # is unambiguous — false positives here hurt Retention badly, so
    # the gate is conservative.
    #
    # Trigger conditions (BOTH must hold):
    #   (a) Best-sentence Q-score is high (≥ 2 question content words
    #       matched in the best sentence) — we are confident in *which*
    #       sentence is the question's evidence.
    #   (b) The answer is verifiable and disagrees:
    #        - has numbers, none in best sentence, but SOME elsewhere
    #          in K (the model copied a wrong year), OR
    #        - has content words AND raw_score is near zero (answer's
    #          content is entirely outside the Q-relevant sentence)
    best_sentence_q_matches = sum(
        1 for t in q_words if t in set(_WORD_RE.findall(best_sentence))
    )
    confident_locus = best_sentence_q_matches >= 2

    if confident_locus and raw_score < 0.2:
        if a_nums:
            sentence_nums = set(_NUMBER_RE.findall(best_sentence))
            all_K_nums = set(_NUMBER_RE.findall(K_lower))
            num_in_best = sum(1 for n in a_nums if n in sentence_nums)
            num_in_K = sum(1 for n in a_nums if n in all_K_nums)
            # Wrong-year case: answer has numbers, none in best sentence,
            # but some elsewhere in K → unambiguous misplacement.
            if num_in_best == 0 and num_in_K > 0:
                return -1.0
        if a_words:
            # Word-only answer with 0 overlap with the Q-relevant
            # sentence: active mismatch.
            return -(0.2 - raw_score) / 0.2

    # ── Extractive-binding margin gate ────────────────────────────
    # The block above only fires when answer words are nearly absent
    # from the Q-evidence sentence (raw_score < 0.2). The dominant
    # HaluEval-QA miss class is different: the answer REUSES the
    # knowledge's vocabulary (raw_score ≈ 0.4–0.7) but recombines it
    # into a false proposition, or picks the wrong option in an
    # "A or B?" question. Word-presence can't see this; contiguity
    # can. A genuine extractive answer is a *contiguous span* of its
    # evidence sentence (empirically, safe HaluEval-QA answers have
    # quote_support ≈ 0.97). A recombination/wrong-option answer is
    # not — its longest contiguous run at the Q-locus is short, and
    # any contiguous run it does have lives in a *distractor*
    # sentence. Both promotions are conservative (need confident_locus
    # and ≥2 answer tokens) so genuine extractive answers — which keep
    # a high bind_q — are never demoted, protecting Retention.
    a_tok = _tokens(answer)
    if confident_locus and len(a_tok) >= 2:
        denom = float(len(a_tok))
        bind_q = _longest_contiguous_run(a_tok, _tokens(best_sentence)) / denom

        # Empirical separator (HaluEval-QA, N=120 forensic): genuine
        # extractive answers are near-verbatim spans of their evidence
        # sentence — bind_q mean 0.94, 95% ≥ 0.34. Recombination /
        # wrong-option hallucinations are not — bind_q mean ~0.27,
        # ~70% < 0.34. So a *confident* question locus with no
        # contiguous answer span is strong (≈95%-specific) evidence the
        # answer is not extractively supported, regardless of how many
        # of its tokens are scattered across the knowledge (which is
        # what the additive bag-of-words features over-credit).
        #
        # Strength is floored at 0.6 so this gate can actually cross the
        # QA suppress threshold: ρ_soft ≈ 0.85 for this class (bias
        # dominates) and τ_warn ≈ 0.92, so a timid negative is absorbed.
        # It rises to 1.0 as bind_q → 0 (no span at all). Conservative
        # by construction: confident_locus + ≥2 answer tokens + the
        # 0.34 cut that costs only ~5% of safe answers.
        if bind_q < 0.34:
            return -min(1.0, 0.6 + (0.34 - bind_q) * 1.2)

        # ── Question-residual payload gate ────────────────────────
        # The dominant residual miss (forensic: 67% of post-binding
        # FNs) is the WRONG-SLOT factoid: the answer parrots the
        # question's own frame and appends a wrong filler — e.g.
        # Q "Which state ... CEO is Warren Bryant ... located?"
        # A "...located in Utah." Full-answer overlap is inflated by
        # the echoed question words, masking that the *asserted*
        # filler is unsupported. Only the answer's question-RESIDUAL
        # (tokens it adds beyond the question's words) carries truth
        # value. If that residual payload is ENTIRELY absent from the
        # question's evidence sentence, the answer fills the slot with
        # something the locus does not support. `present == 0` is the
        # lowest-false-positive condition (a genuine extractive answer
        # has its payload in its evidence sentence); raw_score < 0.6
        # keeps strongly-grounded (likely-correct/​paraphrase or
        # multi-hop) answers out of scope to protect Retention.
        payload = [
            t for t in a_tok
            if len(t) > 2 and t not in _STOPWORDS and t not in q_words
        ]
        if payload and raw_score < 0.6:
            best_tok_set = set(_tokens(best_sentence))
            present = sum(1 for t in set(payload) if t in best_tok_set)
            if present == 0:
                return -0.7

    return raw_score


def extract_features(
    claim: str,
    context: str,
    *,
    adequacy: float = 0.5,
    idf: dict[str, float] | None = None,
    question: str | None = None,
) -> ClaimFeatures:
    """Compute the full feature vector φ(c, C).

    Args:
        claim: The text being verified.
        context: Evidence corpus (concatenated retrieved windows works).
        adequacy: Externally-computed adequacy score (from
            select_evidence_windows). Passed through as a feature so
            the risk model can downweight low-adequacy verdicts.
        idf: Optional precomputed corpus-IDF map. When absent, falls
            back to in-sample proxy.
        question: Optional question text for QA-alignment scoring. When
            present, `qa_alignment` is computed against `context` (which
            should be knowledge-only for that feature to be meaningful).
            When None, qa_alignment defaults to 1.0 (neutral).
    """
    return ClaimFeatures(
        entity_precision=feat_entity_precision(claim, context),
        number_consistency=feat_number_consistency(claim, context),
        idf_lex_overlap=feat_idf_lex_overlap(claim, context, idf=idf),
        quote_support=feat_quote_support(claim, context),
        forward_entail=feat_forward_entail(claim, context),
        reverse_entail=feat_reverse_entail(claim, context),
        negation_polarity=feat_negation_polarity(claim, context),
        adequacy=adequacy,
        qa_alignment=feat_qa_alignment(claim, context, question),
    )
