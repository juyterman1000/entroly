//! IOS — Information-Optimal Selection (v2: full-dominance fix)
//!
//! Two novel algorithms that compose into a single selection pass:
//!
//! 1. **Submodular Diversity Selection (SDS)**
//!    Standard knapsack treats value(A ∪ B) = value(A) + value(B).
//!    Reality: value(A ∪ B) ≤ value(A) + value(B) — information has
//!    diminishing returns. SDS penalizes redundancy using SimHash
//!    Hamming distance as a proxy for content overlap.
//!
//!    Algorithm: Lazy greedy with diversity penalty.
//!    Approximation: the (1 - 1/e) ≈ 0.63 ratio (Feige 1998 for cardinality;
//!    Sviridenko 2004 for knapsack) applies to *monotone* submodular
//!    objectives. A subtractive redundancy penalty can violate monotonicity
//!    — f(S ∪ {x}) may fall below f(S) for a near-duplicate x. For that
//!    regime, the best known polynomial-time ratio under a knapsack is
//!    0.325 (Chekuri, Vondrák, Zenklusen 2011 via continuous greedy +
//!    rounding). This implementation's empirical performance is measured in
//!    `bench/compare.py`; no tight worst-case ratio is claimed here.
//!
//! 2. **Multi-Resolution Knapsack (MRK)**
//!    Each fragment has up to 3 representations:
//!   - Full: ~100% information, ~100% tokens
//!   - Skeleton: ~70% information, ~20% tokens
//!   - Reference: ~15% information, ~2% tokens
//!
//!    This is the Multiple Choice Knapsack Problem (MCKP).
//!    Combined with SDS, each candidate is a (fragment, resolution)
//!    pair with resolution-adjusted value and diversity penalty.
//!
use crate::dedup::{simhash_cosine, simhash_cosine_lcb};

/// Confidence level for the redundancy penalty: a fragment is suppressed only
/// when it is similar to something already selected at 1 - alpha confidence,
/// jointly across every comparison drawn. Chosen to fail toward keeping
/// information, since an unjustified penalty discards evidence while an
/// unjustified reprieve only costs tokens.
const DIVERSITY_ALPHA: f64 = 0.05;
use crate::fragment::{compute_relevance, ContextFragment};
use std::collections::HashMap;

/// Resolution level for a selected fragment.
///
/// Hierarchical Context Synthesis: four abstraction levels that mirror
/// how engineers actually hold code in working memory.
///   Full      → raw code (100% info, 100% tokens)
///   Skeleton  → signatures + structure (70% info, 20% tokens)
///   Belief    → vault knowledge graph summary (50% info, 10-15% tokens)
///   Reference → file path only (15% info, 2% tokens)
// `Ord` is derived so resolution can act as a stable tie-break key when two
// candidates score identically; see the candidate sort in `sds_select`. The
// sibling `conversation_pruner::Resolution` already derives it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Resolution {
    /// Full content — maximum information, maximum tokens
    Full,
    /// Skeleton — signatures + structure, ~20% tokens
    Skeleton,
    /// Belief — vault-compiled summary with wiki-linked context, ~10-15% tokens
    Belief,
    /// Reference — file path + function name only, ~2% tokens
    Reference,
}

/// Configurable information retention factors for each resolution level.
/// These control the value/cost trade-off in multi-resolution knapsack.
/// Tunable via tuning_config.json → autotune daemon.
///
/// Belief factor (0.50): vault beliefs capture ~50% of a file's information
/// value at ~10-15% token cost. This is the key ratio for Hierarchical
/// Context Synthesis — beliefs provide architectural understanding that
/// makes raw code fragments cheaper to comprehend.
pub struct InfoFactors {
    pub skeleton: f64,  // default 0.70
    pub belief: f64,    // default 0.50
    pub reference: f64, // default 0.15
}

impl Default for InfoFactors {
    fn default() -> Self {
        InfoFactors {
            skeleton: 0.70,
            belief: 0.50,
            reference: 0.15,
        }
    }
}

impl Resolution {
    /// Information retention factor for this resolution level.
    fn info_factor(&self, factors: &InfoFactors) -> f64 {
        match self {
            Resolution::Full => 1.0,
            Resolution::Skeleton => factors.skeleton,
            Resolution::Belief => factors.belief,
            Resolution::Reference => factors.reference,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Resolution::Full => "full",
            Resolution::Skeleton => "skeleton",
            Resolution::Belief => "belief",
            Resolution::Reference => "reference",
        }
    }
}

/// A candidate item for the SDS+MRK optimizer.
/// Each fragment generates 1-4 candidates (one per resolution).
struct Candidate {
    frag_idx: usize, // Index into the fragments array
    resolution: Resolution,
    token_cost: u32, // Tokens for this resolution
    base_value: f64, // relevance × info_factor (before diversity penalty)
    /// Fingerprint for diversity computation, or `None` when the fragment has
    /// no content-derived SimHash (`ContextFragment::has_simhash == false`,
    /// e.g. shadow stubs).
    ///
    /// This is an `Option` rather than a bare `u64` on purpose. A fragment
    /// without a fingerprint carries `simhash == 0`, and `0` is a perfectly
    /// reachable real fingerprint, so comparing raw values made every
    /// fingerprint-less fragment an exact "duplicate" of every other one —
    /// the precise failure `fragment.rs` warns about when it says stubs are
    /// "excluded from all similarity ops".
    simhash: Option<u64>,
}

/// Order a selection by relevance, leaving the pinned prefix in place.
///
/// Selection order is chosen by value-per-token, which is right for filling a
/// budget and wrong for presentation: a fragment one token shorter outranks a
/// more relevant one. Measured on a two-file project for the query
/// "who verifies the password":
///
/// ```text
///     auth.py     24 tokens  relevance 0.58640  density 0.024308
///     billing.py  23 tokens  relevance 0.57694  density 0.024960
/// ```
///
/// Both exits of `ios_select` must apply this. The best-fit fast path returns
/// early when everything fits, which is the common case for small projects --
/// exactly where the wrong order is most visible.
///
/// Ties break on fragment index so prompt prefixes stay byte-stable.
#[allow(clippy::too_many_arguments)]
fn order_by_relevance(
    selections: &mut [(usize, Resolution)],
    pinned_len: usize,
    fragments: &[ContextFragment],
    feedback_mults: &HashMap<String, f64>,
    w_recency: f64,
    w_frequency: f64,
    w_semantic: f64,
    w_entropy: f64,
) {
    let split_at = pinned_len.min(selections.len());
    let (_, tail) = selections.split_at_mut(split_at);
    tail.sort_by(|a, b| {
        let rel = |i: usize| {
            let fm = feedback_mults
                .get(&fragments[i].fragment_id)
                .copied()
                .unwrap_or(1.0);
            compute_relevance(
                &fragments[i],
                w_recency,
                w_frequency,
                w_semantic,
                w_entropy,
                fm,
            )
        };
        rel(b.0)
            .partial_cmp(&rel(a.0))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
}

/// Result of the IOS selection.
pub struct SdsResult {
    /// (fragment_index, chosen_resolution) pairs
    pub selections: Vec<(usize, Resolution)>,
    pub total_tokens: u32,
    pub(crate) _total_value: f64,
    pub diversity_score: f64, // Average pairwise diversity of selected set
    pub curvature: SelectionCurvature,
}

/// Curvature certificate for IOS selection (Pillar IV).
///
/// The SDS diversity penalty f(S∪{x}) = base_value(x) · diversity(x,S)
/// is not monotone: adding a near-duplicate can decrease the marginal
/// density below what was available before it was selected. The curvature
/// parameter α captures how far from monotone the objective was during
/// this particular selection.
///
/// For a monotone submodular objective, the greedy algorithm achieves
/// (1-1/e) ≈ 0.632 of optimal. With curvature α ∈ [0,1], the guarantee
/// weakens to (1-1/e)(1 - α). α = 0 is fully monotone; α = 1 is the
/// worst case where adding any item could zero out the objective.
///
/// Production value: when curvature is high, the diversity penalty is
/// costing more than it's saving. The system should consider relaxing
/// DIVERSITY_ALPHA or reducing the diversity floor.
#[derive(Clone, Debug)]
pub struct SelectionCurvature {
    /// Maximum diversity penalty observed: max_over_steps(1 - diversity_factor).
    /// 0.0 = all additions were to fully novel items.
    /// 1.0 = a near-exact duplicate was considered.
    pub max_penalty: f64,
    /// Mean diversity factor across all greedy steps.
    pub mean_diversity: f64,
    /// Number of candidates whose diversity factor fell below 0.5 (high overlap).
    pub high_overlap_count: u32,
    /// Total greedy steps taken.
    pub steps: u32,
    /// Effective curvature α ∈ [0,1]: the mean penalty weighted by
    /// how much value it displaced.
    pub alpha: f64,
    /// Stable rank of the selected set: how many independent fragments the
    /// selection is actually worth. Lies in [1, m] for a selection of size m;
    /// 0.0 when nothing was selected. See `compute_stable_rank`.
    pub stable_rank: f64,
    /// Number of selected fragments that carried a fingerprint, i.e. the `m`
    /// that `stable_rank` is out of. Fragments without a fingerprint are
    /// excluded from both, so this is not the selection length. Reported
    /// because `stable_rank` alone is not interpretable: 31 is a good result
    /// out of 34 and a bad one out of 200.
    pub fingerprinted_count: u32,
}

/// Compute the diversity factor for a candidate given the current selected set.
///
/// diversity = 1 - max_similarity(candidate, selected_set)
///
/// Similarity comes from `dedup::simhash_cosine`, the estimator SimHash
/// actually supports. This previously used `1 - hamming/64`, which reports two
/// unrelated fragments (near-orthogonal, hamming ~= 32) as **0.5 similar**
/// rather than 0 — so every candidate was silently penalised by roughly half
/// regardless of what had already been selected, and the diversity term was
/// close to a constant multiplier. Measured over 1.1M real fragment pairs the
/// old form had MAE 0.502 against TF-cosine ground truth; the current one has
/// MAE 0.080, which is the sampling-noise floor for a 64-bit fingerprint.
///
/// When the selected set is empty, diversity = 1.0 (no penalty).
///
/// A candidate with no fingerprint also gets 1.0. We have no evidence it
/// duplicates anything, and penalising on absent evidence would discard
/// information — the costly direction of the error.
///
/// Returns a value in [0, 1] where:
///   1.0 = completely novel information
///   0.0 = identical to something already selected
#[inline]
fn diversity_factor(candidate_hash: Option<u64>, selected_hashes: &[u64]) -> f64 {
    let Some(candidate_hash) = candidate_hash else {
        return 1.0;
    };
    if selected_hashes.is_empty() {
        return 1.0;
    }

    // Union-bounded over the k comparisons being maximised. Without this, the
    // maximum of k noisy estimates drifts upward like sigma*sqrt(2 ln k), so
    // the penalty grows with how much has already been selected rather than
    // with actual redundancy. See `dedup::simhash_cosine_lcb`.
    let comparisons = selected_hashes.len();
    let max_sim = selected_hashes
        .iter()
        .map(|&h| simhash_cosine_lcb(candidate_hash, h, comparisons, DIVERSITY_ALPHA))
        .fold(0.0_f64, f64::max);

    // Diversity = 1 - max_similarity
    // The caller clamps this to `diversity_floor` — even similar fragments
    // carry SOME new information.
    1.0 - max_sim
}

/// Compute average pairwise diversity from SimHash fingerprints.
///
/// diversity = mean over all pairs of (1 - simhash_cosine).
/// Returns 1.0 when ≤ 1 hash (trivially diverse).
///
/// This previously averaged `hamming/64`, which caps out at ~0.5 for a set of
/// mutually unrelated fragments — so a maximally diverse selection reported
/// `diversity_score = 0.5` instead of 1.0. That value is user-facing, so the
/// reported number was not on the scale its own documentation claimed.
fn compute_pairwise_diversity(hashes: &[u64]) -> f64 {
    if hashes.len() <= 1 {
        return 1.0;
    }
    let n = hashes.len();
    let mut pair_count = 0usize;
    let mut diversity_sum = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            diversity_sum += 1.0 - simhash_cosine(hashes[i], hashes[j]);
            pair_count += 1;
        }
    }
    if pair_count > 0 {
        (diversity_sum / pair_count as f64 * 10000.0).round() / 10000.0
    } else {
        1.0
    }
}

/// Stable rank of the selected set's similarity matrix.
///
/// For the m×m matrix `G` with `G_ii = 1` and `G_ij = simhash_cosine(i, j)`,
///
/// ```text
///   stable_rank = (tr G)^2 / tr(G^2) = m^2 / (m + E),
///   E = 2 * sum_{i<j} G_ij^2
/// ```
///
/// The identity `tr(G^2) = ||G||_F^2 = m + E` holds because `G` is symmetric
/// with unit diagonal, so the whole quantity comes out of the same pair loop
/// `compute_pairwise_diversity` already walks — no new similarity model.
///
/// ## Why this and not `diversity_score`
///
/// `diversity_score` is a mean over pairs, and a mean cannot see clustering.
/// Forty fragments forming four tight clusters of ten and forty fragments
/// spread evenly can report the *same* mean diversity; the first selection is
/// worth about four independent things and the second about forty. Stable rank
/// separates them because squaring the similarities makes concentrated mass
/// dominate, which is exactly the structure a mean averages away.
///
/// Range is `[1, m]`, and both ends are tight: `m` identical fragments give
/// `m^2 / (m + m(m-1)) = 1`, and `m` mutually orthogonal fragments give
/// `m^2 / m = m`.
///
/// ## Bias direction
///
/// `simhash_cosine` clamps negative estimates to zero, which injects positive
/// bias on near-orthogonal pairs (measured +0.079; see the note in
/// `dedup::simhash_cosine_lcb`). Positive bias in `G_ij` inflates `E`, which
/// *deflates* `m^2 / (m + E)`. The reported figure therefore understates the
/// true stable rank rather than overstating it — the safe direction for a
/// diversity claim.
///
/// This is a descriptive statistic of the estimated matrix, not a certified
/// bound on the underlying one. A certificate would need `G` to be PSD, which
/// a clamped per-pair estimator does not guarantee.
fn compute_stable_rank(hashes: &[u64]) -> f64 {
    let m = hashes.len();
    if m == 0 {
        return 0.0;
    }
    if m == 1 {
        return 1.0;
    }
    let mut energy = 0.0_f64;
    for i in 0..m {
        for j in (i + 1)..m {
            let s = simhash_cosine(hashes[i], hashes[j]);
            energy += s * s;
        }
    }
    let m_f = m as f64;
    // tr(G^2) >= m, so the quotient is always defined and at most m.
    let frobenius_sq = m_f + 2.0 * energy;
    ((m_f * m_f / frobenius_sq) * 10000.0).round() / 10000.0
}

/// IOS: Information-Optimal Selection
///
/// Combines Submodular Diversity Selection with Multi-Resolution Knapsack
/// in a single greedy pass.
///
/// Algorithm:
///   1. Generate candidates: each fragment × {full, skeleton, reference}
///   2. Separate pinned fragments (always full resolution, always included)
///   3. Greedy loop (greedy-by-density with diversity penalty):
///      - compute marginal_value = base_value × diversity_factor(hash)
///      - select candidate with highest marginal_value / token_cost
///      - remove all other resolutions of the same fragment
///      - update selected_hashes; repeat until budget exhausted
///
/// Complexity: O(N × K) where N = candidates, K = selected count
/// Typically K << N, so this is effectively O(N log N) after initial sort.
#[allow(clippy::too_many_arguments)]
pub fn ios_select(
    fragments: &[ContextFragment],
    token_budget: u32,
    w_recency: f64,
    w_frequency: f64,
    w_semantic: f64,
    w_entropy: f64,
    feedback_mults: &HashMap<String, f64>,
    enable_diversity: bool,
    enable_multi_resolution: bool,
    info_factors: &InfoFactors,
    diversity_floor: f64,
    min_candidate_value: f64,
) -> SdsResult {
    if fragments.is_empty() {
        return SdsResult {
            selections: vec![],
            total_tokens: 0,
            _total_value: 0.0,
            diversity_score: 1.0,
            curvature: SelectionCurvature {
                max_penalty: 0.0,
                mean_diversity: 1.0,
                high_overlap_count: 0,
                steps: 0,
                alpha: 0.0,
                stable_rank: 0.0,
                fingerprinted_count: 0,
            },
        };
    }

    // ── Phase 1: Budget-aware pinned fragment selection ──
    //
    // Old behavior: unconditionally include ALL pinned fragments, then use
    // remaining budget for everything else. This breaks when pinned tokens
    // exceed the budget (e.g. monorepos with 90 pinned files = 167K tokens
    // on an 8K budget → remaining_budget = 0 → zero query-relevant selection).
    //
    // New behavior: pinned fragments get a STRONG relevance boost (10x) but
    // are still subject to budget constraint. When total pinned tokens fit
    // within 50% of budget, include all (fast path). When they don't, treat
    // them as high-priority candidates in the greedy selection.
    //
    // This preserves the intent (critical files are almost always included)
    // while preventing budget blowout in large monorepos.
    let pinned_cap = token_budget / 2; // Reserve at least 50% for query-relevant fragments

    let mut all_pinned: Vec<(usize, u32)> = Vec::new(); // (index, token_count)
    let mut total_pinned_tokens: u32 = 0;
    for (i, frag) in fragments.iter().enumerate() {
        if frag.is_pinned {
            all_pinned.push((i, frag.token_count));
            total_pinned_tokens += frag.token_count;
        }
    }

    let mut pinned: Vec<(usize, Resolution)> = Vec::new();
    let mut pinned_tokens: u32 = 0;
    let mut pinned_hashes: Vec<u64> = Vec::new();
    let mut demoted_pinned: Vec<usize> = Vec::new(); // Pinned fragments that didn't fit → become high-priority candidates

    if total_pinned_tokens <= pinned_cap {
        // Fast path: all pinned fragments fit within cap
        for &(i, _tc) in &all_pinned {
            pinned.push((i, Resolution::Full));
            pinned_tokens += fragments[i].token_count;
            if fragments[i].has_simhash {
                pinned_hashes.push(fragments[i].simhash);
            }
        }
    } else {
        // Budget pressure: select top pinned fragments by relevance density,
        // demote the rest to high-priority candidates.
        // Sort pinned by entropy score (descending) as a proxy for importance.
        let mut scored_pinned: Vec<(usize, f64)> = all_pinned
            .iter()
            .map(|&(i, _)| {
                let frag = &fragments[i];
                let fm = feedback_mults
                    .get(&frag.fragment_id)
                    .copied()
                    .unwrap_or(1.0);
                let score =
                    compute_relevance(frag, w_recency, w_frequency, w_semantic, w_entropy, fm);
                (i, score)
            })
            .collect();
        // Total order, ties broken on fragment id. Pinned selection runs
        // against a hard cap, so a tie at the cap boundary decides which
        // critical evidence is pinned and which is dropped -- the last place
        // that should depend on sort partitioning. See `knapsack.rs`.
        scored_pinned.sort_unstable_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| fragments[a.0].fragment_id.cmp(&fragments[b.0].fragment_id))
        });

        for &(i, score) in &scored_pinned {
            let tc = fragments[i].token_count;
            if pinned_tokens + tc <= pinned_cap && score > 0.0 {
                pinned.push((i, Resolution::Full));
                pinned_tokens += tc;
                if fragments[i].has_simhash {
                    pinned_hashes.push(fragments[i].simhash);
                }
            } else {
                demoted_pinned.push(i);
            }
        }
    }

    let remaining_budget = token_budget.saturating_sub(pinned_tokens);

    // ── Phase 2: Generate candidates ──
    // Demoted pinned fragments enter as candidates with a 5x relevance boost.
    // This ensures they're strongly preferred without being unconditional.
    let demoted_set: std::collections::HashSet<usize> = demoted_pinned.iter().copied().collect();
    let pinned_included_set: std::collections::HashSet<usize> =
        pinned.iter().map(|&(i, _)| i).collect();
    let mut candidates: Vec<Candidate> = Vec::new();

    for (i, frag) in fragments.iter().enumerate() {
        // Skip fragments already included as pinned
        if pinned_included_set.contains(&i) {
            continue;
        }

        let fm = feedback_mults
            .get(&frag.fragment_id)
            .copied()
            .unwrap_or(1.0);
        let mut relevance =
            compute_relevance(frag, w_recency, w_frequency, w_semantic, w_entropy, fm);

        // Demoted pinned fragments get 5x boost — strongly preferred but budget-constrained
        if demoted_set.contains(&i) {
            relevance *= 5.0;
        }

        if relevance <= 0.0 || frag.token_count == 0 {
            continue;
        }

        // Full resolution — always available
        candidates.push(Candidate {
            frag_idx: i,
            resolution: Resolution::Full,
            token_cost: frag.token_count,
            base_value: relevance * Resolution::Full.info_factor(info_factors),
            simhash: frag.has_simhash.then_some(frag.simhash),
        });

        if enable_multi_resolution {
            // Skeleton resolution — only if skeleton was extracted
            if let Some(skel_tc) = frag.skeleton_token_count {
                let skel_value = relevance * Resolution::Skeleton.info_factor(info_factors);
                if skel_tc < frag.token_count && skel_value >= min_candidate_value {
                    candidates.push(Candidate {
                        frag_idx: i,
                        resolution: Resolution::Skeleton,
                        token_cost: skel_tc,
                        base_value: skel_value,
                        simhash: frag.has_simhash.then_some(frag.simhash),
                    });
                }
            }

            // Belief resolution — only if vault belief was loaded during ingest.
            // Beliefs sit between Skeleton (structural) and Reference (path-only):
            // they capture semantic understanding at ~10-15% token cost.
            // The IOS greedy selector will pick the resolution with the best
            // value/cost ratio, so beliefs naturally win when budget is tight
            // but the file is important enough that a bare reference isn't enough.
            if let Some(belief_tc) = frag.belief_token_count {
                let belief_value = relevance * Resolution::Belief.info_factor(info_factors);
                if belief_tc < frag.token_count && belief_value >= min_candidate_value {
                    candidates.push(Candidate {
                        frag_idx: i,
                        resolution: Resolution::Belief,
                        token_cost: belief_tc,
                        base_value: belief_value,
                        simhash: frag.has_simhash.then_some(frag.simhash),
                    });
                }
            }

            // Reference resolution — always available, very cheap
            // Cost: ~5 tokens for "file:source.py" reference line
            let ref_tokens = (frag.source.len() as u32 / 4).clamp(3, 10);
            let ref_value = relevance * Resolution::Reference.info_factor(info_factors);
            if ref_value >= min_candidate_value {
                candidates.push(Candidate {
                    frag_idx: i,
                    resolution: Resolution::Reference,
                    token_cost: ref_tokens,
                    base_value: ref_value,
                    simhash: frag.has_simhash.then_some(frag.simhash),
                });
            }
        }
    }

    if candidates.is_empty() || remaining_budget == 0 {
        let _total_value: f64 = pinned
            .iter()
            .map(|&(i, _)| {
                let fm = feedback_mults
                    .get(&fragments[i].fragment_id)
                    .copied()
                    .unwrap_or(1.0);
                compute_relevance(
                    &fragments[i],
                    w_recency,
                    w_frequency,
                    w_semantic,
                    w_entropy,
                    fm,
                )
            })
            .sum();
        return SdsResult {
            selections: pinned,
            total_tokens: pinned_tokens,
            _total_value: (_total_value * 10000.0).round() / 10000.0,
            diversity_score: 1.0,
            curvature: SelectionCurvature {
                max_penalty: 0.0,
                mean_diversity: 1.0,
                high_overlap_count: 0,
                steps: 0,
                alpha: 0.0,
                // Measured, not assumed 1.0 like `diversity_score` above: a
                // pinned-only selection can still be highly redundant, and
                // that is worth reporting rather than asserting away.
                stable_rank: compute_stable_rank(&pinned_hashes),
                fingerprinted_count: pinned_hashes.len() as u32,
            },
        };
    }

    // ── Best-Fit Fast Path ──────────────────────────────────────────
    // Best-fit-decreasing bin packing: when
    // ALL non-pinned fragments fit at full resolution, skip the
    // O(N×K) greedy loop. Common for small codebases or generous
    // ECDB budgets. Reduces to O(N).
    // ────────────────────────────────────────────────────────────────
    {
        let mut full_total: u32 = 0;
        let mut seen_frag = vec![false; fragments.len()];
        for c in &candidates {
            if c.resolution == Resolution::Full && !seen_frag[c.frag_idx] {
                seen_frag[c.frag_idx] = true;
                full_total += c.token_cost;
            }
        }
        if full_total <= remaining_budget && full_total > 0 {
            let mut selections = pinned.clone();
            let mut fast_tokens = pinned_tokens;
            let mut fast_value: f64 = selections
                .iter()
                .map(|&(i, _)| {
                    let fm = feedback_mults
                        .get(&fragments[i].fragment_id)
                        .copied()
                        .unwrap_or(1.0);
                    compute_relevance(
                        &fragments[i],
                        w_recency,
                        w_frequency,
                        w_semantic,
                        w_entropy,
                        fm,
                    )
                })
                .sum();
            let mut fast_hashes: Vec<u64> = pinned_hashes.clone();

            for c in &candidates {
                if c.resolution == Resolution::Full {
                    selections.push((c.frag_idx, Resolution::Full));
                    fast_tokens += c.token_cost;
                    fast_value += c.base_value;
                    if let Some(fp) = c.simhash {
                        fast_hashes.push(fp);
                    }
                }
            }

            order_by_relevance(
                &mut selections,
                pinned.len(),
                fragments,
                feedback_mults,
                w_recency,
                w_frequency,
                w_semantic,
                w_entropy,
            );
            return SdsResult {
                selections,
                total_tokens: fast_tokens,
                _total_value: (fast_value * 10000.0).round() / 10000.0,
                diversity_score: compute_pairwise_diversity(&fast_hashes),
                curvature: SelectionCurvature {
                    max_penalty: 0.0,
                    mean_diversity: 1.0,
                    high_overlap_count: 0,
                    steps: 0,
                    alpha: 0.0,
                    stable_rank: compute_stable_rank(&fast_hashes),
                    fingerprinted_count: fast_hashes.len() as u32,
                },
            };
        }
    }

    // ── Phase 3: Greedy SDS+MRK selection ──
    // Captured before `pinned` is moved: Phase 4 re-sorts only the
    // non-pinned tail, so it needs to know where the tail starts.
    let pinned_len = pinned.len();
    let mut selected: Vec<(usize, Resolution)> = pinned;
    let mut selected_hashes: Vec<u64> = pinned_hashes;
    let mut selected_frags: Vec<bool> = vec![false; fragments.len()]; // Track which fragment_idx is selected
    let mut budget_used = pinned_tokens;
    let mut _total_value: f64 = selected
        .iter()
        .map(|&(i, _)| {
            let fm = feedback_mults
                .get(&fragments[i].fragment_id)
                .copied()
                .unwrap_or(1.0);
            compute_relevance(
                &fragments[i],
                w_recency,
                w_frequency,
                w_semantic,
                w_entropy,
                fm,
            )
        })
        .sum();

    // Mark pinned as selected
    for &(idx, _) in &selected {
        selected_frags[idx] = true;
    }

    // ── Curvature tracking (Pillar IV) ──
    let mut curv_max_penalty = 0.0_f64;
    let mut curv_div_sum = 0.0_f64;
    let mut curv_weighted_penalty_sum = 0.0_f64;
    let mut curv_weighted_value_sum = 0.0_f64;
    let mut curv_high_overlap = 0_u32;
    let mut curv_steps = 0_u32;

    // Pre-sort candidates by base_value/cost density for faster convergence
    // (The diversity penalty will reorder, but this is a good initial ordering)
    //
    // "Initial ordering only" is not a reason to leave it unordered on ties.
    // The greedy loop below consumes this order, and the subtractive diversity
    // penalty is computed against what has already been selected -- so which of
    // two equally dense candidates is considered first changes what the other
    // one is penalised against, and the divergence compounds rather than
    // washing out. The key is (frag_idx, resolution) because one fragment
    // appears once per resolution in this multiple-choice knapsack, so frag_idx
    // alone is not unique. See `knapsack.rs` for the defect this fixes.
    candidates.sort_unstable_by(|a, b| {
        let da = a.base_value / a.token_cost.max(1) as f64;
        let db = b.base_value / b.token_cost.max(1) as f64;
        db.total_cmp(&da)
            .then_with(|| {
                fragments[a.frag_idx]
                    .fragment_id
                    .cmp(&fragments[b.frag_idx].fragment_id)
            })
            .then_with(|| a.resolution.cmp(&b.resolution))
    });

    loop {
        let budget_remaining = token_budget.saturating_sub(budget_used);
        if budget_remaining == 0 {
            break;
        }

        // Find the best candidate considering diversity
        let mut best_density = 0.0_f64;
        let mut best_idx: Option<usize> = None;

        // Precompute: for each fragment, does the full resolution fit?
        // If so, skip lower resolutions (full dominates when it fits).
        let mut full_fits: Vec<bool> = vec![false; fragments.len()];
        for cand in candidates.iter() {
            if cand.resolution == Resolution::Full
                && !selected_frags[cand.frag_idx]
                && cand.token_cost <= budget_remaining
            {
                full_fits[cand.frag_idx] = true;
            }
        }

        for (ci, cand) in candidates.iter().enumerate() {
            // Skip if this fragment already has a resolution selected
            if selected_frags[cand.frag_idx] {
                continue;
            }
            // Skip if doesn't fit
            if cand.token_cost > budget_remaining {
                continue;
            }
            // Skip lower resolutions when full fits — full dominates
            // because it carries strictly more information.
            if cand.resolution != Resolution::Full && full_fits[cand.frag_idx] {
                continue;
            }

            let div = if enable_diversity {
                diversity_factor(cand.simhash, &selected_hashes).max(diversity_floor)
            } else {
                1.0
            };

            let marginal_value = cand.base_value * div;
            let density = marginal_value / cand.token_cost.max(1) as f64;

            if density > best_density {
                best_density = density;
                best_idx = Some(ci);
            }
        }

        match best_idx {
            Some(ci) => {
                let cand = &candidates[ci];
                let div = if enable_diversity {
                    diversity_factor(cand.simhash, &selected_hashes).max(diversity_floor)
                } else {
                    1.0
                };
                let penalty = 1.0 - div;
                curv_max_penalty = curv_max_penalty.max(penalty);
                curv_div_sum += div;
                curv_weighted_penalty_sum += penalty * cand.base_value;
                curv_weighted_value_sum += cand.base_value;
                if div < 0.5 {
                    curv_high_overlap += 1;
                }
                curv_steps += 1;

                selected.push((cand.frag_idx, cand.resolution));
                if let Some(fp) = cand.simhash {
                    selected_hashes.push(fp);
                }
                selected_frags[cand.frag_idx] = true;
                budget_used += cand.token_cost;
                _total_value += cand.base_value;
            }
            None => break, // No more candidates fit
        }
    }

    // ── Phase 4: Present by relevance, pack by density ──────────────
    // Shared with the best-fit fast path; see `order_by_relevance`.
    order_by_relevance(
        &mut selected,
        pinned_len,
        fragments,
        feedback_mults,
        w_recency,
        w_frequency,
        w_semantic,
        w_entropy,
    );

    // ── Phase 5: Compute diversity score of final selection ──
    let diversity_score = compute_pairwise_diversity(&selected_hashes);

    let mean_div = if curv_steps > 0 { curv_div_sum / curv_steps as f64 } else { 1.0 };
    let alpha = if curv_weighted_value_sum > 1e-12 {
        (curv_weighted_penalty_sum / curv_weighted_value_sum).clamp(0.0, 1.0)
    } else {
        0.0
    };

    SdsResult {
        selections: selected,
        total_tokens: budget_used,
        _total_value: (_total_value * 10000.0).round() / 10000.0,
        diversity_score,
        curvature: SelectionCurvature {
            max_penalty: (curv_max_penalty * 10000.0).round() / 10000.0,
            mean_diversity: (mean_div * 10000.0).round() / 10000.0,
            high_overlap_count: curv_high_overlap,
            steps: curv_steps,
            alpha: (alpha * 10000.0).round() / 10000.0,
            stable_rank: compute_stable_rank(&selected_hashes),
            fingerprinted_count: selected_hashes.len() as u32,
        },
    }
}

#[cfg(test)]
mod tests {
    fn sds_lcg(state: &mut u64) -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 33) as f64) / ((1u64 << 31) as f64)
    }

    /// The two invariants IOS must hold whatever the objective does.
    ///
    /// No worst-case ratio is claimed for this selector and none is asserted
    /// here. What must hold regardless of approximation quality:
    ///
    ///   1. the selection fits the token budget -- the sibling solver shipped a
    ///      fast path that compared a probability-weighted token count against
    ///      a hard budget and returned a set costing 52 against a budget of 50,
    ///      so this is not hypothetical;
    ///   2. at most one resolution per fragment. This is a multiple-choice
    ///      knapsack: two resolutions of the same fragment would put the same
    ///      content in the context twice and charge for both.
    #[test]
    fn ios_selection_is_feasible_and_picks_one_resolution_per_fragment() {
        let mut seed = 0xA11CE_u64;

        for case in 0..50 {
            let n = 4 + (case % 12);
            let fragments: Vec<ContextFragment> = (0..n)
                .map(|k| {
                    // Mixed magnitudes: tiny items beside large ones are the
                    // shape that broke feasibility in `knapsack.rs`.
                    let tokens = if sds_lcg(&mut seed) < 0.4 {
                        1 + (sds_lcg(&mut seed) * 5.0) as u32
                    } else {
                        20 + (sds_lcg(&mut seed) * 300.0) as u32
                    };
                    // Some near-duplicates, so the redundancy penalty is live.
                    let content = if k % 3 == 0 {
                        "shared duplicated body text".to_string()
                    } else {
                        format!("distinct body {k} with its own words")
                    };
                    let mut f = make_frag(&format!("f{k:03}"), &content, tokens, "s");
                    f.semantic_score = sds_lcg(&mut seed);
                    f.frequency_score = sds_lcg(&mut seed);
                    f
                })
                .collect();

            let total: u32 = fragments.iter().map(|f| f.token_count).sum();
            let budget = ((total as f64) * (0.2 + 0.5 * sds_lcg(&mut seed))) as u32;
            if budget == 0 {
                continue;
            }

            for (diversity, multi_res) in [(false, false), (true, false), (true, true)] {
                let r = ios_select(
                    &fragments,
                    budget,
                    0.25, 0.25, 0.25, 0.25,
                    &empty_feedback(),
                    diversity,
                    multi_res,
                    &default_factors(),
                    DEFAULT_DIV_FLOOR,
                    DEFAULT_MIN_CANDIDATE_VALUE,
                );

                assert!(
                    r.total_tokens <= budget,
                    "case {case} (diversity={diversity}, multi_res={multi_res}):                      selected {} tokens against a budget of {budget}",
                    r.total_tokens
                );

                let mut seen: Vec<usize> = r.selections.iter().map(|&(i, _)| i).collect();
                let before = seen.len();
                seen.sort_unstable();
                seen.dedup();
                assert_eq!(
                    seen.len(), before,
                    "case {case} (diversity={diversity}, multi_res={multi_res}):                      a fragment was selected at more than one resolution"
                );

                // `total_tokens` must describe the selection it ships with.
                assert!(
                    r.total_tokens as u64
                        <= fragments.iter().map(|f| f.token_count as u64).sum::<u64>(),
                    "case {case}: reported more tokens than exist"
                );
            }
        }
    }

    use super::*;
    use crate::dedup::simhash;
    use crate::fragment::ContextFragment;

    fn empty_feedback() -> HashMap<String, f64> {
        HashMap::new()
    }

    fn default_factors() -> InfoFactors {
        InfoFactors::default()
    }

    const DEFAULT_DIV_FLOOR: f64 = 0.1;
    const DEFAULT_MIN_CANDIDATE_VALUE: f64 = 0.05;

    fn make_frag(id: &str, content: &str, tokens: u32, source: &str) -> ContextFragment {
        let mut f = ContextFragment::new(id.into(), content.into(), tokens, source.into());
        f.simhash = simhash(content);
        // Must accompany `simhash`, or the fragment reads as fingerprint-less
        // and is excluded from diversity entirely — which would make every
        // diversity assertion below pass vacuously.
        f.has_simhash = true;
        f.recency_score = 0.9;
        f.entropy_score = 0.7;
        f
    }

    /// A fragment carrying no content-derived fingerprint, as produced by the
    /// shadow-stub path in `lib.rs`.
    fn make_stub(id: &str, content: &str, tokens: u32, source: &str) -> ContextFragment {
        let mut f = ContextFragment::new(id.into(), content.into(), tokens, source.into());
        f.recency_score = 0.9;
        f.entropy_score = 0.7;
        debug_assert!(!f.has_simhash && f.simhash == 0);
        f
    }

    #[test]
    fn test_fingerprintless_fragments_are_not_mutual_duplicates() {
        // Regression: fragments without a content-derived SimHash all carry
        // `simhash == 0`. Comparing those raw values made every such fragment
        // an exact duplicate of every other one, so the second and later stubs
        // were driven down to `diversity_floor` — a 10x value penalty applied
        // on no evidence. `fragment.rs` documents that stubs must be excluded
        // from all similarity ops.
        assert_eq!(diversity_factor(None, &[]), 1.0);
        assert_eq!(diversity_factor(None, &[0, 12345]), 1.0);

        // A real fingerprint of 0 is reachable, so the guard cannot be
        // "simhash == 0"; it must be the explicit absence of a fingerprint.
        //
        // Asserted as "essentially no diversity credit" rather than exactly
        // 0.0: that exact value depended on `simhash_cosine_lcb` returning
        // exactly 1.0 for an exact match, which was a zero-width confidence
        // interval. The bound is Wilson now, so an exact match leaves a hair
        // of diversity (~0.008). What this test is about is that `Some(0)` is
        // compared at all, unlike `None` above.
        let identical = diversity_factor(Some(0), &[0]);
        assert!(
            identical < 0.05,
            "an exact fingerprint match must earn essentially no diversity: {identical}"
        );

        // End to end: a budget that fits every stub must select every stub.
        let frags: Vec<ContextFragment> = (0..6)
            .map(|i| {
                make_stub(
                    &format!("s{i}"),
                    "fn handler() { dispatch(request, context); }",
                    10,
                    &format!("src/mod{i}.rs"),
                )
            })
            .collect();

        let result = ios_select(
            &frags,
            10_000,
            0.3,
            0.25,
            0.25,
            0.2,
            &empty_feedback(),
            true,
            true,
            &default_factors(),
            DEFAULT_DIV_FLOOR,
            DEFAULT_MIN_CANDIDATE_VALUE,
        );

        assert_eq!(
            result.selections.len(),
            frags.len(),
            "fingerprint-less fragments were suppressed as false duplicates"
        );

        // The sharp assertion. Under the bug every stub carried hash 0, so the
        // pairwise diversity of the selected set computed as 0.0 — the engine
        // reported maximum redundancy for fragments it had no fingerprint for
        // and therefore no evidence about. Absence of evidence must report as
        // "not known to be redundant" (1.0), never as "identical".
        assert_eq!(
            result.diversity_score, 1.0,
            "engine claimed redundancy among fragments it never fingerprinted"
        );
    }

    #[test]
    fn test_empty_fragments() {
        let result = ios_select(
            &[],
            1000,
            0.3,
            0.25,
            0.25,
            0.2,
            &empty_feedback(),
            true,
            true,
            &default_factors(),
            DEFAULT_DIV_FLOOR,
            DEFAULT_MIN_CANDIDATE_VALUE,
        );
        assert!(result.selections.is_empty());
        assert_eq!(result.total_tokens, 0);
    }

    #[test]
    fn test_single_fragment_selected() {
        let frags = vec![make_frag("a", "def foo(): return 42", 50, "foo.py")];
        let result = ios_select(
            &frags,
            1000,
            0.3,
            0.25,
            0.25,
            0.2,
            &empty_feedback(),
            true,
            false,
            &default_factors(),
            DEFAULT_DIV_FLOOR,
            DEFAULT_MIN_CANDIDATE_VALUE,
        );
        assert_eq!(result.selections.len(), 1);
        assert_eq!(result.selections[0], (0, Resolution::Full));
    }

    #[test]
    fn test_pinned_always_included() {
        let mut frags = vec![
            make_frag("a", "pinned content", 500, "critical.py"),
            make_frag("b", "normal content", 200, "normal.py"),
        ];
        frags[0].is_pinned = true;
        frags[0].recency_score = 0.1; // Low recency shouldn't matter for pinned

        let result = ios_select(
            &frags,
            1100,
            0.3,
            0.25,
            0.25,
            0.2,
            &empty_feedback(),
            true,
            false,
            &default_factors(),
            DEFAULT_DIV_FLOOR,
            DEFAULT_MIN_CANDIDATE_VALUE,
        );
        let selected_indices: Vec<usize> = result.selections.iter().map(|s| s.0).collect();
        assert!(
            selected_indices.contains(&0),
            "Pinned fragment must be included"
        );
    }

    #[test]
    fn test_diversity_penalizes_duplicates() {
        // Three fragments: two nearly identical, one different
        let frags = vec![
            make_frag(
                "a",
                "def calculate_tax(income, rate): return income * rate",
                100,
                "tax1.py",
            ),
            make_frag(
                "b",
                "def calculate_tax(income, rate): return income * rate * 1.0",
                100,
                "tax2.py",
            ),
            make_frag(
                "c",
                "async fn connect_database(host: str, port: int): pass",
                100,
                "db.py",
            ),
        ];

        // With diversity: should prefer a + c (diverse) over a + b (redundant)
        let result_div = ios_select(
            &frags,
            200,
            0.3,
            0.25,
            0.25,
            0.2,
            &empty_feedback(),
            true,
            false,
            &default_factors(),
            DEFAULT_DIV_FLOOR,
            DEFAULT_MIN_CANDIDATE_VALUE,
        );
        let _div_indices: Vec<usize> = result_div.selections.iter().map(|s| s.0).collect();

        // Without diversity: might select a + b (both have high relevance)
        let result_no_div = ios_select(
            &frags,
            200,
            0.3,
            0.25,
            0.25,
            0.2,
            &empty_feedback(),
            false,
            false,
            &default_factors(),
            DEFAULT_DIV_FLOOR,
            DEFAULT_MIN_CANDIDATE_VALUE,
        );

        // With diversity enabled, we should have higher diversity score
        assert!(
            result_div.diversity_score >= result_no_div.diversity_score,
            "Diversity-enabled selection should have higher diversity: {} vs {}",
            result_div.diversity_score,
            result_no_div.diversity_score
        );
    }

    #[test]
    fn test_multi_resolution_fits_more() {
        // One large fragment that barely fits, and several that don't fit at full resolution
        let mut frags = vec![
            make_frag("big", "a very important function with lots of code\ndef process():\n    x = 1\n    y = 2\n    z = x + y\n    return z", 400, "big.py"),
            make_frag("med1", "def helper_one(): pass\ndef helper_two(): pass\ndef helper_three(): pass\ndef helper_four(): pass\ndef helper_five(): pass", 200, "h1.py"),
            make_frag("med2", "class Config:\n    debug = True\n    port = 8080\n    host = 'localhost'\n    timeout = 30\n    retries = 3", 200, "h2.py"),
        ];
        // Give them skeletons
        frags[1].skeleton_content = Some("def helper_one(): ...\ndef helper_two(): ...".into());
        frags[1].skeleton_token_count = Some(40);
        frags[2].skeleton_content = Some("class Config: ...".into());
        frags[2].skeleton_token_count = Some(30);

        // Budget: 500 tokens — can fit big(400) + one skeleton but not big + two full
        let result_mr = ios_select(
            &frags,
            500,
            0.3,
            0.25,
            0.25,
            0.2,
            &empty_feedback(),
            true,
            true,
            &default_factors(),
            DEFAULT_DIV_FLOOR,
            DEFAULT_MIN_CANDIDATE_VALUE,
        );
        let result_no_mr = ios_select(
            &frags,
            500,
            0.3,
            0.25,
            0.25,
            0.2,
            &empty_feedback(),
            true,
            false,
            &default_factors(),
            DEFAULT_DIV_FLOOR,
            DEFAULT_MIN_CANDIDATE_VALUE,
        );

        // Multi-resolution should cover more fragments
        let mr_frag_count = result_mr
            .selections
            .iter()
            .map(|s| s.0)
            .collect::<std::collections::HashSet<_>>()
            .len();
        let no_mr_frag_count = result_no_mr
            .selections
            .iter()
            .map(|s| s.0)
            .collect::<std::collections::HashSet<_>>()
            .len();

        assert!(
            mr_frag_count >= no_mr_frag_count,
            "Multi-resolution should cover >= fragments: {} vs {}",
            mr_frag_count,
            no_mr_frag_count
        );
    }

    #[test]
    fn test_budget_respected() {
        let frags = vec![
            make_frag("a", "content a", 300, "a.py"),
            make_frag("b", "content b", 300, "b.py"),
            make_frag("c", "content c", 300, "c.py"),
        ];

        let result = ios_select(
            &frags,
            500,
            0.3,
            0.25,
            0.25,
            0.2,
            &empty_feedback(),
            true,
            false,
            &default_factors(),
            DEFAULT_DIV_FLOOR,
            DEFAULT_MIN_CANDIDATE_VALUE,
        );
        assert!(
            result.total_tokens <= 500,
            "Budget must be respected: {} > 500",
            result.total_tokens
        );
    }

    #[test]
    fn test_feedback_multiplier_affects_selection() {
        let frags = vec![
            make_frag(
                "good",
                "useful code fragment for processing",
                200,
                "good.py",
            ),
            make_frag("bad", "unhelpful boilerplate noise padding", 200, "bad.py"),
        ];

        let mut feedback = HashMap::new();
        feedback.insert("good".to_string(), 1.8);
        feedback.insert("bad".to_string(), 0.3);

        // Budget for only one
        let result = ios_select(
            &frags,
            250,
            0.3,
            0.25,
            0.25,
            0.2,
            &feedback,
            true,
            false,
            &default_factors(),
            DEFAULT_DIV_FLOOR,
            DEFAULT_MIN_CANDIDATE_VALUE,
        );
        let selected_indices: Vec<usize> = result.selections.iter().map(|s| s.0).collect();

        assert!(
            selected_indices.contains(&0),
            "Feedback-boosted fragment should be preferred"
        );
    }

    #[test]
    fn test_reference_resolution_very_cheap() {
        let mut frags = vec![make_frag(
            "a",
            "def big_function():\n    x = 1\n    y = 2\n    z = 3\n    return x + y + z",
            500,
            "big.py",
        )];
        frags[0].skeleton_content = Some("def big_function(): ...".into());
        frags[0].skeleton_token_count = Some(50);

        // Budget so small only reference fits
        let result = ios_select(
            &frags,
            15,
            0.3,
            0.25,
            0.25,
            0.2,
            &empty_feedback(),
            true,
            true,
            &default_factors(),
            DEFAULT_DIV_FLOOR,
            DEFAULT_MIN_CANDIDATE_VALUE,
        );
        if !result.selections.is_empty() {
            assert_eq!(
                result.selections[0].1,
                Resolution::Reference,
                "With tiny budget, reference resolution should be chosen"
            );
        }
    }

    #[test]
    fn test_diversity_score_range() {
        let frags = vec![
            make_frag("a", "machine learning neural network training", 100, "a.py"),
            make_frag("b", "kubernetes docker container deployment", 100, "b.py"),
            make_frag(
                "c",
                "react component jsx virtual dom rendering",
                100,
                "c.py",
            ),
        ];

        let result = ios_select(
            &frags,
            1000,
            0.3,
            0.25,
            0.25,
            0.2,
            &empty_feedback(),
            true,
            false,
            &default_factors(),
            DEFAULT_DIV_FLOOR,
            DEFAULT_MIN_CANDIDATE_VALUE,
        );
        assert!(
            result.diversity_score >= 0.0 && result.diversity_score <= 1.0,
            "Diversity score must be in [0, 1], got {}",
            result.diversity_score
        );
    }

    #[test]
    fn test_resolution_preference_by_budget() {
        // When budget is generous, prefer full; when tight, prefer skeleton
        let mut frags = vec![
            make_frag(
                "a",
                "def foo():\n    return 1 + 2 + 3 + 4 + 5\n",
                200,
                "a.py",
            ),
            make_frag(
                "b",
                "def bar():\n    return 6 + 7 + 8 + 9 + 10\n",
                200,
                "b.py",
            ),
        ];
        frags[0].skeleton_content = Some("def foo(): ...".into());
        frags[0].skeleton_token_count = Some(30);
        frags[1].skeleton_content = Some("def bar(): ...".into());
        frags[1].skeleton_token_count = Some(30);

        // Generous budget: both full
        let result_big = ios_select(
            &frags,
            1000,
            0.3,
            0.25,
            0.25,
            0.2,
            &empty_feedback(),
            true,
            true,
            &default_factors(),
            DEFAULT_DIV_FLOOR,
            DEFAULT_MIN_CANDIDATE_VALUE,
        );
        let full_count_big = result_big
            .selections
            .iter()
            .filter(|s| s.1 == Resolution::Full)
            .count();

        // Tight budget: mix of full + skeleton/reference
        let result_tight = ios_select(
            &frags,
            250,
            0.3,
            0.25,
            0.25,
            0.2,
            &empty_feedback(),
            true,
            true,
            &default_factors(),
            DEFAULT_DIV_FLOOR,
            DEFAULT_MIN_CANDIDATE_VALUE,
        );
        let full_count_tight = result_tight
            .selections
            .iter()
            .filter(|s| s.1 == Resolution::Full)
            .count();

        assert!(
            full_count_big >= full_count_tight,
            "Generous budget should select more full-resolution fragments"
        );
    }

    #[test]
    fn test_fast_path_selects_all_when_budget_generous() {
        // 3 fragments totalling 150 tokens, budget = 500
        // Should trigger fast path: all selected at full resolution
        let frags = vec![
            make_frag("a", "def alpha(): return 1", 50, "a.py"),
            make_frag("b", "def beta(): return 2", 50, "b.py"),
            make_frag("c", "def gamma(): return 3", 50, "c.py"),
        ];

        let result = ios_select(
            &frags,
            500,
            0.3,
            0.25,
            0.25,
            0.2,
            &empty_feedback(),
            true,
            false,
            &default_factors(),
            DEFAULT_DIV_FLOOR,
            DEFAULT_MIN_CANDIDATE_VALUE,
        );

        assert_eq!(result.selections.len(), 3, "Fast path should select all 3");
        assert!(
            result.selections.iter().all(|s| s.1 == Resolution::Full),
            "Fast path should use full resolution for all"
        );
        assert_eq!(result.total_tokens, 150);
    }

    #[test]
    fn curvature_tracks_diversity_penalty_across_greedy_steps() {
        let frags: Vec<ContextFragment> = (0..8)
            .map(|i| {
                let content = if i < 4 {
                    "shared function body with identical implementation".to_string()
                } else {
                    format!("completely unique fragment number {i} with distinct words")
                };
                make_frag(&format!("f{i}"), &content, 30, &format!("mod{i}.rs"))
            })
            .collect();

        let result = ios_select(
            &frags,
            120,
            0.3, 0.25, 0.25, 0.2,
            &empty_feedback(),
            true,
            false,
            &default_factors(),
            DEFAULT_DIV_FLOOR,
            DEFAULT_MIN_CANDIDATE_VALUE,
        );

        assert!(result.curvature.steps > 0, "greedy loop must have run");
        assert!(
            result.curvature.alpha >= 0.0 && result.curvature.alpha <= 1.0,
            "alpha must be in [0,1]: {}",
            result.curvature.alpha
        );
        assert!(
            result.curvature.mean_diversity >= 0.0 && result.curvature.mean_diversity <= 1.0,
            "mean_diversity must be in [0,1]: {}",
            result.curvature.mean_diversity
        );
        let guarantee = (1.0 - 1.0_f64.exp().recip()) * (1.0 - result.curvature.alpha);
        assert!(
            guarantee > 0.0,
            "approximation guarantee must be positive: {guarantee}"
        );
    }

    #[test]
    fn curvature_is_zero_when_all_fragments_fit() {
        let frags = vec![
            make_frag("a", "def alpha(): return 1", 20, "a.py"),
            make_frag("b", "def beta(): return 2", 20, "b.py"),
        ];
        let result = ios_select(
            &frags,
            10_000,
            0.3, 0.25, 0.25, 0.2,
            &empty_feedback(),
            true,
            false,
            &default_factors(),
            DEFAULT_DIV_FLOOR,
            DEFAULT_MIN_CANDIDATE_VALUE,
        );
        assert_eq!(result.curvature.alpha, 0.0, "no curvature when everything fits");
        assert_eq!(result.curvature.max_penalty, 0.0);
    }

    /// Deterministic 64-bit stream. Independent draws differ in ~32 of 64 bits,
    /// which is the near-orthogonal regime `simhash_cosine` maps to ~0.
    fn srank_lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state
    }

    #[test]
    fn stable_rank_is_one_for_identical_fragments() {
        let hashes = vec![0xDEAD_BEEF_CAFE_F00D_u64; 8];
        let sr = compute_stable_rank(&hashes);
        assert!(
            (sr - 1.0).abs() < 1e-9,
            "eight copies of one fragment are worth one independent thing, got {sr}"
        );
    }

    #[test]
    fn stable_rank_is_set_size_for_orthogonal_fragments() {
        // Hamming distance 32 of 64 => cos(pi/2) = 0 => no shared mass.
        let hashes = vec![0x0000_0000_0000_0000_u64, 0x0000_0000_FFFF_FFFF_u64];
        assert_eq!(
            crate::dedup::hamming_distance(hashes[0], hashes[1]),
            32,
            "test fixture must actually be orthogonal or it proves nothing"
        );
        let sr = compute_stable_rank(&hashes);
        assert!(
            (sr - 2.0).abs() < 1e-3,
            "two orthogonal fragments are worth two, got {sr}"
        );
    }

    #[test]
    fn stable_rank_stays_within_one_and_set_size() {
        let mut state = 0x5EED_1234_5678_9ABC_u64;
        for m in 1..=24usize {
            let hashes: Vec<u64> = (0..m).map(|_| srank_lcg(&mut state)).collect();
            let sr = compute_stable_rank(&hashes);
            assert!(
                sr >= 1.0 - 1e-9 && sr <= m as f64 + 1e-9,
                "stable rank {sr} escaped [1, {m}]"
            );
        }
    }

    /// The property that justifies reporting this at all: a mean over pairs
    /// cannot distinguish "four tight clusters" from "evenly spread", because
    /// averaging is exactly the operation that discards the concentration.
    /// Squaring the similarities keeps it.
    #[test]
    fn stable_rank_separates_clustered_from_spread_selections() {
        let mut state = 0xC0FF_EE00_1234_5678_u64;

        // Four distinct bases, each selected four times: sixteen fragments
        // carrying four fragments' worth of information.
        let bases: Vec<u64> = (0..4).map(|_| srank_lcg(&mut state)).collect();
        let clustered: Vec<u64> = bases.iter().flat_map(|&b| [b; 4]).collect();

        // Sixteen independent draws.
        let spread: Vec<u64> = (0..16).map(|_| srank_lcg(&mut state)).collect();

        assert_eq!(clustered.len(), spread.len(), "same selection size, or the comparison is meaningless");

        let sr_clustered = compute_stable_rank(&clustered);
        let sr_spread = compute_stable_rank(&spread);

        // Exact arithmetic for the ideal clustered case: within-cluster pairs
        // contribute 1 each, so E = 2 * 4 * C(4,2) = 48 and the quotient is
        // 16^2 / (16 + 48) = 4. Cross-cluster estimator noise moves it a little.
        assert!(
            (3.0..=4.6).contains(&sr_clustered),
            "four clusters of four should be worth about four, got {sr_clustered}"
        );
        assert!(
            sr_spread > 11.0,
            "sixteen independent fragments should be worth most of sixteen, got {sr_spread}"
        );
        assert!(
            sr_spread > 2.5 * sr_clustered,
            "stable rank must separate these: clustered {sr_clustered}, spread {sr_spread}"
        );
    }

    #[test]
    fn stable_rank_is_reported_for_a_real_selection() {
        let frags = vec![
            make_frag("a", "def alpha(): return 1", 20, "a.py"),
            make_frag("b", "def beta(): return 2", 20, "b.py"),
        ];
        let result = ios_select(
            &frags,
            10_000,
            0.3, 0.25, 0.25, 0.2,
            &empty_feedback(),
            true,
            false,
            &default_factors(),
            DEFAULT_DIV_FLOOR,
            DEFAULT_MIN_CANDIDATE_VALUE,
        );
        let c = &result.curvature;
        assert!(
            c.fingerprinted_count > 0,
            "fixture must produce fingerprinted fragments or the field proves nothing"
        );
        assert!(
            c.stable_rank >= 1.0 && c.stable_rank <= c.fingerprinted_count as f64 + 1e-9,
            "stable_rank {} out of {} is outside [1, m]",
            c.stable_rank,
            c.fingerprinted_count
        );
    }
}

#[cfg(test)]
mod ordering_tests {
    use super::*;
    use crate::fragment::ContextFragment;

    fn frag(id: &str, tokens: u32, semantic: f64) -> ContextFragment {
        let mut f = ContextFragment::new(id.into(), format!("content of {id}"), tokens, id.into());
        f.semantic_score = semantic;
        f.recency_score = 1.0;
        f.entropy_score = 0.5;
        f.has_simhash = true;
        f
    }

    /// Relevance and density must be allowed to disagree, and relevance must win.
    ///
    /// `high` is more relevant but one token larger, so value-per-token ranks
    /// `low` first. That is correct for packing a budget and wrong for display,
    /// and it is exactly the case that made the npm build answer an
    /// authentication question with the billing file.
    #[test]
    fn relevance_wins_over_density_in_returned_order() {
        let fragments = vec![frag("low.py", 10, 0.50), frag("high.py", 11, 0.95)];
        let mut sel = vec![(0usize, Resolution::Full), (1usize, Resolution::Full)];

        // Sanity: density really does prefer the less relevant fragment here.
        let mults = HashMap::new();
        let rel = |i: usize| compute_relevance(&fragments[i], 0.3, 0.25, 0.25, 0.2, 1.0);
        let d_low = rel(0) / fragments[0].token_count as f64;
        let d_high = rel(1) / fragments[1].token_count as f64;
        assert!(rel(1) > rel(0), "high.py must be more relevant");

        order_by_relevance(&mut sel, 0, &fragments, &mults, 0.3, 0.25, 0.25, 0.2);
        assert_eq!(
            sel[0].0,
            1,
            "expected high.py first by relevance (rel {:.4} vs {:.4}); density would \
             have picked low.py ({:.5} vs {:.5})",
            rel(1),
            rel(0),
            d_low,
            d_high
        );
    }

    #[test]
    fn pinned_fragments_keep_their_position() {
        let fragments = vec![frag("pinned.py", 10, 0.10), frag("relevant.py", 10, 0.99)];
        let mut sel = vec![(0usize, Resolution::Full), (1usize, Resolution::Full)];
        let mults = HashMap::new();
        order_by_relevance(&mut sel, 1, &fragments, &mults, 0.3, 0.25, 0.25, 0.2);
        assert_eq!(sel[0].0, 0, "the pinned prefix must not be reordered");
    }

    #[test]
    fn ordering_is_deterministic() {
        // Selection feeds prompt prefixes, which must stay byte-stable.
        let fragments = vec![
            frag("a.py", 10, 0.5),
            frag("b.py", 10, 0.5),
            frag("c.py", 10, 0.5),
        ];
        let mults = HashMap::new();
        let mut first: Option<Vec<usize>> = None;
        for _ in 0..25 {
            let mut sel = vec![
                (0usize, Resolution::Full),
                (1, Resolution::Full),
                (2, Resolution::Full),
            ];
            order_by_relevance(&mut sel, 0, &fragments, &mults, 0.3, 0.25, 0.25, 0.2);
            let order: Vec<usize> = sel.iter().map(|s| s.0).collect();
            match &first {
                None => first = Some(order),
                Some(f) => assert_eq!(f, &order, "tie order must be stable"),
            }
        }
    }
}
