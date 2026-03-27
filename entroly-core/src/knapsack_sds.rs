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
//!    Algorithm: Lazy greedy (Minoux 1978) with diversity penalty.
//!    Guarantee: (1 - 1/e) ≈ 63% of optimal for monotone submodular
//!    functions under cardinality/knapsack constraint.
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
//! References:
//!   - Nemhauser, Wolsey, Fisher (1978) — Submodular maximization
//!   - Sviridenko (2004) — Submodular knapsack approximation
//!   - Minoux (1978) — Lazy greedy acceleration
//!   - Kellerer, Pferschy, Pisinger (2004) — MCKP
//!   - Charikar (2002) — SimHash for similarity estimation

use std::collections::HashMap;
use crate::dedup::hamming_distance;
use crate::fragment::{ContextFragment, compute_relevance};

/// Resolution level for a selected fragment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Resolution {
    /// Full content — maximum information, maximum tokens
    Full,
    /// Skeleton — signatures + structure, ~20% tokens
    Skeleton,
    /// Reference — file path + function name only, ~2% tokens
    Reference,
}

/// Configurable information retention factors for each resolution level.
/// These control the value/cost trade-off in multi-resolution knapsack.
/// Tunable via tuning_config.json → autotune daemon.
pub struct InfoFactors {
    pub skeleton: f64,   // default 0.70
    pub reference: f64,  // default 0.15
}

impl Default for InfoFactors {
    fn default() -> Self {
        InfoFactors { skeleton: 0.70, reference: 0.15 }
    }
}

impl Resolution {
    /// Information retention factor for this resolution level.
    fn info_factor(&self, factors: &InfoFactors) -> f64 {
        match self {
            Resolution::Full => 1.0,
            Resolution::Skeleton => factors.skeleton,
            Resolution::Reference => factors.reference,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Resolution::Full => "full",
            Resolution::Skeleton => "skeleton",
            Resolution::Reference => "reference",
        }
    }
}

/// A candidate item for the SDS+MRK optimizer.
/// Each fragment generates 1-3 candidates (one per resolution).
struct Candidate {
    frag_idx: usize,       // Index into the fragments array
    resolution: Resolution,
    token_cost: u32,       // Tokens for this resolution
    base_value: f64,       // relevance × info_factor (before diversity penalty)
    simhash: u64,          // For diversity computation
}

/// Result of the IOS selection.
#[allow(dead_code)]
pub struct SdsResult {
    /// (fragment_index, chosen_resolution) pairs
    pub selections: Vec<(usize, Resolution)>,
    pub total_tokens: u32,
    pub(crate) total_value: f64,
    pub diversity_score: f64,  // Average pairwise diversity of selected set
}

/// Compute the diversity factor for a candidate given the current selected set.
///
/// diversity = 1 - max_similarity(candidate, selected_set)
///
/// Similarity is estimated from SimHash Hamming distance:
///   sim(a, b) = 1 - hamming(a, b) / 64
///
/// When the selected set is empty, diversity = 1.0 (no penalty).
///
/// Returns a value in [0, 1] where:
///   1.0 = completely novel information
///   0.0 = identical to something already selected
#[inline]
fn diversity_factor(candidate_hash: u64, selected_hashes: &[u64]) -> f64 {
    if selected_hashes.is_empty() {
        return 1.0;
    }

    let max_sim = selected_hashes.iter()
        .map(|&h| {
            let dist = hamming_distance(candidate_hash, h);
            // Similarity: 0 distance = identical = similarity 1.0
            (1.0 - dist as f64 / 64.0).max(0.0)
        })
        .fold(0.0_f64, f64::max);

    // Diversity = 1 - max_similarity
    // But don't penalize below diversity_floor — even similar fragments have SOME new info
    1.0 - max_sim
}

/// Compute average pairwise diversity from SimHash fingerprints.
///
/// diversity = mean over all pairs of (hamming_distance / 64).
/// Returns 1.0 when ≤ 1 hash (trivially diverse).
fn compute_pairwise_diversity(hashes: &[u64]) -> f64 {
    if hashes.len() <= 1 {
        return 1.0;
    }
    let n = hashes.len();
    let mut pair_count = 0usize;
    let mut diversity_sum = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = hamming_distance(hashes[i], hashes[j]);
            diversity_sum += dist as f64 / 64.0;
            pair_count += 1;
        }
    }
    if pair_count > 0 {
        (diversity_sum / pair_count as f64 * 10000.0).round() / 10000.0
    } else {
        1.0
    }
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
            total_value: 0.0,
            diversity_score: 1.0,
        };
    }

    // ── Phase 1: Separate pinned fragments ──
    let mut pinned: Vec<(usize, Resolution)> = Vec::new();
    let mut pinned_tokens: u32 = 0;
    let mut pinned_hashes: Vec<u64> = Vec::new();

    for (i, frag) in fragments.iter().enumerate() {
        if frag.is_pinned {
            pinned.push((i, Resolution::Full));
            pinned_tokens += frag.token_count;
            pinned_hashes.push(frag.simhash);
        }
    }

    let remaining_budget = token_budget.saturating_sub(pinned_tokens);

    // ── Phase 2: Generate candidates ──
    let mut candidates: Vec<Candidate> = Vec::new();

    for (i, frag) in fragments.iter().enumerate() {
        if frag.is_pinned {
            continue;
        }

        let fm = feedback_mults.get(&frag.fragment_id).copied().unwrap_or(1.0);
        let relevance = compute_relevance(frag, w_recency, w_frequency, w_semantic, w_entropy, fm);

        if relevance <= 0.0 || frag.token_count == 0 {
            continue;
        }

        // Full resolution — always available
        candidates.push(Candidate {
            frag_idx: i,
            resolution: Resolution::Full,
            token_cost: frag.token_count,
            base_value: relevance * Resolution::Full.info_factor(info_factors),
            simhash: frag.simhash,
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
                        simhash: frag.simhash,
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
                    simhash: frag.simhash,
                });
            }
        }
    }

    if candidates.is_empty() || remaining_budget == 0 {
        let total_value: f64 = pinned.iter()
            .map(|&(i, _)| {
                let fm = feedback_mults.get(&fragments[i].fragment_id).copied().unwrap_or(1.0);
                compute_relevance(&fragments[i], w_recency, w_frequency, w_semantic, w_entropy, fm)
            })
            .sum();
        return SdsResult {
            selections: pinned,
            total_tokens: pinned_tokens,
            total_value,
            diversity_score: 1.0,
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
            let mut fast_value: f64 = selections.iter()
                .map(|&(i, _)| {
                    let fm = feedback_mults.get(&fragments[i].fragment_id).copied().unwrap_or(1.0);
                    compute_relevance(&fragments[i], w_recency, w_frequency, w_semantic, w_entropy, fm)
                })
                .sum();
            let mut fast_hashes: Vec<u64> = pinned_hashes.clone();

            for c in &candidates {
                if c.resolution == Resolution::Full {
                    selections.push((c.frag_idx, Resolution::Full));
                    fast_tokens += c.token_cost;
                    fast_value += c.base_value;
                    fast_hashes.push(c.simhash);
                }
            }

            return SdsResult {
                selections,
                total_tokens: fast_tokens,
                total_value: (fast_value * 10000.0).round() / 10000.0,
                diversity_score: compute_pairwise_diversity(&fast_hashes),
            };
        }
    }

    // ── Phase 3: Greedy SDS+MRK selection ──
    let mut selected: Vec<(usize, Resolution)> = pinned;
    let mut selected_hashes: Vec<u64> = pinned_hashes;
    let mut selected_frags: Vec<bool> = vec![false; fragments.len()]; // Track which fragment_idx is selected
    let mut budget_used = pinned_tokens;
    let mut total_value: f64 = selected.iter()
        .map(|&(i, _)| {
            let fm = feedback_mults.get(&fragments[i].fragment_id).copied().unwrap_or(1.0);
            compute_relevance(&fragments[i], w_recency, w_frequency, w_semantic, w_entropy, fm)
        })
        .sum();

    // Mark pinned as selected
    for &(idx, _) in &selected {
        selected_frags[idx] = true;
    }

    // Pre-sort candidates by base_value/cost density for faster convergence
    // (The diversity penalty will reorder, but this is a good initial ordering)
    candidates.sort_unstable_by(|a, b| {
        let da = a.base_value / a.token_cost.max(1) as f64;
        let db = b.base_value / b.token_cost.max(1) as f64;
        db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
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
                selected.push((cand.frag_idx, cand.resolution));
                selected_hashes.push(cand.simhash);
                selected_frags[cand.frag_idx] = true;
                budget_used += cand.token_cost;
                total_value += cand.base_value; // Track pre-diversity value for reporting
            }
            None => break, // No more candidates fit
        }
    }

    // ── Phase 4: Compute diversity score of final selection ──
    let diversity_score = compute_pairwise_diversity(&selected_hashes);

    SdsResult {
        selections: selected,
        total_tokens: budget_used,
        total_value: (total_value * 10000.0).round() / 10000.0,
        diversity_score,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fragment::ContextFragment;
    use crate::dedup::simhash;

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
        f.recency_score = 0.9;
        f.entropy_score = 0.7;
        f
    }

    #[test]
    fn test_empty_fragments() {
        let result = ios_select(&[], 1000, 0.3, 0.25, 0.25, 0.2, &empty_feedback(), true, true, &default_factors(), DEFAULT_DIV_FLOOR, DEFAULT_MIN_CANDIDATE_VALUE);
        assert!(result.selections.is_empty());
        assert_eq!(result.total_tokens, 0);
    }

    #[test]
    fn test_single_fragment_selected() {
        let frags = vec![make_frag("a", "def foo(): return 42", 50, "foo.py")];
        let result = ios_select(&frags, 1000, 0.3, 0.25, 0.25, 0.2, &empty_feedback(), true, false, &default_factors(), DEFAULT_DIV_FLOOR, DEFAULT_MIN_CANDIDATE_VALUE);
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

        let result = ios_select(&frags, 600, 0.3, 0.25, 0.25, 0.2, &empty_feedback(), true, false, &default_factors(), DEFAULT_DIV_FLOOR, DEFAULT_MIN_CANDIDATE_VALUE);
        let selected_indices: Vec<usize> = result.selections.iter().map(|s| s.0).collect();
        assert!(selected_indices.contains(&0), "Pinned fragment must be included");
    }

    #[test]
    fn test_diversity_penalizes_duplicates() {
        // Three fragments: two nearly identical, one different
        let frags = vec![
            make_frag("a", "def calculate_tax(income, rate): return income * rate", 100, "tax1.py"),
            make_frag("b", "def calculate_tax(income, rate): return income * rate * 1.0", 100, "tax2.py"),
            make_frag("c", "async fn connect_database(host: str, port: int): pass", 100, "db.py"),
        ];

        // With diversity: should prefer a + c (diverse) over a + b (redundant)
        let result_div = ios_select(&frags, 200, 0.3, 0.25, 0.25, 0.2, &empty_feedback(), true, false, &default_factors(), DEFAULT_DIV_FLOOR, DEFAULT_MIN_CANDIDATE_VALUE);
        let _div_indices: Vec<usize> = result_div.selections.iter().map(|s| s.0).collect();

        // Without diversity: might select a + b (both have high relevance)
        let result_no_div = ios_select(&frags, 200, 0.3, 0.25, 0.25, 0.2, &empty_feedback(), false, false, &default_factors(), DEFAULT_DIV_FLOOR, DEFAULT_MIN_CANDIDATE_VALUE);

        // With diversity enabled, we should have higher diversity score
        assert!(result_div.diversity_score >= result_no_div.diversity_score,
            "Diversity-enabled selection should have higher diversity: {} vs {}",
            result_div.diversity_score, result_no_div.diversity_score
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
        let result_mr = ios_select(&frags, 500, 0.3, 0.25, 0.25, 0.2, &empty_feedback(), true, true, &default_factors(), DEFAULT_DIV_FLOOR, DEFAULT_MIN_CANDIDATE_VALUE);
        let result_no_mr = ios_select(&frags, 500, 0.3, 0.25, 0.25, 0.2, &empty_feedback(), true, false, &default_factors(), DEFAULT_DIV_FLOOR, DEFAULT_MIN_CANDIDATE_VALUE);

        // Multi-resolution should cover more fragments
        let mr_frag_count = result_mr.selections.iter().map(|s| s.0).collect::<std::collections::HashSet<_>>().len();
        let no_mr_frag_count = result_no_mr.selections.iter().map(|s| s.0).collect::<std::collections::HashSet<_>>().len();

        assert!(mr_frag_count >= no_mr_frag_count,
            "Multi-resolution should cover >= fragments: {} vs {}", mr_frag_count, no_mr_frag_count
        );
    }

    #[test]
    fn test_budget_respected() {
        let frags = vec![
            make_frag("a", "content a", 300, "a.py"),
            make_frag("b", "content b", 300, "b.py"),
            make_frag("c", "content c", 300, "c.py"),
        ];

        let result = ios_select(&frags, 500, 0.3, 0.25, 0.25, 0.2, &empty_feedback(), true, false, &default_factors(), DEFAULT_DIV_FLOOR, DEFAULT_MIN_CANDIDATE_VALUE);
        assert!(result.total_tokens <= 500,
            "Budget must be respected: {} > 500", result.total_tokens
        );
    }

    #[test]
    fn test_feedback_multiplier_affects_selection() {
        let frags = vec![
            make_frag("good", "useful code fragment for processing", 200, "good.py"),
            make_frag("bad", "unhelpful boilerplate noise padding", 200, "bad.py"),
        ];

        let mut feedback = HashMap::new();
        feedback.insert("good".to_string(), 1.8);
        feedback.insert("bad".to_string(), 0.3);

        // Budget for only one
        let result = ios_select(&frags, 250, 0.3, 0.25, 0.25, 0.2, &feedback, true, false, &default_factors(), DEFAULT_DIV_FLOOR, DEFAULT_MIN_CANDIDATE_VALUE);
        let selected_indices: Vec<usize> = result.selections.iter().map(|s| s.0).collect();

        assert!(selected_indices.contains(&0), "Feedback-boosted fragment should be preferred");
    }

    #[test]
    fn test_reference_resolution_very_cheap() {
        let mut frags = vec![
            make_frag("a", "def big_function():\n    x = 1\n    y = 2\n    z = 3\n    return x + y + z", 500, "big.py"),
        ];
        frags[0].skeleton_content = Some("def big_function(): ...".into());
        frags[0].skeleton_token_count = Some(50);

        // Budget so small only reference fits
        let result = ios_select(&frags, 15, 0.3, 0.25, 0.25, 0.2, &empty_feedback(), true, true, &default_factors(), DEFAULT_DIV_FLOOR, DEFAULT_MIN_CANDIDATE_VALUE);
        if !result.selections.is_empty() {
            assert_eq!(result.selections[0].1, Resolution::Reference,
                "With tiny budget, reference resolution should be chosen"
            );
        }
    }

    #[test]
    fn test_diversity_score_range() {
        let frags = vec![
            make_frag("a", "machine learning neural network training", 100, "a.py"),
            make_frag("b", "kubernetes docker container deployment", 100, "b.py"),
            make_frag("c", "react component jsx virtual dom rendering", 100, "c.py"),
        ];

        let result = ios_select(&frags, 1000, 0.3, 0.25, 0.25, 0.2, &empty_feedback(), true, false, &default_factors(), DEFAULT_DIV_FLOOR, DEFAULT_MIN_CANDIDATE_VALUE);
        assert!(result.diversity_score >= 0.0 && result.diversity_score <= 1.0,
            "Diversity score must be in [0, 1], got {}", result.diversity_score
        );
    }

    #[test]
    fn test_resolution_preference_by_budget() {
        // When budget is generous, prefer full; when tight, prefer skeleton
        let mut frags = vec![
            make_frag("a", "def foo():\n    return 1 + 2 + 3 + 4 + 5\n", 200, "a.py"),
            make_frag("b", "def bar():\n    return 6 + 7 + 8 + 9 + 10\n", 200, "b.py"),
        ];
        frags[0].skeleton_content = Some("def foo(): ...".into());
        frags[0].skeleton_token_count = Some(30);
        frags[1].skeleton_content = Some("def bar(): ...".into());
        frags[1].skeleton_token_count = Some(30);

        // Generous budget: both full
        let result_big = ios_select(&frags, 1000, 0.3, 0.25, 0.25, 0.2, &empty_feedback(), true, true, &default_factors(), DEFAULT_DIV_FLOOR, DEFAULT_MIN_CANDIDATE_VALUE);
        let full_count_big = result_big.selections.iter().filter(|s| s.1 == Resolution::Full).count();

        // Tight budget: mix of full + skeleton/reference
        let result_tight = ios_select(&frags, 250, 0.3, 0.25, 0.25, 0.2, &empty_feedback(), true, true, &default_factors(), DEFAULT_DIV_FLOOR, DEFAULT_MIN_CANDIDATE_VALUE);
        let full_count_tight = result_tight.selections.iter().filter(|s| s.1 == Resolution::Full).count();

        assert!(full_count_big >= full_count_tight,
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

        let result = ios_select(&frags, 500, 0.3, 0.25, 0.25, 0.2,
            &empty_feedback(), true, false, &default_factors(), DEFAULT_DIV_FLOOR, DEFAULT_MIN_CANDIDATE_VALUE);

        assert_eq!(result.selections.len(), 3, "Fast path should select all 3");
        assert!(result.selections.iter().all(|s| s.1 == Resolution::Full),
            "Fast path should use full resolution for all");
        assert_eq!(result.total_tokens, 150);
    }
}
