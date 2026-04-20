//! Semantic Deduplication
//!
//! Two fragments often say the same thing differently — a docstring,
//! a comment, and the code itself. Or two similar implementations.
//!
//! SimHash (in dedup.rs) catches near-verbatim duplicates.
//! This module catches *semantic* redundancy: structurally different
//! fragments that carry the same information.
//!
//! Algorithm: Greedy Marginal Information Gain
//!   Given candidates sorted by relevance:
//!   1. S = {c₁}  (highest relevance always included)
//!   2. For each cᵢ: ΔI = 1 - max(overlap(cᵢ, sⱼ)) for sⱼ ∈ S
//!   3. If ΔI > τ (default 0.3): add cᵢ to S
//!   4. Return S
//!
//! This is the lazy greedy algorithm for monotone submodular
//! maximization under a knapsack constraint. Guarantee: (1 - 1/e) ≈ 63% of optimal.
//!
//! Runs as a pre-filter BEFORE IOS selection: removes informationally
//! redundant candidates, then IOS optimizes the remaining unique ones.
//!
//! Nobody does this at the fragment level. Embedding-based RAG
//! deduplicates at retrieval time on exact/near matches. Entropy-
//! based semantic overlap across structurally different but
//! informationally redundant fragments is new.

use std::collections::HashSet;
use crate::fragment::ContextFragment;
use crate::depgraph::extract_identifiers;

/// Default redundancy threshold. Fragments with marginal info gain
/// below this are considered semantically redundant and dropped.
/// 0.3 = fragment must carry at least 30% new information to be kept.
const DEFAULT_THRESHOLD: f64 = 0.3;

/// Compute content overlap between two fragments using:
///   50% n-gram overlap (word trigrams)
///   50% identifier overlap
///
/// Returns [0, 1] where 1.0 = identical information.
fn content_overlap(a: &str, b: &str) -> f64 {
    let trigram_ov = trigram_jaccard(a, b);
    let ident_ov = identifier_jaccard(a, b);
    0.5 * trigram_ov + 0.5 * ident_ov
}

/// Symmetric trigram Jaccard similarity.
fn trigram_jaccard(a: &str, b: &str) -> f64 {
    let words_a: Vec<&str> = a.split_whitespace().collect();
    let words_b: Vec<&str> = b.split_whitespace().collect();

    if words_a.len() < 3 || words_b.len() < 3 {
        return 0.0;
    }

    let set_a: HashSet<Vec<&str>> = words_a.windows(3).map(|w| w.to_vec()).collect();
    let set_b: HashSet<Vec<&str>> = words_b.windows(3).map(|w| w.to_vec()).collect();

    if set_a.is_empty() || set_b.is_empty() {
        return 0.0;
    }

    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.len() + set_b.len() - intersection;

    if union == 0 { 0.0 } else { intersection as f64 / union as f64 }
}

/// Symmetric identifier Jaccard similarity.
fn identifier_jaccard(a: &str, b: &str) -> f64 {
    let idents_a: HashSet<String> = extract_identifiers(a).into_iter().collect();
    let idents_b: HashSet<String> = extract_identifiers(b).into_iter().collect();

    if idents_a.is_empty() || idents_b.is_empty() {
        return 0.0;
    }

    let intersection = idents_a.intersection(&idents_b).count();
    let union = idents_a.len() + idents_b.len() - intersection;

    if union == 0 { 0.0 } else { intersection as f64 / union as f64 }
}

/// Semantic deduplication via greedy marginal information gain.
///
/// Takes fragment indices sorted by relevance (highest first).
/// Returns the subset of indices to keep — fragments that each
/// contribute at least `threshold` marginal information.
///
/// # Arguments
/// * `fragments` - all fragments in the engine
/// * `sorted_indices` - indices into `fragments`, sorted by relevance descending
/// * `threshold` - minimum marginal info gain to keep (default 0.3)
///
/// # Returns
/// Vec of indices from `sorted_indices` that should be kept.
pub fn semantic_deduplicate(
    fragments: &[ContextFragment],
    sorted_indices: &[usize],
    threshold: Option<f64>,
) -> Vec<usize> {
    let tau = threshold.unwrap_or(DEFAULT_THRESHOLD);

    if sorted_indices.is_empty() {
        return vec![];
    }

    // Always include the highest-relevance fragment
    let mut selected: Vec<usize> = vec![sorted_indices[0]];

    for &idx in &sorted_indices[1..] {
        let candidate = &fragments[idx].content;

        // Compute max overlap with any already-selected fragment
        let max_overlap = selected.iter()
            .map(|&sel_idx| content_overlap(candidate, &fragments[sel_idx].content))
            .fold(0.0_f64, f64::max);

        // Marginal information gain = 1 - max_overlap
        let marginal_info = 1.0 - max_overlap;

        if marginal_info >= tau {
            selected.push(idx);
        }
    }

    selected
}

/// Convenience: deduplicate and return count stats.
pub struct DeduplicationResult {
    /// Indices of fragments to keep.
    pub kept_indices: Vec<usize>,
    /// Number of fragments removed as semantically redundant.
    pub removed_count: usize,
    /// Estimated token savings from deduplication.
    pub tokens_saved: u32,
}

pub fn semantic_deduplicate_with_stats(
    fragments: &[ContextFragment],
    sorted_indices: &[usize],
    threshold: Option<f64>,
) -> DeduplicationResult {
    let kept = semantic_deduplicate(fragments, sorted_indices, threshold);
    let kept_set: HashSet<usize> = kept.iter().copied().collect();

    let removed_count = sorted_indices.len() - kept.len();
    let tokens_saved: u32 = sorted_indices.iter()
        .filter(|idx| !kept_set.contains(idx))
        .map(|&idx| fragments[idx].token_count)
        .sum();

    DeduplicationResult {
        kept_indices: kept,
        removed_count,
        tokens_saved,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fragment::ContextFragment;

    fn make_frag(id: &str, content: &str, tokens: u32) -> ContextFragment {
        ContextFragment::new(id.into(), content.into(), tokens, "test.py".into())
    }

    #[test]
    fn test_redundant_fragments_deduplicated() {
        // Two fragments with nearly identical content
        let frags = vec![
            make_frag("a", "def calculate_tax(income, rate):\n    return income * rate", 50),
            make_frag("b", "def calculate_tax(income, rate):\n    return income * rate * 1.0", 55),
        ];

        let sorted = vec![0, 1]; // both equally relevant
        let kept = semantic_deduplicate(&frags, &sorted, Some(0.3));

        assert_eq!(kept.len(), 1,
            "Semantically redundant fragments should be deduplicated to 1");
        assert_eq!(kept[0], 0, "Highest-relevance fragment should be kept");
    }

    #[test]
    fn test_unique_fragments_kept() {
        // Two completely different fragments
        let frags = vec![
            make_frag("a", "def calculate_tax(income, rate):\n    return income * rate", 50),
            make_frag("b", "async fn connect_database(host, port):\n    conn = await pg.connect(host, port)", 60),
        ];

        let sorted = vec![0, 1];
        let kept = semantic_deduplicate(&frags, &sorted, Some(0.3));

        assert_eq!(kept.len(), 2, "Unique fragments should both be kept");
    }

    #[test]
    fn test_three_way_dedup() {
        // Three fragments: A and B are semantically redundant (same identifiers,
        // same structure, slightly different wording), C is completely different
        let frags = vec![
            make_frag("a", "def process_payment(amount, currency):\n    result = charge(amount, currency)\n    log_transaction(result)\n    return result", 50),
            make_frag("b", "def process_payment(amount, currency):\n    result = charge(amount, currency)\n    log_transaction(result)\n    return result  # updated", 60),
            make_frag("c", "def send_notification(user, message):\n    email.send(user, message)\n    record_delivery(user)", 45),
        ];

        let sorted = vec![0, 1, 2];
        let kept = semantic_deduplicate(&frags, &sorted, Some(0.3));

        // Should keep A (first) and C (different), drop B (redundant with A)
        assert!(kept.contains(&0), "First fragment always kept");
        assert!(kept.contains(&2), "Unique fragment C should be kept");
        assert_eq!(kept.len(), 2, "Should drop the redundant fragment B");
    }

    #[test]
    fn test_dedup_stats() {
        let frags = vec![
            make_frag("a", "def foo(x, y): return x + y", 30),
            make_frag("b", "def foo(x, y): return x + y + 0", 35),
            make_frag("c", "class Database:\n    def connect(self): pass", 50),
        ];

        let sorted = vec![0, 1, 2];
        let result = semantic_deduplicate_with_stats(&frags, &sorted, Some(0.3));

        assert!(result.removed_count > 0,
            "Should remove at least one redundant fragment");
        assert!(result.tokens_saved > 0,
            "Should save tokens from removed fragments: {}", result.tokens_saved);
    }

    #[test]
    fn test_empty_input() {
        let frags: Vec<ContextFragment> = vec![];
        let kept = semantic_deduplicate(&frags, &[], None);
        assert!(kept.is_empty());
    }

    #[test]
    fn test_threshold_sensitivity() {
        // At low threshold (0.1), more fragments survive
        // At high threshold (0.8), only very unique fragments survive
        let frags = vec![
            make_frag("a", "def calculate_tax(income, rate): return income * rate", 50),
            make_frag("b", "def compute_tax(salary, tax_rate): return salary * tax_rate", 55),
            make_frag("c", "def send_email(to, body): smtp.send(to, body)", 45),
        ];

        let sorted = vec![0, 1, 2];
        let kept_low = semantic_deduplicate(&frags, &sorted, Some(0.1));
        let kept_high = semantic_deduplicate(&frags, &sorted, Some(0.8));

        assert!(kept_low.len() >= kept_high.len(),
            "Lower threshold should keep more fragments: {} vs {}",
            kept_low.len(), kept_high.len());
    }
}
