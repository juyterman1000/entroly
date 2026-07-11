//! Pure-Rust text compression for the standalone `entroly-rs` binary (and any
//! non-Python embedder). No PyO3 — links into a zero-dependency binary.
//!
//! Algorithm: split text into blocks, score each by information density, and
//! select the subset maximizing total density under a token budget via an
//! **optimal 0/1 knapsack DP** (with a greedy fallback only when the DP table
//! would be pathologically large). Honors the non-annihilation contract:
//! non-empty input + positive budget never yields empty output.

/// Model capability and budget resolution shared with the standalone Rust API.
pub mod model_registry;

use crate::entropy::information_score;

/// Approximate token count (~4 chars/token — the project-wide heuristic).
#[inline]
pub fn est_tokens(s: &str) -> usize {
    (s.chars().count() / 4).max(1)
}

/// Max DP cells (`n * capacity`) before we fall back to greedy, bounding the
/// knapsack's time/memory. 8M u64 ≈ 64 MB worst case — comfortably safe.
const KNAPSACK_CELL_BUDGET: usize = 8_000_000;

/// Cap on how many *other* blocks each block is scored against, bounding the
/// otherwise-O(n²) cross-fragment redundancy term for large inputs.
const SCORE_SAMPLE_CAP: usize = 48;

/// Optimal 0/1 knapsack: pick the subset maximizing total `values` subject to
/// total `weights` ≤ `capacity`. Returns a keep-mask aligned to the inputs.
///
/// Exact DP in `O(n·capacity)` time. Float values are scaled to integers
/// (×1000, info scores live in `[0,1]`) for an exact integer DP. When the table
/// would exceed [`KNAPSACK_CELL_BUDGET`] cells, falls back to greedy
/// value-per-weight (still a valid, bounded selection).
pub fn knapsack_select(values: &[f64], weights: &[usize], capacity: usize) -> Vec<bool> {
    let n = values.len();
    let mut keep = vec![false; n];
    if n == 0 || capacity == 0 {
        return keep;
    }
    let w: Vec<usize> = weights.iter().map(|&x| x.max(1)).collect();

    // Greedy fallback for pathologically large tables.
    if n.saturating_mul(capacity + 1) > KNAPSACK_CELL_BUDGET {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            let da = values[a] / w[a] as f64;
            let db = values[b] / w[b] as f64;
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut used = 0usize;
        for idx in order {
            if used + w[idx] <= capacity {
                keep[idx] = true;
                used += w[idx];
            }
        }
        return keep;
    }

    let vals: Vec<u64> = values
        .iter()
        .map(|&v| (v.max(0.0) * 1000.0).round() as u64)
        .collect();
    let cap = capacity;
    // Memory-efficient 0/1 knapsack: a 1D value row (O(cap)) plus a flat
    // boolean decision grid (O(n·cap) BYTES, not u64) for reconstruction — ~8×
    // less memory than a 2D u64 table, which matters under the worker pool.
    let mut dp = vec![0u64; cap + 1];
    let row = cap + 1;
    let mut took = vec![false; n * row];
    for i in 0..n {
        let (wi, vi) = (w[i], vals[i]);
        // Descending c preserves 0/1 (each item used at most once).
        for c in (wi..=cap).rev() {
            let cand = dp[c - wi] + vi;
            if cand > dp[c] {
                dp[c] = cand;
                took[i * row + c] = true;
            }
        }
    }
    // Backtrack: if item i was taken at the current capacity, include it.
    let mut c = cap;
    for i in (0..n).rev() {
        if took[i * row + c] {
            keep[i] = true;
            c -= w[i];
        }
    }
    keep
}

/// Split text into blocks: paragraphs (blank-line separated) first, falling
/// back to non-empty lines when there is no paragraph structure.
pub fn split_blocks(text: &str) -> Vec<&str> {
    let mut blocks: Vec<&str> = text
        .split("\n\n")
        .filter(|b| !b.trim().is_empty())
        .collect();
    if blocks.len() < 2 {
        blocks = text.lines().filter(|l| !l.trim().is_empty()).collect();
    }
    blocks
}

/// Score each block by information density. The cross-fragment redundancy term
/// is computed against a bounded sample of the other blocks (every k-th, capped
/// at [`SCORE_SAMPLE_CAP`]) so total work is `O(n · SCORE_SAMPLE_CAP)` instead
/// of `O(n²)` on large inputs.
pub fn score_blocks(blocks: &[&str]) -> Vec<f64> {
    let n = blocks.len();
    let step = (n / SCORE_SAMPLE_CAP).max(1);
    blocks
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let others: Vec<&str> = if n <= SCORE_SAMPLE_CAP {
                blocks
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, s)| *s)
                    .collect()
            } else {
                blocks
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i && j % step == 0)
                    .map(|(_, s)| *s)
                    .take(SCORE_SAMPLE_CAP)
                    .collect()
            };
            information_score(b, &others)
        })
        .collect()
}

/// Compress `text` to fit within `budget_tokens`, keeping the optimal subset of
/// information-dense blocks in original order.
pub fn compress_text(text: &str, budget_tokens: usize) -> String {
    if text.trim().is_empty() || budget_tokens == 0 {
        return String::new();
    }
    if est_tokens(text) <= budget_tokens {
        return text.to_string();
    }

    let blocks = split_blocks(text);
    if blocks.is_empty() {
        return String::new();
    }
    let values = score_blocks(&blocks);
    let weights: Vec<usize> = blocks.iter().map(|b| est_tokens(b)).collect();
    let keep = knapsack_select(&values, &weights, budget_tokens);

    let kept: Vec<&str> = blocks
        .iter()
        .zip(keep.iter())
        .filter(|(_, &k)| k)
        .map(|(b, _)| *b)
        .collect();

    // Non-annihilation guard: if nothing fit (every block exceeds the budget),
    // return a budget-bounded head of the densest block rather than nothing.
    if kept.is_empty() {
        let best = values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let max_chars = budget_tokens.saturating_mul(4).max(1);
        return blocks[best].chars().take(max_chars).collect();
    }

    kept.join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knapsack_optimal_simple() {
        // Two cheap high-value items beat one heavy low-value one at cap=4.
        // values: [a=1.0 w=2, b=1.0 w=2, c=0.5 w=4], cap=4 -> keep a,b (value 2.0)
        let keep = knapsack_select(&[1.0, 1.0, 0.5], &[2, 2, 4], 4);
        assert_eq!(keep, vec![true, true, false]);
    }

    #[test]
    fn test_knapsack_respects_capacity() {
        let keep = knapsack_select(&[1.0, 1.0, 1.0], &[3, 3, 3], 5);
        let total: usize = keep
            .iter()
            .zip([3, 3, 3])
            .filter(|(k, _)| **k)
            .map(|(_, w)| w)
            .sum();
        assert!(total <= 5);
        assert_eq!(keep.iter().filter(|&&k| k).count(), 1);
    }

    #[test]
    fn test_knapsack_empty_and_zero_cap() {
        assert_eq!(knapsack_select(&[], &[], 10), Vec::<bool>::new());
        assert_eq!(knapsack_select(&[1.0], &[1], 0), vec![false]);
    }

    #[test]
    fn test_passthrough_when_under_budget() {
        let t = "short text that fits";
        assert_eq!(compress_text(t, 1000), t);
    }

    #[test]
    fn test_empty_and_zero_budget() {
        assert_eq!(compress_text("", 100), "");
        assert_eq!(compress_text("   ", 100), "");
        assert_eq!(compress_text("real content", 0), "");
    }

    #[test]
    fn test_compresses_under_budget() {
        let para = |n: usize| {
            format!(
                "Paragraph {n} with genuine content describing module {n} behavior and edge cases."
            )
        };
        let text = (0..20).map(para).collect::<Vec<_>>().join("\n\n");
        let budget = 40;
        let out = compress_text(&text, budget);
        assert!(!out.is_empty());
        assert!(
            est_tokens(&out) <= budget + 12,
            "respect budget: {} vs {}",
            est_tokens(&out),
            budget
        );
        assert!(out.len() < text.len());
    }

    #[test]
    fn test_non_annihilation_when_single_block_too_big() {
        let big = "x ".repeat(5000);
        let out = compress_text(&big, 10);
        assert!(!out.is_empty());
        assert!(est_tokens(&out) <= 11);
    }

    #[test]
    fn test_scoring_bounded_for_large_input() {
        // 500 blocks: must not hang (O(n·cap) not O(n²·ngrams)); just complete.
        let text = (0..500)
            .map(|i| format!("Block {i} content here."))
            .collect::<Vec<_>>()
            .join("\n\n");
        let budget = 100;
        let out = compress_text(&text, budget);
        assert!(!out.is_empty());
        // Tolerance: est_tokens() floors per block, so packing to `budget`
        // weight-units yields slightly more actual tokens (accumulated rounding).
        assert!(
            est_tokens(&out) <= budget * 3 / 2,
            "bounded near budget: {}",
            est_tokens(&out)
        );
        assert!(out.len() < text.len() / 3, "compressed substantially");
    }
}
