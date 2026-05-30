//! Pure-Rust text compression entry point for the standalone `entroly-rs`
//! binary (and any non-Python embedder).
//!
//! This is the binary-facing counterpart to the Python SDK's `compress()`. It
//! deliberately uses only pure-Rust primitives (no PyO3), so it links into a
//! zero-dependency binary — the foundation of the single-binary distribution.
//!
//! Strategy: split text into blocks, score each by information density
//! ([`crate::entropy::information_score`]), greedily keep the highest
//! density-per-token blocks that fit the token budget, and emit them in
//! original order. Honors the same non-annihilation contract as the Python
//! surface: non-empty input with a positive budget never yields empty output.

use crate::entropy::information_score;

/// Approximate token count (~4 chars/token — the project-wide heuristic).
#[inline]
fn est_tokens(s: &str) -> usize {
    (s.chars().count() / 4).max(1)
}

/// Compress `text` to fit within `budget_tokens`, keeping the most
/// information-dense blocks in original order.
///
/// Blocks are paragraphs (blank-line separated); if the text has no blank-line
/// structure, lines are used. If the whole input already fits, it is returned
/// unchanged. Non-empty input with a positive budget always returns something
/// (falls back to a budget-bounded head of the densest block).
pub fn compress_text(text: &str, budget_tokens: usize) -> String {
    if text.trim().is_empty() || budget_tokens == 0 {
        return String::new();
    }
    if est_tokens(text) <= budget_tokens {
        return text.to_string();
    }

    // Split into blocks: paragraphs first, fall back to lines.
    let mut blocks: Vec<&str> = text.split("\n\n").filter(|b| !b.trim().is_empty()).collect();
    if blocks.len() < 2 {
        blocks = text.lines().filter(|l| !l.trim().is_empty()).collect();
    }
    if blocks.is_empty() {
        return String::new();
    }

    // Score each block by info density relative to the others.
    let scored: Vec<(f64, usize)> = blocks
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let others: Vec<&str> = blocks
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, s)| *s)
                .collect();
            (information_score(b, &others), est_tokens(b))
        })
        .collect();

    // Greedy by value-per-token (knapsack approximation).
    let mut order: Vec<usize> = (0..blocks.len()).collect();
    order.sort_by(|&a, &b| {
        let da = scored[a].0 / scored[a].1 as f64;
        let db = scored[b].0 / scored[b].1 as f64;
        db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = vec![false; blocks.len()];
    let mut used = 0usize;
    for &idx in &order {
        let t = scored[idx].1;
        if used + t <= budget_tokens {
            keep[idx] = true;
            used += t;
        }
    }

    // Non-annihilation guard: if nothing fit (every block exceeds the budget),
    // return a budget-bounded head of the densest block rather than nothing.
    if used == 0 {
        let best = order[0];
        let max_chars = budget_tokens.saturating_mul(4).max(1);
        return blocks[best].chars().take(max_chars).collect();
    }

    blocks
        .iter()
        .enumerate()
        .filter(|(i, _)| keep[*i])
        .map(|(_, b)| *b)
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // Build text that clearly exceeds a small budget.
        let para = |n: usize| format!("Paragraph {n} with some genuine content describing module {n} behavior and edge cases.");
        let text = (0..20).map(para).collect::<Vec<_>>().join("\n\n");
        let budget = 40; // well under the full size
        let out = compress_text(&text, budget);
        assert!(!out.is_empty(), "must not annihilate");
        assert!(est_tokens(&out) <= budget + 12, "should respect budget (+slack): {} vs {}", est_tokens(&out), budget);
        assert!(out.len() < text.len(), "must actually compress");
    }

    #[test]
    fn test_non_annihilation_when_single_block_too_big() {
        // One huge block bigger than the budget → return a bounded head, not "".
        let big = "x ".repeat(5000);
        let out = compress_text(&big, 10);
        assert!(!out.is_empty(), "non-empty input must never yield empty output");
        assert!(est_tokens(&out) <= 11);
    }
}
