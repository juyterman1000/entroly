//! RNR — Recognize and Reject
//!
//! Deterministic token-level recognition scoring with novel-entity penalty.
//! Used as one of the five signals in the EICV phi-fusion pipeline.
//!
//! Algorithm:
//!   1. Tokenize claim & evidence into lowercased word sets
//!   2. Recognition ratio = |claim ∩ evidence| / |claim|
//!   3. Novel entity penalty = 0.1 × count of capitalized claim words absent from evidence
//!   4. score = max(0, recognition − penalty)
//!
//! 100% deterministic, zero external dependencies.

use std::collections::HashSet;

/// Compute RNR score for a claim against evidence.
///
/// Returns a value in `[0.0, 1.0]` where 1.0 = fully recognized (no novel entities).
pub fn rnr_score(claim: &str, evidence: &str) -> f64 {
    let claim_tokens: HashSet<String> =
        claim.split_whitespace().map(|w| w.to_lowercase()).collect();
    let evidence_tokens: HashSet<String> = evidence
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .collect();

    if claim_tokens.is_empty() {
        return 1.0;
    }

    // Recognition ratio: fraction of claim words found in evidence
    let intersection = claim_tokens.intersection(&evidence_tokens).count();
    let recognition = intersection as f64 / claim_tokens.len() as f64;

    // Novel entity penalty: capitalized words in claim not found in evidence
    let claim_entities: HashSet<String> = claim
        .split_whitespace()
        .filter(|w| w.chars().next().is_some_and(|c| c.is_uppercase()))
        .map(|w| w.to_lowercase())
        .collect();

    let novel_count = claim_entities
        .iter()
        .filter(|e| !evidence_tokens.contains(*e))
        .count();

    let penalty = novel_count as f64 * 0.1;
    (recognition - penalty).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fully_grounded_claim() {
        let evidence = "The capital of France is Paris and it is in Europe.";
        let claim = "Paris is the capital of France.";
        let score = rnr_score(claim, evidence);
        assert!(score > 0.7, "score={score}, expected > 0.7");
    }

    #[test]
    fn invented_entities_penalized() {
        let evidence = "The capital of France is Paris.";
        let claim = "Berlin is the capital of Germany and Munich is large.";
        let score = rnr_score(claim, evidence);
        assert!(score < 0.5, "score={score}, expected < 0.5");
    }

    #[test]
    fn empty_claim_returns_one() {
        assert_eq!(rnr_score("", "some evidence"), 1.0);
    }

    #[test]
    fn no_overlap() {
        let score = rnr_score("Alpha Beta Gamma", "delta epsilon zeta");
        assert!(score < 0.1, "score={score}, expected near 0");
    }
}
