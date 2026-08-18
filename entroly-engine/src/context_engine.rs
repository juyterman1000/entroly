//! Stable product-level facade for Entroly context selection.
//!
//! This module intentionally contains no independent ranking or compression
//! algorithm. `ContextEngine` delegates to the shared primitives that already
//! live in `entroly-engine`, so PyO3 and WASM can bind one semantic API without
//! creating another implementation surface.

use crate::fragment::ContextFragment;
use crate::guardrails::TaskType;
use crate::knapsack::{knapsack_optimize, KnapsackResult, ScoringWeights};
use std::collections::HashMap;

/// Stable facade over Entroly's shared-Rust context selection primitives.
#[derive(Debug, Default, Clone, Copy)]
pub struct ContextEngine;

impl ContextEngine {
    /// Construct the stateless context-engine facade.
    pub const fn new() -> Self {
        Self
    }

    /// Select a budget-bounded set of fragments using the canonical Rust
    /// optimizer. No ranking or selection semantics live in this facade.
    pub fn select(
        &self,
        fragments: &[ContextFragment],
        token_budget: u32,
        weights: &ScoringWeights,
        feedback_mults: &HashMap<String, f64>,
        temperature: f64,
    ) -> KnapsackResult {
        knapsack_optimize(
            fragments,
            token_budget,
            weights,
            feedback_mults,
            temperature,
        )
    }

    /// Apply Entroly's existing task-aware budget policy to a base budget.
    ///
    /// The result is saturated at `u32::MAX` and never silently wraps.
    pub fn task_budget(&self, query: &str, base_budget: u32) -> u32 {
        let multiplier = TaskType::classify(query).budget_multiplier();
        let adjusted = (base_budget as f64) * multiplier;
        if !adjusted.is_finite() || adjusted >= u32::MAX as f64 {
            u32::MAX
        } else if adjusted <= 0.0 {
            0
        } else {
            adjusted.round() as u32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn facade_selection_is_exactly_the_canonical_optimizer() {
        let fragments = vec![
            ContextFragment::new("a".into(), "alpha beta gamma".into(), 3, "a.rs".into()),
            ContextFragment::new("b".into(), "delta epsilon zeta".into(), 3, "b.rs".into()),
        ];
        let weights = ScoringWeights::default();
        let feedback = HashMap::new();

        let facade = ContextEngine::new().select(&fragments, 3, &weights, &feedback, 0.01);
        let direct = knapsack_optimize(&fragments, 3, &weights, &feedback, 0.01);

        assert_eq!(facade.selected_indices, direct.selected_indices);
        assert_eq!(facade.total_tokens, direct.total_tokens);
        assert!((facade.total_relevance - direct.total_relevance).abs() < 1e-12);
        assert!((facade.lambda_star - direct.lambda_star).abs() < 1e-12);
        assert!((facade.dual_gap - direct.dual_gap).abs() < 1e-12);
    }

    #[test]
    fn task_budget_delegates_to_existing_task_policy() {
        let engine = ContextEngine::new();
        assert_eq!(engine.task_budget("debug failing test", 1_000), 1_500);
        assert_eq!(engine.task_budget("write documentation", 1_000), 600);
        assert_eq!(engine.task_budget("neutral request", 1_000), 1_000);
    }
}
