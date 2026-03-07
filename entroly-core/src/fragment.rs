/// Context Fragment — the atomic unit of managed context.
///
/// Mirrors Ebbiforge's Episode struct but optimized for context window
/// management rather than memory storage.
///
/// Scoring follows the ContextScorer pattern from ebbiforge-core/src/memory/lsh.rs:
///   composite = w_recency * recency + w_frequency * frequency
///             + w_semantic * semantic + w_entropy * entropy

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// A single piece of context (code snippet, file content, tool result, etc.)
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextFragment {
    #[pyo3(get, set)]
    pub fragment_id: String,
    #[pyo3(get, set)]
    pub content: String,
    #[pyo3(get, set)]
    pub token_count: u32,
    #[pyo3(get, set)]
    pub source: String,

    // Scoring components (all [0.0, 1.0])
    #[pyo3(get, set)]
    pub recency_score: f64,
    #[pyo3(get, set)]
    pub frequency_score: f64,
    #[pyo3(get, set)]
    pub semantic_score: f64,
    #[pyo3(get, set)]
    pub entropy_score: f64,

    // Metadata
    #[pyo3(get, set)]
    pub turn_created: u32,
    #[pyo3(get, set)]
    pub turn_last_accessed: u32,
    #[pyo3(get, set)]
    pub access_count: u32,
    #[pyo3(get, set)]
    pub is_pinned: bool,
    #[pyo3(get, set)]
    pub simhash: u64,

    // Hierarchical fragmentation: optional skeleton variant
    #[pyo3(get, set)]
    #[serde(default)]
    pub skeleton_content: Option<String>,
    #[pyo3(get, set)]
    #[serde(default)]
    pub skeleton_token_count: Option<u32>,
}

#[pymethods]
impl ContextFragment {
    #[new]
    #[pyo3(signature = (fragment_id, content, token_count=0, source="".to_string()))]
    pub fn new(fragment_id: String, content: String, token_count: u32, source: String) -> Self {
        let tc = if token_count == 0 {
            (content.len() / 4).max(1) as u32
        } else {
            token_count
        };
        ContextFragment {
            fragment_id,
            content,
            token_count: tc,
            source,
            recency_score: 1.0,
            frequency_score: 0.0,
            semantic_score: 0.0,
            entropy_score: 0.5,
            turn_created: 0,
            turn_last_accessed: 0,
            access_count: 0,
            is_pinned: false,
            simhash: 0,
            skeleton_content: None,
            skeleton_token_count: None,
        }
    }
}

/// Compute composite relevance score for a fragment.
///
/// Direct port of ebbiforge-core ContextScorer::score() but
/// with entropy replacing emotion as the fourth dimension.
///
/// `feedback_multiplier` comes from FeedbackTracker::learned_value():
///   > 1.0 = historically useful fragment (boosted)
///   < 1.0 = historically unhelpful fragment (suppressed)
///   = 1.0 = no feedback data (neutral)
#[inline]
pub fn compute_relevance(
    frag: &ContextFragment,
    w_recency: f64,
    w_frequency: f64,
    w_semantic: f64,
    w_entropy: f64,
    feedback_multiplier: f64,
) -> f64 {
    let total = w_recency + w_frequency + w_semantic + w_entropy;
    if total == 0.0 {
        return 0.0;
    }

    let base = (w_recency * frag.recency_score
        + w_frequency * frag.frequency_score
        + w_semantic * frag.semantic_score
        + w_entropy * frag.entropy_score)
        / total;

    (base * feedback_multiplier).min(1.0)
}

/// Apply Ebbinghaus forgetting curve decay to all fragments.
///
///   recency(t) = exp(-λ · Δt)
///   where λ = ln(2) / half_life
///
/// Same math as ebbiforge-core HippocampusEngine.
pub fn apply_ebbinghaus_decay(
    fragments: &mut [ContextFragment],
    current_turn: u32,
    half_life: u32,
) {
    let decay_rate = (2.0_f64).ln() / half_life.max(1) as f64;

    for frag in fragments.iter_mut() {
        let dt = current_turn.saturating_sub(frag.turn_last_accessed) as f64;
        frag.recency_score = (-decay_rate * dt).exp();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ebbinghaus_half_life() {
        let mut frags = vec![ContextFragment::new(
            "x".into(), "test".into(), 10, "".into(),
        )];
        frags[0].turn_last_accessed = 0;

        apply_ebbinghaus_decay(&mut frags, 15, 15);

        // At exactly one half-life, recency should be ~0.5
        assert!((frags[0].recency_score - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_relevance_scoring() {
        let mut frag = ContextFragment::new("a".into(), "test".into(), 10, "".into());
        frag.recency_score = 1.0;
        frag.frequency_score = 0.5;
        frag.semantic_score = 0.8;
        frag.entropy_score = 0.9;

        let score = compute_relevance(&frag, 0.30, 0.25, 0.25, 0.20, 1.0);
        assert!(score > 0.0 && score <= 1.0);

        // With positive feedback, score should be boosted
        let boosted = compute_relevance(&frag, 0.30, 0.25, 0.25, 0.20, 1.5);
        assert!(boosted > score);

        // With negative feedback, score should be suppressed
        let suppressed = compute_relevance(&frag, 0.30, 0.25, 0.25, 0.20, 0.6);
        assert!(suppressed < score);
    }
}
