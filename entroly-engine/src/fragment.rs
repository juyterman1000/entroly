//! Context Fragment — the atomic unit of managed context.
//!
//! The atomic unit of managed context, optimized for context window
//! management.
//!
//! Scoring follows the ContextScorer pattern:
//!   composite = w_recency * recency + w_frequency * frequency
//!             + w_semantic * semantic + w_entropy * entropy

#[cfg(feature = "python")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// A single piece of context (code snippet, file content, tool result, etc.)
// `get_all, set_all` replaces 20 per-field `#[pyo3(get, set)]`
// attributes. Inert field attributes are consumed by the `pyclass`
// macro and therefore cannot be `cfg_attr`-gated, so the accessor
// policy has to live on the struct. This additionally exposes
// `eligibility_trace` and `entropy_pre_belief` to Python, which were
// previously Rust-only — additive, and breaks no existing caller.
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextFragment {
    pub fragment_id: String,
    pub content: String,
    pub token_count: u32,
    pub source: String,

    // Scoring components (all [0.0, 1.0])
    pub recency_score: f64,
    pub frequency_score: f64,
    pub semantic_score: f64,
    pub entropy_score: f64,

    // Metadata
    pub turn_created: u32,
    pub turn_last_accessed: u32,
    pub access_count: u32,
    pub is_pinned: bool,
    /// Eviction protection, distinct from `is_pinned`.
    ///
    /// `is_pinned` means "the operator required this in the answer" and forces
    /// inclusion in every selection. `is_protected` means "never drop this from
    /// the store" and does not. Criticality used to set `is_pinned`, conflating
    /// the two, so a manifest or security file was force-included in every
    /// query regardless of relevance. Absent from older serialized indexes,
    /// hence `serde(default)`; see the migration in `migrate_pin_semantics`.
    #[serde(default)]
    pub is_protected: bool,
    pub simhash: u64,
    /// True iff simhash was computed from actual file content.
    /// False for shadow stubs — they are excluded from all similarity ops.
    /// Explicit flag avoids fragile "simhash==0 means no fingerprint" idiom:
    ///   - Future hash algo changes won't accidentally treat 0 as valid
    ///   - Hydration path can set this to true after content is loaded
    #[serde(default)]
    pub has_simhash: bool,

    // Hierarchical fragmentation: optional skeleton variant
    #[serde(default)]
    pub skeleton_content: Option<String>,
    #[serde(default)]
    pub skeleton_token_count: Option<u32>,

    // Hierarchical Context Synthesis: vault belief variant.
    // Beliefs are pre-compiled summaries from the knowledge graph that
    // capture ~50% of a file's information at ~10-15% token cost.
    // Populated by Python during ingest (from vault/beliefs/*.md).
    // Used by IOS as a 4th resolution level between Skeleton and Reference.
    #[serde(default)]
    pub belief_content: Option<String>,
    #[serde(default)]
    pub belief_token_count: Option<u32>,

    // Language identifier for calibrated token estimation.
    // Set during batch_ingest from file extension ("python", "rust", etc.).
    // Used by downstream systems for per-language char/token ratios.
    #[serde(default)]
    pub language: String,

    // RL eligibility trace for TD(λ) temporal credit assignment.
    // Accumulates decaying credit: e_i(t) = λ·e_i(t-1) + ∂log π / ∂θ.
    // Fragments selected in recent requests receive attenuated reward.
    #[serde(default)]
    pub eligibility_trace: f64,

    /// Pristine `entropy_score` captured before the first belief-conditioning
    /// discount. Makes `apply_belief_conditioning` idempotent on a persistent
    /// engine: every pass re-discounts from this baseline instead of compounding,
    /// and a changed belief corpus correctly restores then re-discounts.
    /// `None` until conditioning first touches the fragment.
    #[serde(default)]
    pub entropy_pre_belief: Option<f64>,
}

impl ContextFragment {
    /// The plain Rust constructor. Kept outside `#[pymethods]` so every
    /// consumer can build a fragment — WebAssembly included, where PyO3 cannot
    /// be linked at all. The Python constructor below is a thin wrapper over
    /// this, so the two can never disagree.
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
            is_protected: false,
            simhash: 0,
            has_simhash: false, // must be set explicitly after content processing
            skeleton_content: None,
            skeleton_token_count: None,
            belief_content: None,
            belief_token_count: None,
            language: String::new(),
            eligibility_trace: 0.0,
            entropy_pre_belief: None,
        }
    }
}

/// Python constructor. Delegates straight to `ContextFragment::new` — PyO3's
/// inert `#[new]` / `#[pyo3(..)]` attributes are consumed by the `pymethods`
/// macro, so they cannot themselves be `cfg_attr`-gated; the whole block is
/// gated instead.
#[cfg(feature = "python")]
#[pymethods]
impl ContextFragment {
    #[new]
    #[pyo3(signature = (fragment_id, content, token_count=0, source="".to_string()))]
    fn py_new(fragment_id: String, content: String, token_count: u32, source: String) -> Self {
        Self::new(fragment_id, content, token_count, source)
    }
}

/// Compute composite relevance score for a fragment.
///
/// Composite relevance scorer with entropy as the fourth dimension
/// (replacing emotion from the original ContextScorer design).
///
/// `feedback_multiplier` comes from FeedbackTracker::learned_value():
/// - > 1.0 = historically useful fragment (boosted)
/// - < 1.0 = historically unhelpful fragment (suppressed)
/// - = 1.0 = no feedback data (neutral)
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

    let raw = base * feedback_multiplier;

    // ── Logit Softcap ──────────────────────────────────────────────
    // f(x) = c · tanh(x / c)
    //
    // Compresses [0, ∞) → [0, c) while preserving rank ordering.
    // Properties:
    //   f'(0) = 1   — linear regime for small scores (no distortion)
    //   f(x) → c    — asymptotes at cap (prevents outlier dominance)
    //   Monotone, smooth, C∞-differentiable
    //
    // Prevents a single high-feedback fragment from monopolising the
    // knapsack budget. Cap = 0.85: score=0.5→0.49, score=1.0→0.76,
    // score=2.0→0.84 (near saturation).
    softcap(raw, SOFTCAP)
}

/// Default softcap. 0.85 is calibrated so typical scores (0.3–0.7)
/// remain nearly linear while boosted scores (>1.0) are compressed.
const SOFTCAP: f64 = 0.85;

/// Logit softcap: `c · tanh(x / c)`.
///
/// Gemini-style bounded scoring. When `cap ≤ 0`, falls back to `min(x, 1)`.
#[inline]
pub fn softcap(x: f64, cap: f64) -> f64 {
    if cap <= 0.0 || cap >= 10.0 {
        return x.min(1.0);
    }
    cap * (x / cap).tanh()
}

/// Apply Ebbinghaus forgetting curve decay to all fragments.
///
///   recency(t) = exp(-λ · Δt)
///   where λ = ln(2) / half_life
///
/// Standard Ebbinghaus forgetting curve implementation.
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
            "x".into(),
            "test".into(),
            10,
            "".into(),
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

    #[test]
    fn test_softcap_properties() {
        // Property 1: f(0) = 0
        assert_eq!(softcap(0.0, 0.85), 0.0);

        // Property 2: f'(0) ≈ 1 (linear for small inputs)
        let small = softcap(0.01, 0.85);
        assert!(
            (small - 0.01).abs() < 0.001,
            "Near-zero should be linear: {small}"
        );

        // Property 3: f(x) ≤ cap (tanh saturates to exactly 1.0 for large x)
        assert!(softcap(100.0, 0.85) <= 0.85);

        // Property 4: monotonically increasing
        let a = softcap(0.3, 0.85);
        let b = softcap(0.5, 0.85);
        let c = softcap(0.8, 0.85);
        assert!(a < b && b < c, "Must be monotone: {a} < {b} < {c}");

        // Property 5: compression — boosted score (2.0) is near cap
        let capped = softcap(2.0, 0.85);
        assert!(
            capped > 0.80 && capped < 0.85,
            "2.0 should be near cap: {capped}"
        );

        // Property 6: disabled when cap=0
        assert_eq!(softcap(0.5, 0.0), 0.5);
    }
}
