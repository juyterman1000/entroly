//! Context Resonance — Pairwise Fragment Interaction Scoring
//!
//! # The Insight
//!
//! Traditional context selection scores fragments independently:
//!   score(A) + score(B) + score(C) → select top-k
//!
//! But certain *combinations* of fragments produce dramatically better LLM
//! outputs than the sum of their individual scores. Two fragments scoring
//! 0.4 alone might score 0.95 together because they create a complete
//! causal picture the LLM can reason over.
//!
//! # The Algorithm: Resonance Matrix with Decay
//!
//! We maintain a sparse pairwise resonance matrix R where R[i][j] tracks
//! the learned interaction strength between fragments i and j.
//!
//! ## Learning Signal
//!
//! On `record_success(fragment_ids)`:
//!   For every co-selected pair (i, j) in the success set:
//!     R[i][j] += η · (reward − baseline)
//!
//! On `record_failure(fragment_ids)`:
//!   For every co-selected pair (i, j) in the failure set:
//!     R[i][j] += η · (reward − baseline)   [reward < 0 → suppresses pair]
//!
//! ## Scoring Integration
//!
//! The resonance bonus for a candidate fragment `c` given already-selected set S:
//!
//!   resonance_bonus(c | S) = Σ_{s ∈ S} R[c][s] / |S|
//!
//! This is a **supermodular** bonus: adding fragments with high pairwise
//! resonance is more valuable when their partners are already selected.
//!
//! ## Complexity Management
//!
//! - Sparse storage: only pairs that have been co-selected are tracked
//! - Decay: R[i][j] *= decay_rate each turn (prevents stale resonances)
//! - Capacity: max 10,000 pairs (LRU eviction by last-update turn)
//! - The resonance bonus feeds into PRISM's 5th dimension, where spectral
//!   shaping automatically handles its higher variance
//!
//! # Mathematical Properties
//!
//! The resonance matrix is symmetric: R[i][j] = R[j][i].
//! Learning rate η = 0.1 is calibrated so ~10 co-selections bring R from
//! 0 to the typical feedback_multiplier range [0.5, 2.0].
//!
//! The per-turn decay (0.98) gives a half-life of ~34 turns, matching
//! the Ebbinghaus decay but slower (pairwise patterns are more stable
//! than individual recency).

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// A symmetric pair key — order-independent.
/// Always stores (min, max) to ensure R[a][b] == R[b][a].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct PairKey(String, String);

impl PairKey {
    fn new(a: &str, b: &str) -> Self {
        if a <= b {
            PairKey(a.to_string(), b.to_string())
        } else {
            PairKey(b.to_string(), a.to_string())
        }
    }

    /// Convert to a string key for JSON serialization.
    fn to_key_string(&self) -> String {
        format!("{}::{}", self.0, self.1)
    }

    /// Parse from a string key.
    fn from_key_string(s: &str) -> Option<Self> {
        let (a, b) = s.split_once("::")?;
        Some(PairKey::new(a, b))
    }
}

/// Custom serde for HashMap<PairKey, V> — JSON requires string keys.
mod pair_key_map_serde {
    use super::*;
    use serde::ser::SerializeMap;
    use serde::de::{self, MapAccess, Visitor};
    use std::fmt;

    pub fn serialize<S>(map: &HashMap<PairKey, super::ResonanceEntry>, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
        let mut ser_map = serializer.serialize_map(Some(map.len()))?;
        for (k, v) in map {
            ser_map.serialize_entry(&k.to_key_string(), v)?;
        }
        ser_map.end()
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<HashMap<PairKey, super::ResonanceEntry>, D::Error>
    where D: serde::Deserializer<'de> {
        struct PairKeyMapVisitor;
        impl<'de> Visitor<'de> for PairKeyMapVisitor {
            type Value = HashMap<PairKey, super::ResonanceEntry>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a map with 'a::b' string keys")
            }
            fn visit_map<M: MapAccess<'de>>(self, mut access: M) -> Result<Self::Value, M::Error> {
                let mut map = HashMap::with_capacity(access.size_hint().unwrap_or(0));
                while let Some((key_str, value)) = access.next_entry::<String, super::ResonanceEntry>()? {
                    let key = PairKey::from_key_string(&key_str)
                        .ok_or_else(|| de::Error::custom(format!("invalid pair key: {key_str}")))?;
                    map.insert(key, value);
                }
                Ok(map)
            }
        }
        deserializer.deserialize_map(PairKeyMapVisitor)
    }
}

/// A single resonance entry tracking pairwise interaction strength.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct ResonanceEntry {
    /// Learned interaction strength (can be negative for anti-resonance).
    strength: f64,
    /// Turn when this entry was last updated (for LRU eviction).
    last_update_turn: u32,
    /// Number of co-selections observed (for confidence weighting).
    co_selections: u32,
}

/// The Resonance Matrix — learns which fragment pairs produce synergistic
/// LLM outputs through outcome tracking.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResonanceMatrix {
    #[serde(with = "pair_key_map_serde")]
    pairs: HashMap<PairKey, ResonanceEntry>,
    /// Learning rate for resonance updates.
    eta: f64,
    /// Per-turn decay factor (applied in `decay_tick`).
    decay_rate: f64,
    /// Maximum tracked pairs (LRU eviction when exceeded).
    max_pairs: usize,
}

impl Default for ResonanceMatrix {
    fn default() -> Self {
        Self::new()
    }
}

impl ResonanceMatrix {
    pub fn new() -> Self {
        ResonanceMatrix {
            pairs: HashMap::new(),
            eta: 0.1,
            decay_rate: 0.98,
            max_pairs: 10_000,
        }
    }

    /// Record outcome for a set of co-selected fragments.
    ///
    /// For N selected fragments, updates C(N,2) = N*(N-1)/2 pairs.
    /// With typical N ≈ 10-20, this is 45-190 updates — negligible.
    pub fn record_outcome(&mut self, fragment_ids: &[String], reward: f64, current_turn: u32) {
        let n = fragment_ids.len();
        if n < 2 { return; }

        // Update all co-selected pairs
        for i in 0..n {
            for j in (i + 1)..n {
                let key = PairKey::new(&fragment_ids[i], &fragment_ids[j]);
                let entry = self.pairs.entry(key).or_insert(ResonanceEntry {
                    strength: 0.0,
                    last_update_turn: current_turn,
                    co_selections: 0,
                });
                entry.strength += self.eta * reward;
                entry.last_update_turn = current_turn;
                entry.co_selections += 1;
            }
        }

        // LRU eviction if over capacity
        if self.pairs.len() > self.max_pairs {
            // Find the pair with the oldest last_update_turn
            let mut oldest_turn = u32::MAX;
            let mut oldest_key = None;
            for (key, entry) in &self.pairs {
                if entry.last_update_turn < oldest_turn {
                    oldest_turn = entry.last_update_turn;
                    oldest_key = Some(key.clone());
                }
            }
            if let Some(key) = oldest_key {
                self.pairs.remove(&key);
            }
        }
    }

    /// Compute the resonance bonus for a candidate fragment given the
    /// already-selected set.
    ///
    /// Returns the mean pairwise resonance strength, weighted by confidence:
    ///   bonus = Σ_{s ∈ selected} w(s) · R[candidate][s] / Σ w(s)
    ///
    /// where w(s) = 1 − 1/(1 + co_selections) is the Wilson-inspired
    /// confidence weight (ranges from 0 → 1 as observations grow).
    pub fn resonance_bonus(&self, candidate_id: &str, selected_ids: &[&str]) -> f64 {
        if selected_ids.is_empty() { return 0.0; }

        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;

        for &sel_id in selected_ids {
            let key = PairKey::new(candidate_id, sel_id);
            if let Some(entry) = self.pairs.get(&key) {
                // Confidence weight: converges to 1.0 as co_selections → ∞
                // At 1 observation: w = 0.5, at 5: w = 0.83, at 10: w = 0.91
                let w = 1.0 - 1.0 / (1.0 + entry.co_selections as f64);
                weighted_sum += w * entry.strength;
                weight_total += w;
            }
        }

        if weight_total < 1e-10 { return 0.0; }
        weighted_sum / weight_total
    }

    /// Compute resonance bonuses for all candidate fragments at once.
    ///
    /// Returns a map from fragment_id → resonance_bonus.
    /// Candidates not in the resonance matrix get 0.0 (cold start).
    pub fn batch_resonance_bonuses(
        &self,
        candidate_ids: &[&str],
        selected_ids: &[&str],
    ) -> HashMap<String, f64> {
        candidate_ids.iter()
            .map(|&cid| (cid.to_string(), self.resonance_bonus(cid, selected_ids)))
            .collect()
    }

    /// Apply per-turn decay to all resonance entries.
    /// Removes entries that have decayed below threshold (|strength| < 0.001).
    pub fn decay_tick(&mut self) {
        self.pairs.retain(|_, entry| {
            entry.strength *= self.decay_rate;
            entry.strength.abs() > 0.001
        });
    }

    /// Total number of tracked pairs.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Whether the matrix is empty (no learned resonances yet).
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// Get the strongest resonance pairs (for diagnostics).
    /// Returns up to `top_k` pairs sorted by |strength| descending.
    pub fn top_pairs(&self, top_k: usize) -> Vec<(String, String, f64, u32)> {
        let mut pairs: Vec<_> = self.pairs.iter()
            .map(|(k, v)| (k.0.clone(), k.1.clone(), v.strength, v.co_selections))
            .collect();
        pairs.sort_by(|a, b| b.2.abs().total_cmp(&a.2.abs()));
        pairs.truncate(top_k);
        pairs
    }

    /// Mean absolute resonance strength (health metric).
    /// Low (< 0.01) = cold start, no patterns learned yet.
    /// High (> 0.5) = strong interaction patterns detected.
    pub fn mean_strength(&self) -> f64 {
        if self.pairs.is_empty() { return 0.0; }
        let sum: f64 = self.pairs.values().map(|e| e.strength.abs()).sum();
        sum / self.pairs.len() as f64
    }
}


// ═══════════════════════════════════════════════════════════════════
//  FRAGMENT CONSOLIDATION — Maxwell's Demon for Information Entropy
// ═══════════════════════════════════════════════════════════════════

/// Fragment Consolidation Result
#[derive(Clone, Debug)]
pub struct ConsolidationResult {
    /// Fragment IDs that were consolidated (merged into the winner).
    pub consolidated_ids: Vec<String>,
    /// The winning fragment ID (highest feedback score).
    pub winner_id: String,
    /// Number of tokens saved by the consolidation.
    pub tokens_saved: u32,
}

/// Identifies groups of near-duplicate fragments that can be consolidated.
///
/// # Algorithm: SimHash Clustering + Outcome Selection
///
/// 1. Group fragments by SimHash similarity (Hamming distance ≤ threshold)
/// 2. Within each group, pick the "winner" — the fragment with the highest
///    feedback multiplier (proven useful by RL outcomes)
/// 3. The losers are evicted, their access counts transferred to the winner
///
/// # Why this works
///
/// Over time, a codebase accumulates multiple versions of the same function
/// (pre-refactor, post-refactor, test fixture, etc.). The existing dedup
/// catches exact matches (Hamming ≤ 3), but near-duplicates (Hamming 4-8)
/// slip through. These waste context budget and dilute the knapsack signal.
///
/// By consolidating based on *outcome* (which version led to better LLM
/// outputs), we're doing Maxwell's Demon: reducing entropy by keeping the
/// thermodynamically "useful" variant.
///
/// # Complexity: O(N²) pairwise comparison
///
/// Acceptable because consolidation runs once per `advance_turn()`, not
/// per `optimize()`. With N ≤ 10,000 fragments and early-exit on
/// already-pinned fragments, practical runtime is sub-millisecond.
pub fn find_consolidation_groups(
    fragments: &[(String, u64, f64, bool, u32)], // (id, simhash, feedback_mult, is_pinned, token_count)
    hamming_threshold: u32,
) -> Vec<ConsolidationResult> {
    let n = fragments.len();
    if n < 2 { return Vec::new(); }

    let mut used = vec![false; n];
    let mut results = Vec::new();

    for i in 0..n {
        if used[i] || fragments[i].3 { continue; } // skip if used or pinned

        let mut group = vec![i];
        for j in (i + 1)..n {
            if used[j] || fragments[j].3 { continue; }

            let dist = (fragments[i].1 ^ fragments[j].1).count_ones();
            if dist <= hamming_threshold {
                group.push(j);
            }
        }

        if group.len() < 2 { continue; }

        // Pick winner: highest feedback_mult, breaking ties by lower token_count
        // (prefer the more concise version at equal quality)
        let winner_idx = *group.iter()
            .max_by(|&&a, &&b| {
                fragments[a].2.total_cmp(&fragments[b].2)
                    .then(fragments[b].4.cmp(&fragments[a].4)) // lower tokens wins ties
            })
            .unwrap();

        let tokens_saved: u32 = group.iter()
            .filter(|&&idx| idx != winner_idx)
            .map(|&idx| fragments[idx].4)
            .sum();

        let consolidated_ids: Vec<String> = group.iter()
            .filter(|&&idx| idx != winner_idx)
            .map(|&idx| fragments[idx].0.clone())
            .collect();

        for &idx in &group {
            used[idx] = true;
        }

        results.push(ConsolidationResult {
            consolidated_ids,
            winner_id: fragments[winner_idx].0.clone(),
            tokens_saved,
        });
    }

    results
}


// ═══════════════════════════════════════════════════════════════════
//  COVERAGE SUFFICIENCY ESTIMATOR — The Unknown Unknowns Engine
// ═══════════════════════════════════════════════════════════════════

/// Coverage estimation result.
#[derive(Clone, Debug)]
pub struct CoverageEstimate {
    /// Fraction of the estimated relevant information space that our
    /// selected context covers. Range: [0, 1].
    ///
    /// 0.85 means "we estimate we have 85% of what the LLM needs."
    pub coverage: f64,

    /// Estimated number of relevant fragments we're missing.
    pub estimated_gap: f64,

    /// Confidence in the coverage estimate itself. Range: [0, 1].
    /// Low when we have few observations (cold start).
    pub confidence: f64,

    /// Human-readable risk level.
    pub risk_level: &'static str,
}

/// Estimates context coverage using a capture-recapture inspired method.
///
/// # The Lincoln-Petersen Estimator (adapted)
///
/// Classical capture-recapture: to estimate the size of a population,
/// capture N₁ individuals, mark them, release. Capture N₂ individuals,
/// count how many (m) are marked. Estimated population:
///
///   N̂ = N₁ · N₂ / m
///
/// # Adaptation for Context
///
/// We have two independent "captures" of the relevant fragment space:
///
/// 1. **Semantic capture (N₁)**: fragments retrieved by SimHash/LSH
///    similarity to the query (content-based relevance)
///
/// 2. **Structural capture (N₂)**: fragments retrieved by dependency
///    graph traversal from the initially selected set (structural relevance)
///
/// 3. **Overlap (m)**: fragments found by BOTH methods
///
/// If both methods independently sample from the true relevant set,
/// the Chapman estimator (bias-corrected Lincoln-Petersen) gives:
///
///   N̂ = (N₁ + 1)(N₂ + 1) / (m + 1) − 1
///
/// Coverage = |selected| / N̂
///
/// # Why This Works
///
/// - High overlap (m ≈ N₁ ≈ N₂): both methods find the same fragments
///   → small estimated population → high coverage. We're probably not
///   missing much.
///
/// - Low overlap (m << min(N₁, N₂)): methods find different fragments
///   → large estimated population → low coverage. There's likely a
///   whole category of relevant fragments neither method found.
///
/// - Zero overlap: degenerate case, falls back to union size estimate.
///
/// # Confidence Estimation
///
/// The coefficient of variation of the Chapman estimator is approximately:
///   CV = sqrt((N₁ + 1)(N₂ + 1)(N₁ − m)(N₂ − m) / ((m + 1)²(m + 2)))
///
/// We convert this to a [0, 1] confidence: conf = 1 / (1 + CV).
/// Higher m (more overlap) → lower CV → higher confidence.
pub fn estimate_coverage(
    selected_count: usize,
    semantic_candidates: usize,   // N₁: fragments found by SimHash similarity
    structural_candidates: usize, // N₂: fragments found by dep graph traversal
    overlap: usize,               // m: fragments found by BOTH methods
) -> CoverageEstimate {
    // Edge cases
    if selected_count == 0 {
        return CoverageEstimate {
            coverage: 0.0,
            estimated_gap: 0.0,
            confidence: 0.0,
            risk_level: "unknown",
        };
    }

    let n1 = semantic_candidates.max(1) as f64;
    let n2 = structural_candidates.max(1) as f64;
    let m = overlap as f64;

    // Chapman estimator (bias-corrected Lincoln-Petersen)
    let estimated_total = if m > 0.0 {
        (n1 + 1.0) * (n2 + 1.0) / (m + 1.0) - 1.0
    } else {
        // No overlap: use union as lower bound (conservative)
        n1 + n2
    };

    // Coverage: what fraction of the estimated relevant space do we have?
    let coverage = if estimated_total > 0.0 {
        (selected_count as f64 / estimated_total).clamp(0.0, 1.0)
    } else {
        1.0
    };

    // Confidence via coefficient of variation of Chapman estimator
    let confidence = if m > 0.0 {
        let cv_sq = (n1 + 1.0) * (n2 + 1.0) * (n1 - m).max(0.0) * (n2 - m).max(0.0)
            / ((m + 1.0).powi(2) * (m + 2.0));
        let cv = cv_sq.max(0.0).sqrt();
        (1.0 / (1.0 + cv)).clamp(0.0, 1.0)
    } else {
        // No overlap → very low confidence in the estimate
        0.2
    };

    // Estimated gap (how many relevant fragments are we missing?)
    let estimated_gap = (estimated_total - selected_count as f64).max(0.0);

    // Risk level based on coverage × confidence
    let effective = coverage * confidence;
    let risk_level = if effective >= 0.75 {
        "low"
    } else if effective >= 0.45 {
        "medium"
    } else {
        "high"
    };

    CoverageEstimate {
        coverage,
        estimated_gap,
        confidence,
        risk_level,
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    // ── Resonance Matrix Tests ──

    #[test]
    fn test_resonance_symmetric() {
        let mut rm = ResonanceMatrix::new();
        let ids = vec!["a".to_string(), "b".to_string()];
        rm.record_outcome(&ids, 1.0, 1);

        // R[a][b] == R[b][a] due to symmetric PairKey
        let bonus_ab = rm.resonance_bonus("a", &["b"]);
        let bonus_ba = rm.resonance_bonus("b", &["a"]);
        assert!((bonus_ab - bonus_ba).abs() < 1e-10,
            "Resonance must be symmetric: {bonus_ab} vs {bonus_ba}");
    }

    #[test]
    fn test_resonance_positive_reinforcement() {
        let mut rm = ResonanceMatrix::new();
        let ids = vec!["a".to_string(), "b".to_string()];

        // 10 positive outcomes
        for t in 0..10 {
            rm.record_outcome(&ids, 1.0, t);
        }

        let bonus = rm.resonance_bonus("a", &["b"]);
        assert!(bonus > 0.5, "10 positive outcomes should create strong resonance: {bonus}");
    }

    #[test]
    fn test_resonance_negative_suppression() {
        let mut rm = ResonanceMatrix::new();
        let ids = vec!["a".to_string(), "b".to_string()];

        // 10 negative outcomes
        for t in 0..10 {
            rm.record_outcome(&ids, -1.0, t);
        }

        let bonus = rm.resonance_bonus("a", &["b"]);
        assert!(bonus < -0.5, "10 negative outcomes should create anti-resonance: {bonus}");
    }

    #[test]
    fn test_resonance_cold_start_zero() {
        let rm = ResonanceMatrix::new();
        let bonus = rm.resonance_bonus("x", &["y", "z"]);
        assert!((bonus).abs() < 1e-10, "Unknown pairs should have zero resonance");
    }

    #[test]
    fn test_resonance_decay() {
        let mut rm = ResonanceMatrix::new();
        let ids = vec!["a".to_string(), "b".to_string()];
        rm.record_outcome(&ids, 1.0, 0);

        let before = rm.resonance_bonus("a", &["b"]);

        // Apply 50 turns of decay (0.98^50 ≈ 0.36)
        for _ in 0..50 {
            rm.decay_tick();
        }

        let after = rm.resonance_bonus("a", &["b"]);
        assert!(after < before * 0.5,
            "Decay should reduce resonance: before={before}, after={after}");
    }

    #[test]
    fn test_resonance_multiple_partners() {
        let mut rm = ResonanceMatrix::new();

        // Fragment "a" resonates strongly with "b" but not with "c"
        for t in 0..10 {
            rm.record_outcome(&["a".to_string(), "b".to_string()], 1.0, t);
            rm.record_outcome(&["a".to_string(), "c".to_string()], -0.5, t);
        }

        let bonus_with_b = rm.resonance_bonus("a", &["b"]);
        let bonus_with_c = rm.resonance_bonus("a", &["c"]);
        let bonus_with_both = rm.resonance_bonus("a", &["b", "c"]);

        assert!(bonus_with_b > bonus_with_c,
            "Should resonate more with b than c: {bonus_with_b} vs {bonus_with_c}");
        // Mixed resonance: b pulls up, c pulls down
        assert!(bonus_with_both < bonus_with_b && bonus_with_both > bonus_with_c,
            "Mixed resonance should be between pure b and pure c: {bonus_with_both}");
    }

    // ── Consolidation Tests ──

    #[test]
    fn test_consolidation_near_duplicates() {
        // Two fragments with Hamming distance 5 (near-duplicate)
        let frags = vec![
            ("f1".to_string(), 0b1111_0000u64, 1.5, false, 100),
            ("f2".to_string(), 0b1110_1000u64, 0.8, false, 120), // Hamming 2 from f1
            ("f3".to_string(), 0xFF00FF00u64, 1.0, false, 80),   // different
        ];

        let groups = find_consolidation_groups(&frags, 6);
        assert_eq!(groups.len(), 1, "Should find one consolidation group");
        assert_eq!(groups[0].winner_id, "f1", "Higher feedback_mult should win");
        assert_eq!(groups[0].consolidated_ids, vec!["f2"]);
        assert_eq!(groups[0].tokens_saved, 120);
    }

    #[test]
    fn test_consolidation_pinned_exempt() {
        let frags = vec![
            ("f1".to_string(), 0b1111_0000u64, 1.5, true, 100), // pinned
            ("f2".to_string(), 0b1110_1000u64, 0.8, false, 120),
        ];

        let groups = find_consolidation_groups(&frags, 6);
        assert!(groups.is_empty(), "Pinned fragments should not be consolidated");
    }

    // ── Coverage Estimator Tests ──

    #[test]
    fn test_coverage_high_overlap() {
        // Both methods find ~same fragments → high coverage
        // When overlap is high relative to both N₁ and N₂, Chapman estimates
        // a small population, giving high coverage and confidence.
        let est = estimate_coverage(
            20,  // selected (most of the estimated population)
            20,  // semantic candidates
            20,  // structural candidates
            19,  // overlap (19 of 20 found by both methods)
        );
        assert!(est.coverage > 0.7, "High overlap should give high coverage: {}", est.coverage);
        assert!(est.confidence > 0.6, "High overlap should give good confidence: {}", est.confidence);
        assert_eq!(est.risk_level, "low");
    }

    #[test]
    fn test_coverage_low_overlap() {
        // Methods find different fragments → low coverage (unknown unknowns)
        let est = estimate_coverage(
            10,  // selected
            20,  // semantic candidates
            15,  // structural candidates
            2,   // overlap (very low!)
        );
        assert!(est.coverage < 0.5, "Low overlap should give low coverage: {}", est.coverage);
        assert_eq!(est.risk_level, "high");
    }

    #[test]
    fn test_coverage_zero_selected() {
        let est = estimate_coverage(0, 10, 5, 3);
        assert!((est.coverage).abs() < 1e-10);
        assert_eq!(est.risk_level, "unknown");
    }

    #[test]
    fn test_coverage_monotone_in_overlap() {
        // More overlap → higher coverage estimate
        let est_low = estimate_coverage(10, 20, 15, 2);
        let est_high = estimate_coverage(10, 20, 15, 10);
        assert!(est_high.coverage > est_low.coverage,
            "More overlap should increase coverage: {} vs {}", est_high.coverage, est_low.coverage);
    }

    #[test]
    fn test_chapman_estimator_accuracy() {
        // Known population test: 100 fragments in true relevant set
        // Semantic finds 40, structural finds 30, overlap = 12
        // Chapman estimate: (41)(31)/(13) - 1 = 96.77 ≈ 100 ✓
        let est = estimate_coverage(
            25,  // we selected 25
            40,  // semantic found 40
            30,  // structural found 30
            12,  // overlap 12
        );
        // True population ≈ 97, coverage ≈ 25/97 ≈ 0.26
        assert!((est.coverage - 0.26).abs() < 0.1,
            "Chapman should estimate ~26% coverage, got {:.2}", est.coverage);
        assert!(est.estimated_gap > 60.0,
            "Should estimate ~72 missing fragments, got {:.0}", est.estimated_gap);
    }
}
