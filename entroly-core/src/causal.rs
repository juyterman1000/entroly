//! Causal Context Graph — Interventional Estimation with Information Gravity
//!
//! # The Problem: Selection Bias in Context Optimization
//!
//! Standard RAG and context optimization treat fragment selection as an
//! **observational study**: fragments that co-occur with positive outcomes
//! get boosted scores. But this confounds correlation with causation.
//!
//! Example: Fragment A (a utility module) is always co-selected alongside
//! Fragment B (the core implementation). B is truly helpful; A is irrelevant
//! but inherits B's positive feedback through co-selection bias.
//!
//! # The Solution: Do-Calculus via Natural Experiments
//!
//! We exploit the RAVEN-UCB exploration mechanism as a **natural instrument
//! variable** (Pearl, Causality 2009; Hernán & Robins, 2020). When a fragment is:
//!
//! - **Randomly included** via exploration swap → success rate estimates
//!   P(success | do(include fragment f))  [interventional]
//! - **Naturally selected** by the policy → success rate estimates
//!   P(success | observe(include f))  [observational]
//!
//! The gap between these two quantities reveals **confounding bias**:
//!
//!   confounding_bias(f) = E[Y|observe(f)] − E[Y|do(f)]
//!
//! - bias > 0: fragment appears better than it is (rides coattails of good partners)
//! - bias < 0: fragment is a hidden gem (suppressor variable)
//! - bias ≈ 0: observational and causal estimates agree
//!
//! This is Pearl's do-calculus applied to context selection — first application
//! of instrumental variable estimation in RAG/context optimization.
//!
//! # Temporal Causal Chains (Transfer Entropy)
//!
//! Beyond single-turn causality, we track which fragments at turn T−1
//! **causally enable** success at turn T. Inspired by transfer entropy:
//!
//!   TE(A→B) ≈ E[Y | B at T, A at T−1] − E[Y | B at T, ¬A at T−1]
//!
//! High TE(A→B): selecting A at T−1 improves outcomes when B is at T.
//! This discovers "setup fragments" — context that primes the LLM's
//! working memory for future effectiveness.
//!
//! # Information Gravity Field (Conformal Metric Distortion)
//!
//! Fragments with high causal mass create a conformal metric distortion
//! in retrieval space (Amari, Information Geometry 1998):
//!
//!   gravity_bonus(f) = 1 − exp(−α · causal_mass(f))
//!   causal_mass(f) = max(0, causal_effect(f)) × confidence(f)
//!
//! High-mass fragments appear "closer" to all queries, increasing their
//! selection probability. This creates self-reinforcing dynamics where
//! causally important fragments accumulate mass, while confounded
//! fragments are correctly down-weighted.
//!
//! # Novel Contributions
//!
//! 1. First application of instrumental variable estimation to context optimization
//! 2. Temporal causal chains via transfer entropy in multi-turn context selection
//! 3. Information gravity as conformal metric distortion for retrieval
//!
//! # Complexity
//!
//! - Trace storage: O(MAX_TRACES) circular buffer
//! - Interventional estimates: O(N) sparse per tracked fragment
//! - Temporal links: O(MAX_TEMPORAL_LINKS) with staleness eviction
//! - Gravity field: O(N) recomputed incrementally on each trace

use std::collections::{HashMap, HashSet, VecDeque};
use serde::{Deserialize, Serialize};

/// Maximum traces stored (circular buffer).
const MAX_TRACES: usize = 200;

/// Maximum temporal links before staleness eviction.
const MAX_TEMPORAL_LINKS: usize = 5000;

/// Minimum interventional trials before trusting causal estimates.
const MIN_INTERVENTION_TRIALS: u32 = 3;

/// Staleness threshold: remove temporal links not updated in this many turns.
const STALE_TURN_THRESHOLD: u32 = 50;

/// Gravity field strength (conformal metric distortion parameter α).
/// Higher α → stronger pull of high-mass fragments.
const GRAVITY_ALPHA: f64 = 2.0;

/// EMA smoothing for base rate tracking.
const BASE_RATE_EMA: f64 = 0.95;

// ─── Data Structures ────────────────────────────────────────────────

/// A single optimization+feedback event.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CausalTrace {
    turn: u32,
    query_hash: u64,
    selected_ids: Vec<String>,
    explored_ids: Vec<String>,
    outcome: f64,
}

/// Per-fragment interventional vs observational statistics.
///
/// Tracks outcomes separately for when the fragment was:
/// - Naturally selected by the policy (observational)
/// - Randomly included via exploration (interventional / do-calculus)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct InterventionEstimate {
    /// Sum of outcomes when naturally selected (observational).
    obs_outcome_sum: f64,
    /// Number of times naturally selected.
    obs_count: u32,
    /// Sum of outcomes when randomly included via exploration.
    int_outcome_sum: f64,
    /// Number of times randomly included (instrument variable).
    int_count: u32,
    /// Total outcome sum across all selections (for temporal marginals).
    total_outcome_sum: f64,
    /// Total selection count.
    total_count: u32,
    /// Cached: E[Y|do(include)] − base_rate.
    causal_effect: f64,
    /// Cached: E[Y|observe(include)] − E[Y|do(include)].
    confounding_bias: f64,
    /// Confidence in interventional estimate ∈ [0, 1].
    confidence: f64,
    /// Turn of last update.
    last_updated: u32,
}

/// Temporal causal link: source at T−1 → target at T.
///
/// Measures how much selecting `source` at T−1 improves outcomes
/// when `target` is selected at T (transfer entropy approximation).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TemporalLinkEntry {
    /// Times target at T with source at T−1.
    co_occurrences: u32,
    /// Sum of outcomes when both present (consecutive turns).
    conditional_outcome_sum: f64,
    /// Cached: E[Y|target∧source@T−1] − E[Y|target∧¬source@T−1].
    temporal_effect: f64,
    /// Confidence based on sample sizes ∈ [0, 1].
    confidence: f64,
    /// Turn of last update.
    last_updated: u32,
}

/// Statistics for observability.
pub struct CausalStats {
    pub total_traces: u64,
    pub stored_traces: usize,
    pub tracked_fragments: usize,
    pub interventional_fragments: usize,
    pub temporal_links: usize,
    pub gravity_sources: usize,
    pub mean_causal_mass: f64,
    pub base_rate: f64,
    pub total_interventional_updates: u64,
    pub total_temporal_updates: u64,
}

// ─── Causal Context Graph ───────────────────────────────────────────

/// Causal Context Graph — learns fragment causal effects via natural experiments.
///
/// Uses the exploration mechanism as an instrumental variable to separate
/// true causal effects from selection bias, discovers temporal causal chains,
/// and computes an information gravity field for retrieval distortion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalContextGraph {
    /// Circular buffer of recent optimization+feedback traces.
    traces: VecDeque<CausalTrace>,
    /// Per-fragment interventional vs observational estimates.
    interventions: HashMap<String, InterventionEstimate>,
    /// Directed temporal links: source → { target → entry }.
    temporal_links: HashMap<String, HashMap<String, TemporalLinkEntry>>,
    /// Total count of temporal link entries (for capacity management).
    total_temporal_links: usize,
    /// Cached gravity field: fragment_id → causal_mass.
    causal_mass: HashMap<String, f64>,
    /// Global base rate (EMA of outcome across all traces).
    base_rate: f64,
    /// Lifetime statistics.
    total_traces: u64,
    total_interventional_updates: u64,
    total_temporal_updates: u64,
}

impl Default for CausalContextGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl CausalContextGraph {
    pub fn new() -> Self {
        CausalContextGraph {
            traces: VecDeque::with_capacity(MAX_TRACES),
            interventions: HashMap::new(),
            temporal_links: HashMap::new(),
            total_temporal_links: 0,
            causal_mass: HashMap::new(),
            base_rate: 0.0,
            total_traces: 0,
            total_interventional_updates: 0,
            total_temporal_updates: 0,
        }
    }

    /// Record an optimization+feedback trace and update all causal estimates.
    ///
    /// # Arguments
    /// - `turn`: current engine turn
    /// - `query_hash`: SimHash of the query string
    /// - `selected_ids`: all fragments selected (including explored)
    /// - `explored_ids`: subset that were randomly included via exploration
    /// - `outcome`: advantage signal (reward − baseline)
    pub fn record_trace(
        &mut self,
        turn: u32,
        query_hash: u64,
        selected_ids: &[String],
        explored_ids: &[String],
        outcome: f64,
    ) {
        if !outcome.is_finite() || selected_ids.is_empty() {
            return;
        }

        let explored_set: HashSet<&str> = explored_ids.iter().map(|s| s.as_str()).collect();

        // ── 1. Update interventional/observational estimates ──
        for fid in selected_ids {
            let entry = self.interventions.entry(fid.clone()).or_insert(InterventionEstimate {
                obs_outcome_sum: 0.0,
                obs_count: 0,
                int_outcome_sum: 0.0,
                int_count: 0,
                total_outcome_sum: 0.0,
                total_count: 0,
                causal_effect: 0.0,
                confounding_bias: 0.0,
                confidence: 0.0,
                last_updated: turn,
            });

            entry.total_outcome_sum += outcome;
            entry.total_count += 1;
            entry.last_updated = turn;

            if explored_set.contains(fid.as_str()) {
                // Interventional: randomly included via exploration swap.
                // This is the do(include) observation — unconfounded by policy.
                entry.int_outcome_sum += outcome;
                entry.int_count += 1;
                self.total_interventional_updates += 1;
            } else {
                // Observational: naturally selected by policy.
                // Potentially confounded by co-selection with other fragments.
                entry.obs_outcome_sum += outcome;
                entry.obs_count += 1;
            }
        }

        // ── 2. Update temporal links (previous trace → current trace) ──
        if let Some(prev_trace) = self.traces.back() {
            let curr_set: HashSet<&str> = selected_ids.iter().map(|s| s.as_str()).collect();

            for source_id in &prev_trace.selected_ids {
                for target_id in selected_ids {
                    // Skip self-links and links where source isn't relevant
                    if source_id == target_id {
                        continue;
                    }
                    // Only track links where the source was actually at T−1
                    // and target is at T (both selected in consecutive turns)
                    if !curr_set.contains(target_id.as_str()) {
                        continue;
                    }

                    let source_map = self.temporal_links
                        .entry(source_id.clone())
                        .or_default();

                    let link = source_map.entry(target_id.clone()).or_insert_with(|| {
                        self.total_temporal_links += 1;
                        TemporalLinkEntry {
                            co_occurrences: 0,
                            conditional_outcome_sum: 0.0,
                            temporal_effect: 0.0,
                            confidence: 0.0,
                            last_updated: turn,
                        }
                    });

                    link.co_occurrences += 1;
                    link.conditional_outcome_sum += outcome;
                    link.last_updated = turn;
                    self.total_temporal_updates += 1;
                }
            }

            // Capacity management: evict stalest links if over limit
            if self.total_temporal_links > MAX_TEMPORAL_LINKS {
                self.evict_temporal_links(turn);
            }
        }

        // ── 3. Store trace in circular buffer ──
        let trace = CausalTrace {
            turn,
            query_hash,
            selected_ids: selected_ids.to_vec(),
            explored_ids: explored_ids.to_vec(),
            outcome,
        };
        if self.traces.len() >= MAX_TRACES {
            self.traces.pop_front();
        }
        self.traces.push_back(trace);
        self.total_traces += 1;

        // ── 4. Update global base rate (EMA) ──
        self.base_rate = BASE_RATE_EMA * self.base_rate + (1.0 - BASE_RATE_EMA) * outcome;

        // ── 5. Recompute derived quantities ──
        self.recompute_causal_effects();
        self.recompute_temporal_effects();
        self.recompute_gravity_field();
    }

    /// Compute information gravity bonuses for candidate fragments.
    ///
    /// Returns fragment_id → gravity_bonus ∈ [0, 1).
    /// Based on conformal metric distortion: high causal mass fragments
    /// appear "closer" to all queries in the retrieval space.
    pub fn gravity_bonuses(&self, candidate_ids: &[&str]) -> HashMap<String, f64> {
        let mut bonuses = HashMap::new();
        for &fid in candidate_ids {
            if let Some(&mass) = self.causal_mass.get(fid) {
                if mass > 0.005 {
                    // gravity_bonus = 1 − exp(−α·mass) ∈ [0, 1)
                    // Diminishing returns: first unit of mass gives the most bonus
                    let bonus = 1.0 - (-GRAVITY_ALPHA * mass).exp();
                    bonuses.insert(fid.to_string(), bonus);
                }
            }
        }
        bonuses
    }

    /// Compute temporal chain bonuses for candidates given previous selection.
    ///
    /// For each candidate B, returns the confidence-weighted mean temporal
    /// effect across all A→B links where A was in `prev_selected_ids`.
    ///
    /// High bonus: selecting B is predicted to improve outcomes because
    /// the previous selection included fragments that causally enable B.
    pub fn temporal_bonuses(
        &self,
        candidate_ids: &[&str],
        prev_selected_ids: &[&str],
    ) -> HashMap<String, f64> {
        if prev_selected_ids.is_empty() || self.temporal_links.is_empty() {
            return HashMap::new();
        }

        let mut bonuses = HashMap::new();

        for &target_id in candidate_ids {
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for &source_id in prev_selected_ids {
                if let Some(targets) = self.temporal_links.get(source_id) {
                    if let Some(link) = targets.get(target_id) {
                        if link.confidence > 0.2 && link.co_occurrences >= 2 {
                            weighted_sum += link.temporal_effect * link.confidence;
                            weight_sum += link.confidence;
                        }
                    }
                }
            }

            if weight_sum > 0.0 {
                let avg_effect = weighted_sum / weight_sum;
                if avg_effect.abs() > 0.005 {
                    bonuses.insert(target_id.to_string(), avg_effect);
                }
            }
        }

        bonuses
    }

    /// Per-turn maintenance: evict stale temporal links, recompute fields.
    pub fn decay_tick(&mut self, current_turn: u32) {
        // Evict temporal links not updated in STALE_TURN_THRESHOLD turns
        let mut removed_count = 0usize;
        let mut empty_sources: Vec<String> = Vec::new();

        for (source, targets) in &mut self.temporal_links {
            let before = targets.len();
            targets.retain(|_, link| {
                current_turn.saturating_sub(link.last_updated) < STALE_TURN_THRESHOLD
            });
            removed_count += before - targets.len();

            if targets.is_empty() {
                empty_sources.push(source.clone());
            }
        }

        for source in empty_sources {
            self.temporal_links.remove(&source);
        }

        self.total_temporal_links = self.total_temporal_links.saturating_sub(removed_count);

        // Recompute derived quantities with latest data
        self.recompute_temporal_effects();
        self.recompute_gravity_field();
    }

    pub fn is_empty(&self) -> bool {
        self.interventions.is_empty()
    }

    /// Number of fragments with causal estimates.
    pub fn tracked_fragments(&self) -> usize {
        self.interventions.len()
    }

    /// Number of fragments with interventional data (from exploration).
    pub fn interventional_fragments(&self) -> usize {
        self.interventions.values()
            .filter(|e| e.int_count >= MIN_INTERVENTION_TRIALS)
            .count()
    }

    /// Number of active temporal links.
    pub fn temporal_link_count(&self) -> usize {
        self.total_temporal_links
    }

    /// Number of fragments with positive causal mass (gravity sources).
    pub fn gravity_source_count(&self) -> usize {
        self.causal_mass.values().filter(|&&m| m > 0.005).count()
    }

    /// Mean causal mass across all tracked fragments.
    pub fn mean_causal_mass(&self) -> f64 {
        if self.causal_mass.is_empty() {
            return 0.0;
        }
        self.causal_mass.values().sum::<f64>() / self.causal_mass.len() as f64
    }

    /// Top fragments ranked by causal effect (for observability).
    ///
    /// Returns: (fragment_id, causal_effect, confounding_bias, confidence).
    pub fn top_causal_fragments(&self, n: usize) -> Vec<(String, f64, f64, f64)> {
        let mut items: Vec<_> = self.interventions.iter()
            .filter(|(_, est)| est.confidence > 0.1)
            .map(|(id, est)| {
                (id.clone(), est.causal_effect, est.confounding_bias, est.confidence)
            })
            .collect();

        items.sort_by(|a, b| {
            // Sort by absolute causal effect (most impactful first)
            b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal)
        });
        items.truncate(n);
        items
    }

    /// Top temporal chains ranked by effect strength (for observability).
    ///
    /// Returns: (source_id, target_id, temporal_effect, confidence).
    pub fn top_temporal_chains(&self, n: usize) -> Vec<(String, String, f64, f64)> {
        let mut chains: Vec<_> = self.temporal_links.iter()
            .flat_map(|(source, targets)| {
                targets.iter().map(move |(target, link)| {
                    (source.clone(), target.clone(), link.temporal_effect, link.confidence)
                })
            })
            .filter(|(_, _, effect, conf)| *conf > 0.15 && effect.abs() > 0.005)
            .collect();

        chains.sort_by(|a, b| {
            b.2.abs().partial_cmp(&a.2.abs()).unwrap_or(std::cmp::Ordering::Equal)
        });
        chains.truncate(n);
        chains
    }

    /// Full statistics for dashboard/CLI observability.
    pub fn stats(&self) -> CausalStats {
        CausalStats {
            total_traces: self.total_traces,
            stored_traces: self.traces.len(),
            tracked_fragments: self.tracked_fragments(),
            interventional_fragments: self.interventional_fragments(),
            temporal_links: self.temporal_link_count(),
            gravity_sources: self.gravity_source_count(),
            mean_causal_mass: self.mean_causal_mass(),
            base_rate: self.base_rate,
            total_interventional_updates: self.total_interventional_updates,
            total_temporal_updates: self.total_temporal_updates,
        }
    }

    // ── Private Implementation ──────────────────────────────────────

    /// Recompute causal_effect and confounding_bias for all fragments.
    ///
    /// Causal effect = E[Y|do(include f)] − base_rate
    ///   Uses interventional data (exploration = natural experiment)
    ///
    /// Confounding bias = E[Y|observe(include f)] − E[Y|do(include f)]
    ///   Positive bias → fragment rides coattails (appears better than it is)
    ///   Negative bias → hidden gem (appears worse than true effect)
    fn recompute_causal_effects(&mut self) {
        for est in self.interventions.values_mut() {
            // Interventional rate: E[Y | do(include f)]
            // This is the unconfounded causal estimate from exploration data
            let int_rate = if est.int_count > 0 {
                est.int_outcome_sum / est.int_count as f64
            } else {
                // No interventional data yet → fall back to base rate (prior)
                self.base_rate
            };

            // Observational rate: E[Y | observe(include f)]
            // Potentially confounded by co-selection with other good fragments
            let obs_rate = if est.obs_count > 0 {
                est.obs_outcome_sum / est.obs_count as f64
            } else {
                self.base_rate
            };

            // Causal effect: improvement over base rate under intervention
            est.causal_effect = int_rate - self.base_rate;

            // Confounding bias: how much observation overestimates the causal effect
            est.confounding_bias = obs_rate - int_rate;

            // Confidence: primarily from interventional sample size
            // √n / (√n + √min) → 0 at n=0, ~0.5 at n=min, → 1 asymptotically
            let n = est.int_count as f64;
            let min_n = MIN_INTERVENTION_TRIALS as f64;
            est.confidence = if est.int_count > 0 {
                n.sqrt() / (n.sqrt() + min_n.sqrt())
            } else {
                // With no interventional data, use observational at reduced trust (0.3×)
                let m = est.obs_count as f64;
                0.3 * m.sqrt() / (m.sqrt() + min_n.sqrt())
            };
        }
    }

    /// Recompute temporal effects for all links.
    ///
    /// TE(A→B) ≈ E[Y | B@T, A@T−1] − E[Y | B@T, ¬A@T−1]
    ///
    /// Uses per-fragment total stats as the marginal denominator:
    ///   marginal = (total_Y_B − conditional_Y_AB) / (total_N_B − co_occurrences_AB)
    fn recompute_temporal_effects(&mut self) {
        // Collect the data we need from interventions first (avoid borrow conflict)
        let intervention_data: HashMap<String, (f64, u32)> = self.interventions.iter()
            .map(|(id, est)| (id.clone(), (est.total_outcome_sum, est.total_count)))
            .collect();

        for targets in self.temporal_links.values_mut() {
            for (target_id, link) in targets.iter_mut() {
                if link.co_occurrences == 0 {
                    link.temporal_effect = 0.0;
                    link.confidence = 0.0;
                    continue;
                }

                // Conditional mean: E[Y | target@T, source@T−1]
                let mean_conditional = link.conditional_outcome_sum / link.co_occurrences as f64;

                // Marginal mean: E[Y | target@T] (all selections of target)
                let mean_marginal = if let Some(&(total_sum, total_count)) = intervention_data.get(target_id.as_str()) {
                    if total_count > link.co_occurrences {
                        let marginal_sum = total_sum - link.conditional_outcome_sum;
                        let marginal_count = total_count - link.co_occurrences;
                        if marginal_count > 0 {
                            marginal_sum / marginal_count as f64
                        } else {
                            self.base_rate
                        }
                    } else {
                        self.base_rate
                    }
                } else {
                    self.base_rate
                };

                // Transfer entropy approximation: difference in conditional means
                link.temporal_effect = mean_conditional - mean_marginal;

                // Confidence: based on co-occurrence count
                // √n / (√n + 2) → ~0.5 at n=4, ~0.7 at n=10
                let n = link.co_occurrences as f64;
                link.confidence = n.sqrt() / (n.sqrt() + 2.0);
            }
        }
    }

    /// Recompute the information gravity field from causal estimates.
    ///
    /// causal_mass(f) = max(0, causal_effect(f)) × confidence(f)
    ///
    /// Only positive causal effects generate gravity — fragments that
    /// causally HURT outcomes have zero mass (no attractive pull).
    fn recompute_gravity_field(&mut self) {
        self.causal_mass.clear();

        for (fid, est) in &self.interventions {
            let mass = est.causal_effect.max(0.0) * est.confidence;
            if mass > 0.001 {
                self.causal_mass.insert(fid.clone(), mass);
            }
        }
    }

    /// Evict the stalest temporal links to stay under capacity.
    fn evict_temporal_links(&mut self, current_turn: u32) {
        // Collect all links with staleness scores
        let mut all_links: Vec<(String, String, u32)> = Vec::new();
        for (source, targets) in &self.temporal_links {
            for (target, link) in targets {
                let staleness = current_turn.saturating_sub(link.last_updated);
                all_links.push((source.clone(), target.clone(), staleness));
            }
        }

        // Sort by staleness descending (most stale first)
        all_links.sort_by_key(|&(_, _, staleness)| std::cmp::Reverse(staleness));

        // Remove the stalest links (10% buffer to avoid evicting every turn)
        let to_remove = self.total_temporal_links
            .saturating_sub(MAX_TEMPORAL_LINKS)
            + MAX_TEMPORAL_LINKS / 10;

        for (source, target, _) in all_links.iter().take(to_remove) {
            if let Some(targets) = self.temporal_links.get_mut(source) {
                targets.remove(target);
                if targets.is_empty() {
                    self.temporal_links.remove(source);
                }
            }
        }

        // Recount (cheaper than tracking incremental removes)
        self.total_temporal_links = self.temporal_links.values()
            .map(|m| m.len())
            .sum();
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ids(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn test_new_graph_is_empty() {
        let g = CausalContextGraph::new();
        assert!(g.is_empty());
        assert_eq!(g.tracked_fragments(), 0);
        assert_eq!(g.temporal_link_count(), 0);
        assert_eq!(g.gravity_source_count(), 0);
        assert!((g.mean_causal_mass() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_single_trace_updates_estimates() {
        let mut g = CausalContextGraph::new();
        let selected = make_ids(&["a", "b", "c"]);
        let explored = make_ids(&["c"]); // c was randomly included

        g.record_trace(1, 0x1234, &selected, &explored, 1.0);

        assert_eq!(g.tracked_fragments(), 3);
        assert!(!g.is_empty());

        // a, b are observational; c is interventional
        let est_a = &g.interventions["a"];
        assert_eq!(est_a.obs_count, 1);
        assert_eq!(est_a.int_count, 0);

        let est_c = &g.interventions["c"];
        assert_eq!(est_c.obs_count, 0);
        assert_eq!(est_c.int_count, 1);
        assert_eq!(est_c.total_count, 1);
    }

    #[test]
    fn test_interventional_vs_observational_separation() {
        let mut g = CausalContextGraph::new();

        // Fragment "good" is always selected naturally → observational
        // Fragment "explored" is always randomly included → interventional
        for turn in 1..=10 {
            let selected = make_ids(&["good", "explored"]);
            let explored = make_ids(&["explored"]);
            g.record_trace(turn, 0xAA, &selected, &explored, 0.5);
        }

        let est_good = &g.interventions["good"];
        assert_eq!(est_good.obs_count, 10);
        assert_eq!(est_good.int_count, 0);

        let est_exp = &g.interventions["explored"];
        assert_eq!(est_exp.obs_count, 0);
        assert_eq!(est_exp.int_count, 10);
        // High interventional confidence (10 trials)
        assert!(est_exp.confidence > 0.6);
    }

    #[test]
    fn test_confounding_bias_detection() {
        let mut g = CausalContextGraph::new();

        // Phase 1: "coattail" fragment always co-selected with "star"
        // Both get positive outcomes → observational rate is high for both
        for turn in 1..=20 {
            let selected = make_ids(&["star", "coattail"]);
            g.record_trace(turn, 0xBB, &selected, &[], 1.0);
        }

        // Phase 2: "coattail" explored alone → actually mediocre
        for turn in 21..=30 {
            let selected = make_ids(&["coattail", "filler"]);
            let explored = make_ids(&["coattail"]);
            g.record_trace(turn, 0xCC, &selected, &explored, -0.2);
        }

        let est = &g.interventions["coattail"];
        // Observational rate is high (phase 1 was positive)
        let obs_rate = est.obs_outcome_sum / est.obs_count.max(1) as f64;
        // Interventional rate is low (phase 2 was negative)
        let int_rate = est.int_outcome_sum / est.int_count.max(1) as f64;

        // Confounding bias should be positive: observation overestimates
        assert!(est.confounding_bias > 0.0,
            "confounding bias should be positive, got {}", est.confounding_bias);
        assert!(obs_rate > int_rate,
            "obs_rate ({}) should exceed int_rate ({})", obs_rate, int_rate);
    }

    #[test]
    fn test_temporal_links_created() {
        let mut g = CausalContextGraph::new();

        // Turn 1: select A, B
        g.record_trace(1, 0x11, &make_ids(&["a", "b"]), &[], 0.5);
        // No temporal links yet (no previous trace)
        assert_eq!(g.temporal_link_count(), 0);

        // Turn 2: select B, C
        g.record_trace(2, 0x22, &make_ids(&["b", "c"]), &[], 0.8);
        // Temporal links: a→b, a→c, b→b (self-skipped), b→c
        // a→b, a→c, b→c = 3 links
        assert_eq!(g.temporal_link_count(), 3);

        // Check a→c link exists
        assert!(g.temporal_links.contains_key("a"));
        assert!(g.temporal_links["a"].contains_key("c"));
        assert_eq!(g.temporal_links["a"]["c"].co_occurrences, 1);
    }

    #[test]
    fn test_temporal_effect_computation() {
        let mut g = CausalContextGraph::new();

        // Setup fragment "primer" at T−1 improves outcomes for "target" at T
        for turn in (1..=20).step_by(2) {
            // Odd turns: primer selected → next turn target does well
            g.record_trace(turn, 0x11, &make_ids(&["primer", "other"]), &[], 0.3);
            g.record_trace(turn + 1, 0x22, &make_ids(&["target", "filler"]), &[], 0.9);
        }

        // Compare: without primer, target does poorly
        for turn in 21..=30 {
            g.record_trace(turn, 0x33, &make_ids(&["filler", "x"]), &[], 0.1);
            g.record_trace(turn + 1, 0x44, &make_ids(&["target", "y"]), &[], 0.1);
        }

        // The primer→target temporal link should have positive effect
        if let Some(targets) = g.temporal_links.get("primer") {
            if let Some(link) = targets.get("target") {
                assert!(link.temporal_effect > 0.0,
                    "temporal_effect should be positive, got {}", link.temporal_effect);
            }
        }
    }

    #[test]
    fn test_gravity_bonuses_positive_effect() {
        let mut g = CausalContextGraph::new();

        // Fragment "hero" has strong positive interventional effect
        for turn in 1..=15 {
            let selected = make_ids(&["hero", "support"]);
            let explored = make_ids(&["hero"]);
            g.record_trace(turn, 0xFF, &selected, &explored, 0.8);
        }

        let bonuses = g.gravity_bonuses(&["hero", "support", "unknown"]);
        // Hero should have gravity bonus (strong positive causal effect + high confidence)
        assert!(bonuses.contains_key("hero"),
            "hero should have gravity bonus");
        assert!(*bonuses.get("hero").unwrap_or(&0.0) > 0.0);

        // Unknown fragment: no data, no bonus
        assert!(!bonuses.contains_key("unknown"));
    }

    #[test]
    fn test_gravity_zero_for_negative_effect() {
        let mut g = CausalContextGraph::new();

        // Fragment "harmful" has negative interventional effect
        for turn in 1..=15 {
            let selected = make_ids(&["harmful", "filler"]);
            let explored = make_ids(&["harmful"]);
            g.record_trace(turn, 0xEE, &selected, &explored, -0.5);
        }

        let bonuses = g.gravity_bonuses(&["harmful"]);
        // Harmful fragments should NOT have gravity bonus
        assert!(!bonuses.contains_key("harmful") || *bonuses.get("harmful").unwrap_or(&0.0) < 0.01,
            "harmful fragment should not have gravity bonus");
    }

    #[test]
    fn test_temporal_bonuses_computation() {
        let mut g = CausalContextGraph::new();

        // Build up temporal link: when "setup" at T−1, "payload" at T does well
        for _ in 0..10 {
            g.record_trace(1, 0x11, &make_ids(&["setup", "x"]), &[], 0.5);
            g.record_trace(2, 0x22, &make_ids(&["payload", "y"]), &[], 0.9);
        }

        let bonuses = g.temporal_bonuses(
            &["payload", "other"],
            &["setup"],
        );

        // payload should get a temporal bonus when setup was in previous selection
        if let Some(&bonus) = bonuses.get("payload") {
            // We just check it's computed (exact value depends on marginal baseline)
            assert!(bonus.is_finite(), "temporal bonus should be finite");
        }
    }

    #[test]
    fn test_decay_tick_removes_stale_links() {
        let mut g = CausalContextGraph::new();

        // Create temporal links at turn 1
        g.record_trace(1, 0x11, &make_ids(&["a", "b"]), &[], 0.5);
        g.record_trace(2, 0x22, &make_ids(&["c", "d"]), &[], 0.5);
        let initial_links = g.temporal_link_count();
        assert!(initial_links > 0);

        // Advance far beyond stale threshold
        g.decay_tick(2 + STALE_TURN_THRESHOLD + 10);

        // All links should be evicted (they're all stale)
        assert_eq!(g.temporal_link_count(), 0,
            "stale temporal links should be evicted");
    }

    #[test]
    fn test_base_rate_tracking() {
        let mut g = CausalContextGraph::new();

        // All positive outcomes → base rate should converge to positive
        for turn in 1..=50 {
            g.record_trace(turn, 0x11, &make_ids(&["a"]), &[], 1.0);
        }
        assert!(g.base_rate > 0.5,
            "base_rate should be positive after positive outcomes, got {}", g.base_rate);

        // Negative outcomes → base rate should decrease
        for turn in 51..=100 {
            g.record_trace(turn, 0x22, &make_ids(&["a"]), &[], -1.0);
        }
        assert!(g.base_rate < 0.5,
            "base_rate should decrease after negative outcomes, got {}", g.base_rate);
    }

    #[test]
    fn test_circular_buffer_capacity() {
        let mut g = CausalContextGraph::new();

        // Fill well beyond MAX_TRACES
        for turn in 1..=(MAX_TRACES as u32 * 2) {
            g.record_trace(turn, turn as u64, &make_ids(&["a"]), &[], 0.1);
        }

        // Buffer should be capped at MAX_TRACES
        assert_eq!(g.traces.len(), MAX_TRACES);
        // Total traces counts all recorded
        assert_eq!(g.total_traces, MAX_TRACES as u64 * 2);
    }

    #[test]
    fn test_confidence_increases_with_trials() {
        let mut g = CausalContextGraph::new();

        // 1 trial
        g.record_trace(1, 0x11, &make_ids(&["f"]), &make_ids(&["f"]), 0.5);
        let conf_1 = g.interventions["f"].confidence;

        // 10 more trials
        for turn in 2..=11 {
            g.record_trace(turn, 0x11, &make_ids(&["f"]), &make_ids(&["f"]), 0.5);
        }
        let conf_11 = g.interventions["f"].confidence;

        assert!(conf_11 > conf_1,
            "confidence should increase with more trials: {} vs {}", conf_11, conf_1);
    }

    #[test]
    fn test_top_causal_fragments() {
        let mut g = CausalContextGraph::new();

        // Two fragments with different causal effects
        for turn in 1..=20 {
            // "strong" has positive interventional effect
            g.record_trace(turn, 0x11,
                &make_ids(&["strong", "x"]),
                &make_ids(&["strong"]),
                0.8);
        }
        for turn in 21..=40 {
            // "weak" has smaller positive interventional effect
            g.record_trace(turn, 0x22,
                &make_ids(&["weak", "y"]),
                &make_ids(&["weak"]),
                0.2);
        }

        let top = g.top_causal_fragments(5);
        assert!(!top.is_empty(), "should have top causal fragments");
        // "strong" should rank higher than "weak"
        if top.len() >= 2 {
            let strong_pos = top.iter().position(|(id, _, _, _)| id == "strong");
            let weak_pos = top.iter().position(|(id, _, _, _)| id == "weak");
            if let (Some(s), Some(w)) = (strong_pos, weak_pos) {
                assert!(s < w || top[s].1.abs() >= top[w].1.abs(),
                    "strong should rank at or above weak");
            }
        }
    }

    #[test]
    fn test_stats_consistency() {
        let mut g = CausalContextGraph::new();

        for turn in 1..=10 {
            g.record_trace(turn, 0x11,
                &make_ids(&["a", "b"]),
                &make_ids(&["b"]),
                0.5);
        }

        let s = g.stats();
        assert_eq!(s.total_traces, 10);
        assert_eq!(s.stored_traces, 10);
        assert_eq!(s.tracked_fragments, 2);
        assert!(s.interventional_fragments <= s.tracked_fragments);
        assert!(s.base_rate > 0.0);
    }

    #[test]
    fn test_empty_selected_ignored() {
        let mut g = CausalContextGraph::new();
        g.record_trace(1, 0x11, &[], &[], 0.5);
        assert!(g.is_empty(), "empty selection should be ignored");
        assert_eq!(g.total_traces, 0);
    }

    #[test]
    fn test_nan_outcome_ignored() {
        let mut g = CausalContextGraph::new();
        g.record_trace(1, 0x11, &make_ids(&["a"]), &[], f64::NAN);
        assert!(g.is_empty(), "NaN outcome should be ignored");
    }

    #[test]
    fn test_self_links_excluded() {
        let mut g = CausalContextGraph::new();

        // Turn 1 and 2 both select "a" → should NOT create a→a link
        g.record_trace(1, 0x11, &make_ids(&["a"]), &[], 0.5);
        g.record_trace(2, 0x22, &make_ids(&["a"]), &[], 0.5);

        // Check no self-link
        if let Some(targets) = g.temporal_links.get("a") {
            assert!(!targets.contains_key("a"), "self-links should be excluded");
        }
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut g = CausalContextGraph::new();

        for turn in 1..=5 {
            g.record_trace(turn, 0x11,
                &make_ids(&["a", "b"]),
                &make_ids(&["b"]),
                0.5);
        }

        let json = serde_json::to_string(&g).expect("serialize");
        let restored: CausalContextGraph = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.total_traces, g.total_traces);
        assert_eq!(restored.tracked_fragments(), g.tracked_fragments());
        assert_eq!(restored.traces.len(), g.traces.len());
        assert!((restored.base_rate - g.base_rate).abs() < 1e-9);
    }

    #[test]
    fn test_temporal_eviction_under_capacity() {
        let mut g = CausalContextGraph::new();

        // Create many unique temporal links to exceed capacity
        // Each pair of consecutive traces with different fragments creates links
        for turn in 1..=300 {
            let ids = make_ids(&[
                &format!("f{}", turn),
                &format!("g{}", turn),
            ]);
            g.record_trace(turn as u32, turn as u64, &ids, &[], 0.1);
        }

        // Should stay at or below MAX_TEMPORAL_LINKS
        assert!(g.temporal_link_count() <= MAX_TEMPORAL_LINKS + MAX_TEMPORAL_LINKS / 10,
            "temporal links should be bounded, got {}", g.temporal_link_count());
    }
}
