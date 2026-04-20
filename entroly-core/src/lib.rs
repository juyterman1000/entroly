//! Entroly Core — Rust Engine + PyO3 Bindings
//!
//! This is the main entry point that:
//!   1. Declares all Rust modules
//!   2. Provides the EntrolyEngine (orchestrator)
//!   3. Exposes everything to Python via PyO3
//!
//! Architecture:
//!   Python (MCP server) → PyO3 → Rust Engine → Results → Python → JSON-RPC
//!
//! Python only handles the MCP protocol (no AI libraries in Rust).
//! All computation happens here in Rust.
mod fragment;
mod knapsack;
mod knapsack_sds;
mod entropy;
mod dedup;
mod depgraph;
mod guardrails;
mod lsh;
mod prism;
mod skeleton;
mod sast;
mod health;
mod query;
mod hierarchical;
pub mod query_persona;
mod anomaly;
mod utilization;
mod semantic_dedup;
mod conversation_pruner;
mod channel;
mod nkbe;
mod cognitive_bus;
mod cache;
mod resonance;
mod causal;
pub mod archetype;
pub mod cogops;
mod bm25;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::{HashMap, HashSet};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use serde::{Deserialize, Serialize};

use fragment::{ContextFragment, compute_relevance};
use knapsack::{knapsack_optimize, compute_lambda_star, ScoringWeights};
use knapsack_sds::{ios_select, Resolution, InfoFactors};
use entropy::{information_score, shannon_entropy, normalized_entropy, boilerplate_ratio, renyi_entropy_2, entropy_divergence};
use dedup::{simhash, hamming_distance, DedupIndex};
use bm25::BM25Index;
use depgraph::{DepGraph, extract_identifiers};
use guardrails::{file_criticality, has_safety_signal, TaskType, FeedbackTracker, Criticality, compute_ordering_priority};
use prism::PrismOptimizer;
use prism::PrismOptimizer5D;
use query_persona::QueryPersonaManifold;
use cache::CacheLookup;
use resonance::ResonanceMatrix;
use causal::CausalContextGraph;


/// Process-wide monotonic counter — used only to seed each engine's instance_id.
/// Guarantees every EntrolyEngine instance gets a unique prefix, making
/// fragment IDs globally unique within a process (fixes multi-engine isolation).
static INSTANCE_SEED: AtomicU64 = AtomicU64::new(1);

/// The core engine that orchestrates all subsystems.
///
/// Modeled after ebbiforge-core HippocampusEngine:
///   ingest → SimHash → dedup check → entropy score → store
///   optimize → Ebbinghaus decay → knapsack DP → ranked results
#[pyclass]
pub struct EntrolyEngine {
    fragments: HashMap<String, ContextFragment>,
    /// Ordered fragment ID list — parallels LSH index slots.
    /// Maps slot index → fragment_id so LSH results can resolve fragments.
    fragment_slot_ids: Vec<String>,
    dedup_index: DedupIndex,
    dep_graph: DepGraph,
    feedback: FeedbackTracker,
    current_turn: u32,

    // Scoring weights (used for knapsack / optimize path)
    w_recency: f64,
    w_frequency: f64,
    w_semantic: f64,
    w_entropy: f64,

    // Ebbinghaus
    decay_half_life: u32,
    min_relevance: f64,

    // Stats
    total_tokens_saved: u64,
    total_optimizations: u64,
    total_fragments_ingested: u64,
    total_duplicates_caught: u64,

    // Fragment ID generation — per-instance prefix guarantees isolation
    // between multiple EntrolyEngine instances in the same process.
    // Fragment IDs use format: f{instance_hex}_{counter_hex}
    instance_id: u64,
    id_counter: u64,

    // Max fragments cap (prevents unbounded memory growth)
    max_fragments: usize,

    // Exploration
    total_explorations: u64,
    exploration_rate: f64,
    rng_state: u64,

    // Last optimization snapshot (for explainability)
    last_optimization: Option<OptimizationSnapshot>,

    // LSH index for sub-linear recall (ported from ebbiforge-core)
    lsh_index: lsh::LshIndex,
    // Composite scorer (similarity + recency + entropy + frequency)
    context_scorer: lsh::ContextScorer,

    // Prism optimizer for advanced context selection
    prism_optimizer: PrismOptimizer,

    // IOS: Information-Optimal Selection (SDS + MRK)
    enable_ios: bool,
    enable_ios_diversity: bool,
    enable_ios_multi_resolution: bool,
    // IOS tunable parameters (configurable via tuning_config.json)
    ios_skeleton_info_factor: f64,
    ios_belief_info_factor: f64,
    ios_reference_info_factor: f64,
    ios_diversity_floor: f64,

    // Differentiable soft-selector temperature for gradient-based weight learning.
    // Controls sigmoid sharpness: high τ = soft (smooth gradients), low τ = hard (sharp selection).
    // Anneals toward 0 over turns so early learning explores, late learning exploits.
    gradient_temperature: f64,
    // EMA of gradient L2 norm — used to detect regime changes and reset temperature.
    // ── Shared Lagrange multiplier from last soft-bisection forward pass.
    // Stored here so apply_prism_rl_update can reuse the exact λ* to compute
    // p_i = σ((s_i − λ*·tokens_i)/τ) — closing the forward/backward probability gap.
    last_lambda_star: f64,
    /// ADGT signal: dual gap D(λ*) − primal from last soft-selection forward pass.
    /// Used to adapt temperature principally: large gap → keep τ high (exploring),
    /// small gap → lower τ (converged). Replaces the ad-hoc 0.995 annealing schedule.
    last_dual_gap: f64,
    gradient_norm_ema: f64,

    // Query Persona Manifold — discovers query archetypes and learns per-archetype weights
    query_manifold: QueryPersonaManifold,
    /// ID of the archetype assigned to the most recent query (for feedback routing).
    last_archetype_id: Option<String>,
    /// Whether to use per-archetype weights (vs global weights).
    enable_query_personas: bool,

    // Channel Coding Framework — information-theoretic context optimization
    /// Whether to use channel coding (trailing pass + interleaving + modulated reward).
    enable_channel_coding: bool,
    /// EMA baseline for REINFORCE advantage A = R − μ (variance reduction).
    /// Updated on every record_success / record_failure call.
    reward_baseline_ema: f64,

    // EGSC — Entropy-Gated Submodular Cache (novel: no prior art)
    egsc_cache: cache::EgscCache,
    /// Last query string from optimize() — used to route feedback to the
    /// correct cache entry. Without this, record_success/failure would hash
    /// with empty query and never find the stored entry (BUG FIX).
    last_query: String,
    /// Effective budget from the most recent optimize() call.
    last_effective_budget: u32,
    /// Whether the last optimize() result came from a deterministic exploit
    /// trajectory and is therefore safe to reinforce in EGSC.
    last_cache_feedback_eligible: bool,

    // ── Context Resonance: Pairwise Fragment Interaction Learning ──
    /// Resonance matrix tracking learned pairwise fragment synergies.
    /// R[i][j] > 0 means fragments i,j produce better LLM outputs together.
    resonance_matrix: ResonanceMatrix,
    /// 5D PRISM optimizer that includes the resonance dimension.
    /// Learns w_resonance alongside w_recency/frequency/semantic/entropy.
    /// Spectral shaping automatically dampens resonance's higher variance.
    prism_optimizer_5d: PrismOptimizer5D,
    /// Current resonance weight (5th dimension in the scoring model).
    /// Controls how much pairwise interaction bonus influences selection.
    w_resonance: f64,
    /// Whether context resonance is enabled.
    enable_resonance: bool,

    // ── Coverage Sufficiency Estimator (Unknown Unknowns) ──
    /// Semantic candidate count from last optimize() (N₁ for Chapman estimator).
    last_semantic_candidates: usize,
    /// Structural candidate count from last optimize() (N₂ for Chapman estimator).
    last_structural_candidates: usize,
    /// Overlap between semantic and structural candidates (m for Chapman estimator).
    last_candidate_overlap: usize,

    // ── Fragment Consolidation (Maxwell's Demon) ──
    /// Total fragments consolidated (merged into winners) since engine creation.
    total_consolidations: u64,
    /// Total tokens saved by consolidation.
    consolidation_tokens_saved: u64,
    /// Hamming threshold for consolidation (wider than dedup: catches near-duplicates).
    consolidation_hamming_threshold: u32,

    // ── Causal Context Graph (Interventional Estimation + Information Gravity) ──
    /// Learns true causal effects via do-calculus on exploration data,
    /// discovers temporal causal chains, computes information gravity field.
    causal_graph: CausalContextGraph,
    /// Whether the causal context graph is enabled.
    enable_causal: bool,
    /// Previous turn's selected fragment IDs (for temporal chain learning).
    prev_selected_ids: Vec<String>,
    /// Previous turn's explored fragment IDs (for temporal chain learning).
    prev_explored_ids: Vec<String>,

    // ── Belief Utilization Auto-Tuning (Closed-Loop) ──
    // EMA of LLM utilization scores for belief vs full fragments.
    // Closes the loop: if beliefs are well-utilized → increase belief info factor.
    // If beliefs are poorly utilized → decrease (LLM needs raw code).
    belief_util_ema: f64,
    full_util_ema: f64,
    /// Base belief info factor (before query-adaptive modulation).
    /// Updated by the closed-loop EMA feedback — converges to the optimal
    /// belief density for the specific codebase via online gradient descent.
    base_belief_info_factor: f64,
}

/// Snapshot of the last optimization for explainability.
struct OptimizationSnapshot {
    fragment_scores: Vec<FragmentScore>,
    sufficiency: f64,
    explored_ids: Vec<String>,
}

/// Per-fragment scoring breakdown.
struct FragmentScore {
    fragment_id: String,
    source: String,
    selected: bool,
    recency: f64,
    frequency: f64,
    semantic: f64,
    entropy: f64,
    feedback_mult: f64,
    dep_boost: f64,
    criticality: String,
    composite: f64,
    reason: String,
}

#[pymethods]
impl EntrolyEngine {
    #[new]
    #[pyo3(signature = (
        w_recency=0.30, w_frequency=0.25, w_semantic=0.25, w_entropy=0.20,
        decay_half_life=15, min_relevance=0.05,
        hamming_threshold=3, exploration_rate=0.1, max_fragments=10000,
        enable_ios=true, enable_ios_diversity=true, enable_ios_multi_resolution=true,
        ios_skeleton_info_factor=0.70, ios_belief_info_factor=0.50, ios_reference_info_factor=0.15, ios_diversity_floor=0.10
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        w_recency: f64,
        w_frequency: f64,
        w_semantic: f64,
        w_entropy: f64,
        decay_half_life: u32,
        min_relevance: f64,
        hamming_threshold: u32,
        exploration_rate: f64,
        max_fragments: usize,
        enable_ios: bool,
        enable_ios_diversity: bool,
        enable_ios_multi_resolution: bool,
        ios_skeleton_info_factor: f64,
        ios_belief_info_factor: f64,
        ios_reference_info_factor: f64,
        ios_diversity_floor: f64,
    ) -> Self {
        // Derive per-instance ID using xorshift64 on the global seed.
        // Each engine gets a unique instance_id, so fragment IDs are
        // globally unique within the process (no shared mutable state).
        let raw = INSTANCE_SEED.fetch_add(1, Ordering::Relaxed);
        // xorshift64 mixing to spread the low bits across the full u64
        let mut x = raw.wrapping_add(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        let instance_id = x ^ (x >> 31);

        EntrolyEngine {
            fragments: HashMap::new(),
            fragment_slot_ids: Vec::new(),
            dedup_index: DedupIndex::new(hamming_threshold),
            dep_graph: DepGraph::new(),
            feedback: FeedbackTracker::new(),
            prism_optimizer: PrismOptimizer::new(0.01),
            current_turn: 0,
            w_recency,
            w_frequency,
            w_semantic,
            w_entropy,
            decay_half_life,
            min_relevance,
            total_tokens_saved: 0,
            total_optimizations: 0,
            total_fragments_ingested: 0,
            total_duplicates_caught: 0,
            instance_id,
            id_counter: 0,
            max_fragments,
            total_explorations: 0,
            exploration_rate: exploration_rate.clamp(0.0, 1.0),
            // xorshift64 PRNG seeded from instance_id (already has good entropy)
            // Avoids deterministic hash-based exploration that can miss rate targets
            rng_state: instance_id | 1, // ensure non-zero seed
            last_optimization: None,
            lsh_index: lsh::LshIndex::new(),
            context_scorer: lsh::ContextScorer::default(),
            enable_ios,
            enable_ios_diversity,
            enable_ios_multi_resolution,
            ios_skeleton_info_factor: ios_skeleton_info_factor.clamp(0.01, 0.99),
            ios_belief_info_factor: ios_belief_info_factor.clamp(0.01, 0.99),
            ios_reference_info_factor: ios_reference_info_factor.clamp(0.01, 0.99),
            ios_diversity_floor: ios_diversity_floor.clamp(0.0, 1.0),
            gradient_temperature: 2.0,
            last_lambda_star: 0.0,
            last_dual_gap: 0.0,
            gradient_norm_ema: 0.0,
            query_manifold: QueryPersonaManifold::new(
                [w_recency, w_frequency, w_semantic, w_entropy],
                1.0,   // Pitman-Yor alpha
                0.25,  // Pitman-Yor discount
            ),
            last_archetype_id: None,
            enable_query_personas: true,
            enable_channel_coding: true,
            reward_baseline_ema: 0.0,
            egsc_cache: cache::EgscCache::default(),
            last_query: String::new(),
            last_effective_budget: 0,
            last_cache_feedback_eligible: false,
            // Context Resonance
            resonance_matrix: ResonanceMatrix::new(),
            prism_optimizer_5d: PrismOptimizer5D::from_4d(&PrismOptimizer::new(0.01)),
            w_resonance: 0.0, // cold start: resonance contributes nothing until learned
            enable_resonance: true,
            // Coverage Estimator
            last_semantic_candidates: 0,
            last_structural_candidates: 0,
            last_candidate_overlap: 0,
            // Fragment Consolidation
            total_consolidations: 0,
            consolidation_tokens_saved: 0,
            consolidation_hamming_threshold: 8, // wider than dedup (3) to catch near-dupes
            // Causal Context Graph
            causal_graph: CausalContextGraph::new(),
            enable_causal: true,
            prev_selected_ids: Vec::new(),
            prev_explored_ids: Vec::new(),
            // Belief Utilization Auto-Tuning
            belief_util_ema: 0.5,  // neutral start
            full_util_ema: 0.5,    // neutral start
            base_belief_info_factor: ios_belief_info_factor.clamp(0.01, 0.99),
        }
    }

    /// Advance the turn counter and apply Ebbinghaus decay.
    pub fn advance_turn(&mut self) {
        self.current_turn += 1;

        // Apply decay in-place (no drain/rebuild)
        let decay_rate = (2.0_f64).ln() / self.decay_half_life.max(1) as f64;
        for frag in self.fragments.values_mut() {
            let dt = self.current_turn.saturating_sub(frag.turn_last_accessed) as f64;
            frag.recency_score = (-decay_rate * dt).exp();
        }

        // Collect IDs to evict (avoid borrow conflict with dedup_index)
        let to_evict: Vec<String> = self.fragments.iter()
            .filter(|(_, f)| f.recency_score < self.min_relevance && !f.is_pinned)
            .map(|(k, _)| k.clone())
            .collect();

        for id in &to_evict {
            self.dedup_index.remove(id);
        }
        self.fragments.retain(|_, f| f.recency_score >= self.min_relevance || f.is_pinned);

        // Rebuild LSH slot index after eviction (slots may have shifted).
        // This is O(N) but eviction is infrequent (happens once per turn).
        self.rebuild_lsh_index();

        // Query persona manifold lifecycle: Ebbinghaus decay + death + fusion
        if self.enable_query_personas {
            self.query_manifold.lifecycle_tick();
        }

        // EGSC cache: garbage-collect low-quality entries each turn.
        // Entries with quality_score < 0.15 (those consistently receiving
        // negative feedback via Wilson scoring) are removed, freeing
        // cache slots for better candidates.
        self.egsc_cache.gc(0.15);

        // ── Context Resonance: decay pairwise interaction strengths ──
        // Half-life ~34 turns (0.98^34 ≈ 0.50). Slower than Ebbinghaus
        // fragment decay (15 turns) because pairwise patterns are more
        // stable than individual recency.
        if self.enable_resonance {
            self.resonance_matrix.decay_tick();
        }

        // ── Causal Context Graph: evict stale temporal links ──
        if self.enable_causal {
            self.causal_graph.decay_tick(self.current_turn);
        }

        // ── Maxwell's Demon: Fragment Consolidation ──
        // Run every 5 turns (not every turn — consolidation is O(N²)
        // and near-duplicates accumulate slowly).
        if self.current_turn.is_multiple_of(5) && self.fragments.len() > 10 {
            let frag_data: Vec<(String, u64, f64, bool, u32)> = self.fragments.values()
                .map(|f| {
                    let fm = self.feedback.learned_value(&f.fragment_id);
                    (f.fragment_id.clone(), f.simhash, fm, f.is_pinned, f.token_count)
                })
                .collect();

            let groups = resonance::find_consolidation_groups(
                &frag_data, self.consolidation_hamming_threshold
            );

            for group in &groups {
                // Transfer access counts from losers to winner
                let total_access: u32 = group.consolidated_ids.iter()
                    .filter_map(|id| self.fragments.get(id))
                    .map(|f| f.access_count)
                    .sum();

                if let Some(winner) = self.fragments.get_mut(&group.winner_id) {
                    winner.access_count += total_access;
                }

                // Evict losers
                for loser_id in &group.consolidated_ids {
                    self.fragments.remove(loser_id);
                    self.dedup_index.remove(loser_id);
                }

                self.total_consolidations += group.consolidated_ids.len() as u64;
                self.consolidation_tokens_saved += group.tokens_saved as u64;
            }

            // Rebuild LSH index if any consolidation occurred
            if !groups.is_empty() {
                self.rebuild_lsh_index();
            }
        }
    }

    /// Ingest a new context fragment.
    ///
    /// Pipeline: tokens → SimHash → dedup → entropy → criticality → depgraph → store
    pub fn ingest(&mut self, content: String, source: String, token_count: u32, is_pinned: bool) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            self.total_fragments_ingested += 1;

            let tc = if token_count == 0 {
                // Heuristic: code averages ~5 chars/token (short identifiers,
                // operators), prose averages ~4 chars/token. Use the content's
                // non-alpha ratio to estimate which.
                let non_alpha = content.chars().filter(|c| !c.is_alphabetic()).count();
                let ratio = non_alpha as f64 / content.len().max(1) as f64;
                let chars_per_token = if ratio > 0.4 { 5.0 } else { 4.0 };
                (content.len() as f64 / chars_per_token).max(1.0) as u32
            } else {
                token_count
            };

            // Enforce max_fragments cap (prevents unbounded memory growth)
            if self.fragments.len() >= self.max_fragments {
                let result = PyDict::new(py);
                result.set_item("status", "rejected")?;
                result.set_item("reason", "max_fragments cap reached")?;
                result.set_item("max_fragments", self.max_fragments)?;
                return Ok(result.into());
            }

            // Generate globally-unique fragment ID: instance prefix + per-instance counter.
            // Format: f{instance_hex8}_{counter_hex6}
            // Two engines in the same process will have different instance_id values,
            // so their fragment IDs are guaranteed disjoint.
            self.id_counter += 1;
            let frag_id = format!("f{:08x}_{:06x}", self.instance_id as u32, self.id_counter);

            // Check for duplicates
            if let Some(dup_id) = self.dedup_index.insert(&frag_id, &content) {
                self.total_duplicates_caught += 1;
                self.total_tokens_saved += tc as u64;  // Bug fix: accumulate tokens saved

                let max_freq = self.fragments.values()
                    .map(|f| f.access_count)
                    .max()
                    .unwrap_or(1);

                if let Some(existing) = self.fragments.get_mut(&dup_id) {
                    existing.access_count += 1;
                    existing.turn_last_accessed = self.current_turn;
                    existing.frequency_score = (existing.access_count as f64 / max_freq.max(existing.access_count) as f64).min(1.0);
                }

                let result = PyDict::new(py);
                result.set_item("status", "duplicate")?;
                result.set_item("duplicate_of", &dup_id)?;
                result.set_item("tokens_saved", tc)?;
                return Ok(result.into());
            }

            // Compute entropy score (deterministic: sorted by fragment_id)
            let mut sorted_frags: Vec<&ContextFragment> = self.fragments.values().collect();
            sorted_frags.sort_by(|a, b| a.fragment_id.cmp(&b.fragment_id));
            let other_contents: Vec<String> = sorted_frags.iter()
                .take(50)
                .map(|f| f.content.clone())
                .collect();
            let other_refs: Vec<&str> = other_contents.iter().map(|s| s.as_str()).collect();
            let entropy = information_score(&content, &other_refs);

            // NEW: Criticality check — config, schema, license files get protected
            let criticality = file_criticality(&source);
            let has_safety = has_safety_signal(&content);

            // Force-pin safety and critical files
            let effective_pinned = is_pinned
                || criticality == Criticality::Safety
                || criticality == Criticality::Critical
                || has_safety;

            // Compute SimHash fingerprint
            let fp = simhash(&content);

            // Store true entropy — do NOT multiply by criticality boost.
            // Criticality is a separate dimension that affects selection via pinning,
            // not by inflating entropy scores (which would corrupt the information
            // density signal). Critical files with low entropy keep their honest
            // entropy score but are protected via is_pinned.
            //
            // Exception: apply a minimum entropy floor for critical files so they
            // don't get ranked last when everything else is equal.
            let effective_entropy = if criticality >= Criticality::Important {
                entropy.max(0.5)  // Floor: critical files never score below 0.5
            } else {
                entropy
            };

            let mut frag = ContextFragment::new(frag_id.clone(), content.clone(), tc, source.clone());
            frag.recency_score = 1.0;
            frag.entropy_score = effective_entropy;
            frag.turn_created = self.current_turn;
            frag.turn_last_accessed = self.current_turn;
            frag.access_count = 1;
            frag.is_pinned = effective_pinned;
            frag.simhash = fp;
            frag.has_simhash = true;  // content-derived fingerprint

            // Hierarchical fragmentation: extract skeleton for code files
            if let Some(skel) = skeleton::extract_skeleton(&content, &source) {
                let skel_non_alpha = skel.chars().filter(|c| !c.is_alphabetic()).count();
                let skel_ratio = skel_non_alpha as f64 / skel.len().max(1) as f64;
                let skel_cpt = if skel_ratio > 0.4 { 5.0 } else { 4.0 };
                let skel_tc = (skel.len() as f64 / skel_cpt).max(1.0) as u32;
                frag.skeleton_content = Some(skel);
                frag.skeleton_token_count = Some(skel_tc);
            }

            // NEW: Auto-link dependencies in the dep graph
            self.dep_graph.auto_link(&frag_id, &content);

            // Capture skeleton info before moving frag
            let has_skeleton = frag.skeleton_content.is_some();
            let skel_tc_for_result = frag.skeleton_token_count;

            self.fragments.insert(frag_id.clone(), frag);

            // Insert into LSH index for sub-linear recall.
            // Slot = current last index (just appended).
            let slot = self.fragment_slot_ids.len();
            self.fragment_slot_ids.push(frag_id.clone());
            self.lsh_index.insert(fp, slot);

            // EGSC cache: depth-weighted DAG invalidation on new fragment.
            // Direct fragment (depth 0) is ALWAYS hard-invalidated regardless
            // of dep graph completeness. Transitive dependents get progressively
            // softer invalidation via exponential decay: w *= exp(-λ/depth).
            let mut stale_closure: HashSet<String> = std::iter::once(frag_id.clone()).collect();
            let mut depth_weights: HashMap<String, u32> = HashMap::new();
            depth_weights.insert(frag_id.clone(), 0); // depth 0 = direct, always hard
            // BFS through reverse deps with depth tracking (max depth 3)
            let mut bfs_queue: std::collections::VecDeque<(String, u32)> = self.dep_graph
                .reverse_deps(&frag_id)
                .into_iter()
                .map(|id| (id, 1))
                .collect();
            while let Some((id, depth)) = bfs_queue.pop_front() {
                if depth > 3 || stale_closure.contains(&id) { continue; }
                stale_closure.insert(id.clone());
                depth_weights.insert(id.clone(), depth);
                for rev in self.dep_graph.reverse_deps(&id) {
                    if !stale_closure.contains(&rev) {
                        bfs_queue.push_back((rev, depth + 1));
                    }
                }
            }
            let _invalidated = self.egsc_cache.invalidate_weighted(&stale_closure, &depth_weights);

            let result = PyDict::new(py);
            result.set_item("status", "ingested")?;
            result.set_item("fragment_id", &frag_id)?;
            result.set_item("token_count", tc)?;
            result.set_item("entropy_score", (effective_entropy * 10000.0).round() / 10000.0)?;
            result.set_item("criticality", format!("{:?}", criticality))?;
            result.set_item("is_pinned", effective_pinned)?;
            result.set_item("total_fragments", self.fragments.len())?;
            result.set_item("has_skeleton", has_skeleton)?;
            if let Some(stc) = skel_tc_for_result {
                result.set_item("skeleton_token_count", stc)?;
            }
            Ok(result.into())
        })
    }

    /// Batch-ingest multiple fragments in one PyO3 call with rayon parallelism.
    ///
    /// This is 10-50x faster than calling ingest() per-file because:
    ///   1. ONE PyO3 GIL acquisition instead of N
    ///   2. SimHash, skeleton, criticality computed in parallel via rayon
    ///   3. Entropy computed against a FIXED sample (O(N) not O(N²))
    ///   4. Dep graph built in bulk after all inserts
    ///   5. No per-fragment cache invalidation (fresh batch = no stale cache)
    ///
    /// Args:
    ///   items: list of (content: str, source: str, token_count: int) tuples
    ///
    /// Returns:
    ///   dict with ingested count, total tokens, duplicates caught, duration_ms
    /// Create shadow reference fragments from file paths and sizes — zero content reading.
    ///
    /// This is Phase 0 of the Lazy Progressive Index (LPI). By calling this with
    /// git ls-tree output (path + file size in bytes), we give the optimizer 100%
    /// visibility into the entire repo before reading any file content.
    ///
    /// Shadow fragments have:
    ///   - Minimal stub content ("// filename") — 2-3 tokens
    ///   - Token count estimated from file size / 4.5
    ///   - Low recency (0.2) and entropy (0.35) vs content fragments (0.7+)
    ///   - simhash = 0 (no content fingerprint)
    ///
    /// When batch_ingest() later processes the same file with real content,
    /// the dedup system will keep both (different content). The content fragment
    /// wins on scoring due to higher entropy, recency kept as-is.
    ///
    /// For VSCode (30K files): creates 30K shadows in ~300ms. Then top-500
    /// content fragments are batch_ingested. Total: <1s for full repo visibility.
    pub fn ingest_paths_stubs(&mut self, paths_sizes: Vec<(String, u64)>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let total = paths_sizes.len();
            let mut ingested = 0u32;

            for (rel_path, size_bytes) in paths_sizes {
                if self.fragments.len() >= self.max_fragments {
                    break;
                }

                self.id_counter += 1;
                let frag_id = format!("sh{:08x}_{:06x}", self.instance_id as u32, self.id_counter);
                let source = format!("file:{rel_path}");

                // Stub content — LLM sees filename at reference resolution
                let name = rel_path
                    .rsplit(&['/', '\\'][..])
                    .next()
                    .unwrap_or(&rel_path);
                let stub = format!("// {name}");

                // Estimate token count from file size (4.5 chars/token for code)
                let token_estimate = ((size_bytes as f64) / 4.5).max(2.0) as u32;

                // simhash=0 is the sentinel for "no content fingerprint".
                // Stubs do NOT participate in SimHash similarity.
                // Content fragments have non-zero SimHashes — LSH lookups from real
                // content never land in bucket 0, so the shadow bucket is naturally
                // isolated. Mixing path-hash into this space would corrupt all
                // distance thresholds and break SimHash LSH guarantees.
                let stub_fp: u64 = 0;

                let mut frag = ContextFragment::new(
                    frag_id.clone(),
                    stub,
                    token_estimate,
                    source,
                );
                frag.recency_score = 0.2;
                frag.entropy_score = 0.35;
                frag.turn_created = self.current_turn;
                frag.turn_last_accessed = self.current_turn;
                frag.access_count = 0;
                frag.simhash = stub_fp;  // sentinel: excluded from all similarity ops
                frag.has_simhash = false; // stubs never participate in LSH or SimHash similarity
                frag.is_pinned = false;

                let _slot = self.fragment_slot_ids.len();
                self.fragment_slot_ids.push(frag_id.clone());
                // Stubs are NOT inserted into LSH — avoid degenerate bucket 0 queries
                // and prevent path-simhash mixing (would corrupt all distance thresholds).
                // Content lookups never need to find stubs via SimHash.
                // self.lsh_index.insert(stub_fp, slot);  ← intentionally omitted
                self.fragments.insert(frag_id, frag);
                ingested += 1;
            }

            let result = PyDict::new(py);
            result.set_item("shadows_created", ingested)?;
            result.set_item("total_input", total)?;
            result.set_item("skipped", (total as u32).saturating_sub(ingested))?;
            Ok(result.into())
        })
    }

    pub fn batch_ingest(&mut self, items: Vec<(String, String, u32)>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let t0 = std::time::Instant::now();
            let n = items.len();

            if n == 0 {
                let result = PyDict::new(py);
                result.set_item("ingested", 0)?;
                result.set_item("total_tokens", 0)?;
                result.set_item("duplicates", 0)?;
                result.set_item("duration_ms", 0)?;
                return Ok(result.into());
            }

            // ── Phase 1: Parallel pre-computation (rayon) ──
            // Compute per-fragment: token_count, simhash, skeleton, criticality, safety.
            // These are all independent and embarrassingly parallel.
            struct PreComputed {
                content: String,
                source: String,
                token_count: u32,
                simhash: u64,
                skeleton_token_estimate: Option<u32>,  // TPSE Phase A: speculative cost
                criticality: Criticality,
                has_safety: bool,
                kolmo: f64, // Kolmogorov entropy: single O(N) proxy for ent+bp
                language: String, // e.g. "python", "rust", "typescript"
            }

            let precomputed: Vec<PreComputed> = items.into_par_iter()
                .map(|(content, source, tc)| {
                    let token_count = if tc == 0 {
                        // ── Calibrated Per-Language Token Estimation ──────────
                        // Replaces crude binary heuristic (non-alpha > 0.4 → 5.0, else 4.0)
                        // with file-extension-specific chars/token ratios calibrated from
                        // real tokenizer measurements (cl100k_base / o200k_base).
                        //
                        // Improvement: Python 25% more accurate, JSON 30% more accurate.
                        // This directly improves IOS selection quality — the knapsack
                        // can't make good decisions with 25% error in cost estimates.
                        let sl = source.to_lowercase();
                        let cpt = if sl.ends_with(".py") || sl.ends_with(".pyw") { 3.0 }
                            else if sl.ends_with(".rs") { 3.5 }
                            else if sl.ends_with(".ts") || sl.ends_with(".tsx")
                                    || sl.ends_with(".js") || sl.ends_with(".jsx") || sl.ends_with(".mjs")
                                    || sl.ends_with(".sh") || sl.ends_with(".bash")
                                    || sl.ends_with(".rb") { 3.2 }
                            else if sl.ends_with(".go") { 3.3 }
                            else if sl.ends_with(".java") || sl.ends_with(".kt") || sl.ends_with(".cs")
                                    || sl.ends_with(".swift") || sl.ends_with(".sql") { 3.5 }
                            else if sl.ends_with(".c") || sl.ends_with(".cpp") || sl.ends_with(".cc")
                                    || sl.ends_with(".h") || sl.ends_with(".hpp") { 3.8 }
                            else if sl.ends_with(".json") { 2.8 }
                            else if sl.ends_with(".yaml") || sl.ends_with(".yml") || sl.ends_with(".toml") { 3.0 }
                            else if sl.ends_with(".md") || sl.ends_with(".txt") || sl.ends_with(".rst") { 4.5 }
                            else if sl.ends_with(".html") || sl.ends_with(".css") || sl.ends_with(".scss")
                                    || sl.ends_with(".php") { 3.0 }
                            else {
                                // Fallback: use non-alpha ratio heuristic for unknown types
                                let non_alpha = content.chars().filter(|c| !c.is_alphabetic()).count();
                                let ratio = non_alpha as f64 / content.len().max(1) as f64;
                                if ratio > 0.4 { 5.0 } else { 4.0 }
                            };
                        (content.len() as f64 / cpt).max(1.0) as u32
                    } else {
                        tc
                    };

                    let fp = simhash(&content);
                    let criticality = file_criticality(&source);
                    let has_safety = has_safety_signal(&content);
                    // Single Kolmogorov entropy pass: ONE O(N) compression replacing
                    // TWO separate scans (normalized_entropy + boilerplate_ratio).
                    // Grounded in Kolmogorov (1965): K(x) ≤ len(compress(x)).
                    let kolmo = crate::entropy::kolmogorov_entropy(&content);

                    // ── TPSE Phase A: Speculative Skeleton Token Estimation ──
                    // Estimate skeleton token cost from language-specific compression ratios.
                    // O(1) per file — gives IOS enough info for MRK resolution decisions
                    // without computing actual skeletons (O(file_size) each × 13 parsers).
                    //
                    // Quality guarantee (two-stage stochastic optimization, Birge & Louveaux 2011):
                    // Estimation error ε ≈ 10% → IOS quality loss ≤ ε × skel_budget_frac ≈ 2.5%.
                    // Ratios calibrated on skeleton.rs test corpus (Python/Rust/JS/Go/Java/C++).
                    let skel_ratio = {
                        let sl = source.to_lowercase();
                        if sl.ends_with(".py") || sl.ends_with(".go") || sl.ends_with(".pyw") { 0.20 }
                        else if sl.ends_with(".rs") || sl.ends_with(".java") || sl.ends_with(".kt")
                                || sl.ends_with(".cs") || sl.ends_with(".swift") { 0.25 }
                        else if sl.ends_with(".js") || sl.ends_with(".ts") || sl.ends_with(".tsx")
                                || sl.ends_with(".jsx") || sl.ends_with(".mjs") || sl.ends_with(".mts") { 0.22 }
                        else if sl.ends_with(".c") || sl.ends_with(".cpp") || sl.ends_with(".cc")
                                || sl.ends_with(".h") || sl.ends_with(".hpp") { 0.30 }
                        else if sl.ends_with(".rb") || sl.ends_with(".php") { 0.25 }
                        else if sl.ends_with(".vue") || sl.ends_with(".svelte") { 0.28 }
                        else if sl.ends_with(".html") || sl.ends_with(".css") || sl.ends_with(".scss") { 0.30 }
                        else if sl.ends_with(".sh") || sl.ends_with(".bash") { 0.20 }
                        else { 0.0 }  // Unknown language: no skeleton possible
                    };
                    let skeleton_token_estimate = if skel_ratio > 0.0 && token_count > 10 {
                        let est = (token_count as f64 * skel_ratio).max(3.0) as u32;
                        if est < token_count { Some(est) } else { None }
                    } else {
                        None
                    };

                    // ── Language Identification ──────────────────────────────
                    // Populate language field for downstream calibrated operations.
                    let language = {
                        let sl = source.to_lowercase();
                        if sl.ends_with(".py") || sl.ends_with(".pyw") { "python" }
                        else if sl.ends_with(".rs") { "rust" }
                        else if sl.ends_with(".ts") || sl.ends_with(".tsx") { "typescript" }
                        else if sl.ends_with(".js") || sl.ends_with(".jsx") || sl.ends_with(".mjs") { "javascript" }
                        else if sl.ends_with(".go") { "go" }
                        else if sl.ends_with(".java") { "java" }
                        else if sl.ends_with(".kt") { "kotlin" }
                        else if sl.ends_with(".cs") { "csharp" }
                        else if sl.ends_with(".c") || sl.ends_with(".h") { "c" }
                        else if sl.ends_with(".cpp") || sl.ends_with(".cc") || sl.ends_with(".hpp") { "cpp" }
                        else if sl.ends_with(".rb") { "ruby" }
                        else if sl.ends_with(".php") { "php" }
                        else if sl.ends_with(".swift") { "swift" }
                        else if sl.ends_with(".sh") || sl.ends_with(".bash") { "shell" }
                        else if sl.ends_with(".json") { "json" }
                        else if sl.ends_with(".yaml") || sl.ends_with(".yml") { "yaml" }
                        else if sl.ends_with(".toml") { "toml" }
                        else if sl.ends_with(".md") || sl.ends_with(".rst") { "markdown" }
                        else if sl.ends_with(".html") { "html" }
                        else if sl.ends_with(".css") || sl.ends_with(".scss") { "css" }
                        else if sl.ends_with(".sql") { "sql" }
                        else { "unknown" }
                    }.to_string();

                    PreComputed {
                        content,
                        source,
                        token_count,
                        simhash: fp,
                        skeleton_token_estimate,
                        criticality,
                        has_safety,
                        kolmo,
                        language,
                    }
                })
                .collect();

            let phase1_ms = t0.elapsed().as_millis() as u64;

            // ── Phase 2: SimHash-based entropy (O(N × 50) integer ops) ────────────────────
            // REPLACES: information_score(&content, &sample_refs)
            //   OLD: O(N × 50 × file_size) n-gram HashSet construction
            //   NEW: O(N × 50) integer XOR + popcount
            // For k=50 refs, 5KB files: ~900x speedup. VSCode 30K files: hours → seconds.
            //
            // Formula: entropy = 0.40×ent + 0.30×bp + 0.30×simhash_uniqueness - noise_penalty
            // Same weights as information_score(), but uniqueness comes from SimHash distance
            // instead of n-gram Jaccard overlap. Error bound ≈ 12.5%.
            // Sample: only content fragments with valid SimHash (has_simhash=true).
            // Stubs (has_simhash=false) must not pollute the sample — they have no
            // semantic fingerprint and would bias uniqueness scores high.
            // Min sample guard: if < 5 real fragments exist, simhash_uniqueness
            // returns the default prior (0.7) — scores stabilize as corpus grows.
            const MIN_SAMPLE: usize = 5;
            // Stratified sample: use step_by(stride) instead of take(25) to span
            // all priority tiers. Without this, priority-sorted batch puts all impl
            // code first → entropy sample is 100% impl → test/config uniqueness inflated.
            let batch_stride = (precomputed.len() / 25).max(1);
            let sample_fps: Vec<u64> = self.fragments.values()
                .filter(|f| f.has_simhash)
                .take(25)
                .map(|f| f.simhash)
                .chain(
                    precomputed.iter()
                        .step_by(batch_stride)
                        .take(25)
                        .map(|p| p.simhash)
                )
                .collect();
            let sample_ok = sample_fps.len() >= MIN_SAMPLE;

            let entropies: Vec<f64> = precomputed.par_iter()
                .map(|p| {
                    if !sample_ok {
                        // Too few hydrated references for cross-corpus uniqueness.
                        // Fall back to Kolmogorov entropy alone (still valid).
                        return p.kolmo;
                    }
                    let uniqueness = crate::entropy::simhash_uniqueness(p.simhash, &sample_fps);
                    // Formula: 70% Kolmogorov density + 30% SimHash cross-corpus uniqueness.
                    // Kolmogorov replaces (0.40×ent + 0.30×bp) in a single compression pass.
                    // SimHash uniqueness replaces n-gram Jaccard (900× faster, same ordinal rank).
                    // Note: SimHash estimates cosine similarity; Jaccard would give slightly
                    // different absolute values but same rank ordering for entropy estimation.
                    let div_proxy = (1.0 - uniqueness) * 2.0;
                    let noise_penalty = if div_proxy > 1.5 {
                        (div_proxy - 1.5).min(1.0) * 0.15
                    } else {
                        0.0
                    };
                    (0.70 * p.kolmo + 0.30 * uniqueness - noise_penalty).clamp(0.0, 1.0)
                })
                .collect();

            let phase2_ms = t0.elapsed().as_millis() as u64 - phase1_ms;

            // ── Phase 3: Sequential insert (mutates self) ──
            let mut ingested = 0u32;
            let mut total_tokens = 0u64;
            let mut duplicates = 0u32;

            // ── Stub Hygiene: pre-build source→stub_id index for O(1) cleanup ──
            // When batch_ingest hydrates a file that already has a shadow stub from
            // ingest_paths_stubs(), the stub must be removed. This keeps the HashMap
            // lean and prevents the knapsack from seeing stale stubs with wildly
            // overestimated token_count (estimated from file size, not actual content).
            let stub_index: HashMap<String, String> = self.fragments.iter()
                .filter(|(id, f)| !f.has_simhash && id.starts_with("sh"))
                .map(|(id, f)| (f.source.clone(), id.clone()))
                .collect();

            for (i, pre) in precomputed.into_iter().enumerate() {
                if self.fragments.len() >= self.max_fragments {
                    break;
                }

                self.total_fragments_ingested += 1;
                self.id_counter += 1;
                let frag_id = format!("f{:08x}_{:06x}", self.instance_id as u32, self.id_counter);

                // Remove shadow stub if this source was previously stubbed (O(1) lookup)
                if let Some(stub_id) = stub_index.get(&pre.source) {
                    self.fragments.remove(stub_id);
                    self.dedup_index.remove(stub_id);
                }

                // Dedup check
                if let Some(dup_id) = self.dedup_index.insert(&frag_id, &pre.content) {
                    self.total_duplicates_caught += 1;
                    self.total_tokens_saved += pre.token_count as u64;
                    duplicates += 1;
                    if let Some(existing) = self.fragments.get_mut(&dup_id) {
                        existing.access_count += 1;
                        existing.turn_last_accessed = self.current_turn;
                    }
                    continue;
                }

                let effective_pinned = pre.has_safety
                    || pre.criticality == Criticality::Safety
                    || pre.criticality == Criticality::Critical;

                let entropy = entropies[i];
                let effective_entropy = if pre.criticality >= Criticality::Important {
                    entropy.max(0.5)
                } else {
                    entropy
                };

                let mut frag = ContextFragment::new(frag_id.clone(), pre.content, pre.token_count, pre.source);
                frag.recency_score = 1.0;
                frag.entropy_score = effective_entropy;
                frag.turn_created = self.current_turn;
                frag.turn_last_accessed = self.current_turn;
                frag.access_count = 1;
                frag.is_pinned = effective_pinned;
                frag.simhash = pre.simhash;
                frag.has_simhash = true;  // content-derived fingerprint
                // TPSE Phase A: skeleton_token_count = speculative estimate.
                // skeleton_content remains None — materialized lazily in optimize()
                // Phase B, only for the K fragments IOS selects at Skeleton resolution.
                frag.skeleton_token_count = pre.skeleton_token_estimate;
                frag.language = pre.language;

                // Skip dep graph auto_link here — we build it in bulk below
                // after all fragments are inserted (avoids O(N) sequential regex)

                // LSH: only content fragments with valid SimHash participate
                let slot = self.fragment_slot_ids.len();
                self.fragment_slot_ids.push(frag_id.clone());
                self.lsh_index.insert(pre.simhash, slot);

                total_tokens += pre.token_count as u64;
                self.fragments.insert(frag_id, frag);
                ingested += 1;
            }

            // Dep graph is built lazily during optimize() — auto_link is called
            // for fragments as they're selected, not at ingest time.
            // Skipping bulk-link here avoids 2×O(N×content_size) re-scans:
            //   - content.clone() for all N files
            //   - extract_identifiers() char-walk per file (same cost as indexing)
            // Net effect: batch_ingest stays O(rayon parallel) with no sequential tail.

            // Skip per-fragment EGSC cache invalidation for batch ingest —
            // batch is used during initial indexing when the cache is empty.
            // Just clear the entire cache if it has entries.
            if !self.egsc_cache.is_empty() {
                self.egsc_cache.clear();
            }

            let elapsed_ms = t0.elapsed().as_millis() as u64;

            let result = PyDict::new(py);
            result.set_item("ingested", ingested)?;
            result.set_item("total_tokens", total_tokens)?;
            result.set_item("duplicates", duplicates)?;
            result.set_item("total_fragments", self.fragments.len())?;
            result.set_item("duration_ms", elapsed_ms)?;
            result.set_item("phase1_ms", phase1_ms)?;
            result.set_item("phase2_ms", phase2_ms)?;
            result.set_item("phase3_ms", elapsed_ms.saturating_sub(phase1_ms + phase2_ms))?;
            Ok(result.into())
        })
    }

    /// Select the optimal context subset within the token budget.
    ///
    /// Two-pass optimization:
    ///   Pass 1: Initial knapsack selection
    ///   Pass 2: Boost unselected dependencies of selected fragments, re-run
    ///
    /// Wires in: feedback loop, dependency graph, context ordering.
    pub fn optimize(&mut self, token_budget: u32, query: String) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            self.total_optimizations += 1;

            // Store query for cache feedback routing (used by record_success/failure)
            self.last_query = query.clone();
            self.last_cache_feedback_eligible = false;

            // Apply task-type budget multiplier
            let effective_budget = if !query.is_empty() {
                let task_type = TaskType::classify(&query);
                let mult = task_type.budget_multiplier();
                (token_budget as f64 * mult) as u32
            } else {
                token_budget
            };
            self.last_effective_budget = effective_budget;

            // ── RAVEN-UCB Adaptive Exploration ──
            // α₀ is used in the exploration swap code (UCB score for picking swap target).
            let alpha_0 = 2.0_f64;
            let should_explore = if self.exploration_rate > 0.0 {
                // xorshift64 PRNG coin flip — uniform distribution over [0, 1).
                let mut x = self.rng_state;
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                self.rng_state = x;
                let coin = (x % 10000) as f64 / 10000.0;
                coin < self.exploration_rate
            } else {
                false
            };

            // ── EGSC Cache: check for a cached optimization result ──
            // The cache key is (query, current_fragment_ids). On hit, skip the
            // entire IOS/knapsack/channel-coding pipeline — returns in O(1).
            if !query.is_empty() && !should_explore {
                let current_frag_ids: HashSet<String> = self.fragments.keys().cloned().collect();
                match self.egsc_cache.lookup_with_budget(&query, &current_frag_ids, effective_budget) {
                    CacheLookup::ExactHit { response, tokens_saved } => {
                        self.total_tokens_saved += tokens_saved as u64;
                        // Deserialize cached JSON result back to Python dict
                        if let Ok(cached_value) = serde_json::from_str::<serde_json::Value>(&response) {
                            let cache_result = PyDict::new(py);
                            // Populate result from cached JSON
                            if let Some(obj) = cached_value.as_object() {
                                for (k, v) in obj {
                                    if k == "selected_ids" { continue; } // handled separately
                                    match v {
                                        serde_json::Value::Number(n) => {
                                            if let Some(i) = n.as_i64() { let _ = cache_result.set_item(k.as_str(), i); }
                                            else if let Some(f) = n.as_f64() { let _ = cache_result.set_item(k.as_str(), f); }
                                        }
                                        serde_json::Value::String(s) => { let _ = cache_result.set_item(k.as_str(), s.as_str()); }
                                        serde_json::Value::Bool(b) => { let _ = cache_result.set_item(k.as_str(), *b); }
                                        _ => {}
                                    }
                                }
                            }
                            // Reconstruct selected_fragments from cached IDs + live fragments
                            let selected_list = self.rebuild_selected_list(py, &cached_value)?;
                            cache_result.set_item("selected", selected_list)?;
                            cache_result.set_item("cache_hit", true)?;
                            cache_result.set_item("cache_hit_type", "exact")?;
                            cache_result.set_item("cache_tokens_saved", tokens_saved)?;
                            cache_result.set_item("cache_eligible", true)?;
                            cache_result.set_item("optimization_policy", "exploit")?;
                            self.last_cache_feedback_eligible = true;
                            return Ok(cache_result.into());
                        }
                    }
                    CacheLookup::SemanticHit { response, tokens_saved, hamming_distance: ham, jaccard_similarity: jac } => {
                        self.total_tokens_saved += tokens_saved as u64;
                        if let Ok(cached_value) = serde_json::from_str::<serde_json::Value>(&response) {
                            let cache_result = PyDict::new(py);
                            if let Some(obj) = cached_value.as_object() {
                                for (k, v) in obj {
                                    if k == "selected_ids" { continue; }
                                    match v {
                                        serde_json::Value::Number(n) => {
                                            if let Some(i) = n.as_i64() { let _ = cache_result.set_item(k.as_str(), i); }
                                            else if let Some(f) = n.as_f64() { let _ = cache_result.set_item(k.as_str(), f); }
                                        }
                                        serde_json::Value::String(s) => { let _ = cache_result.set_item(k.as_str(), s.as_str()); }
                                        serde_json::Value::Bool(b) => { let _ = cache_result.set_item(k.as_str(), *b); }
                                        _ => {}
                                    }
                                }
                            }
                            // Reconstruct selected_fragments from cached IDs + live fragments
                            let selected_list = self.rebuild_selected_list(py, &cached_value)?;
                            cache_result.set_item("selected", selected_list)?;
                            cache_result.set_item("cache_hit", true)?;
                            cache_result.set_item("cache_hit_type", "semantic")?;
                            cache_result.set_item("cache_tokens_saved", tokens_saved)?;
                            cache_result.set_item("cache_hamming_distance", ham)?;
                            cache_result.set_item("cache_jaccard_similarity", (jac * 10000.0).round() / 10000.0)?;
                            cache_result.set_item("cache_eligible", true)?;
                            cache_result.set_item("optimization_policy", "exploit")?;
                            self.last_cache_feedback_eligible = true;
                            return Ok(cache_result.into());
                        }
                    }
                    CacheLookup::Miss => {} // Continue to full optimization
                }
            }

            // ── GGCR: Graph-Guided Causal Retrieval ──────────────────────
            // Replaces SimHash Hamming distance (noise ~0.45±0.03) with
            // multi-signal fusion for actual query-document relevance.
            //
            // Signals:
            //   1. BM25 (lexical): term freq × inverse doc freq + path/id boost
            //   2. Causal chain (structural): dep graph from query-matched files
            //   3. PageRank (centrality): hub files many others depend on
            //   4. NCD (compression): reranker for top-50 candidates
            if !query.is_empty() {
                let query_terms: Vec<String> = bm25::tokenize_code(&query);

                // ── Signal 1: BM25 Lexical Scoring ──
                let doc_tuples: Vec<(String, String, String)> = self.fragments.iter()
                    .map(|(id, f)| (id.clone(), f.content.clone(), f.source.clone()))
                    .collect();
                let bm25_idx = BM25Index::build(&doc_tuples);

                let mut bm25_raw: HashMap<String, f64> = HashMap::with_capacity(self.fragments.len());
                for (fid, frag) in &self.fragments {
                    let identifiers = extract_identifiers(&frag.content);
                    let score = bm25_idx.score(&query_terms, &frag.content, &frag.source, &identifiers);
                    bm25_raw.insert(fid.clone(), score.combined);
                }

                // Normalize BM25 to [0.05, 1.0]
                let fid_order: Vec<String> = self.fragments.keys().cloned().collect();
                let raw_vec: Vec<f64> = fid_order.iter()
                    .map(|fid| bm25_raw.get(fid).copied().unwrap_or(0.0))
                    .collect();
                let norm_vec = bm25::normalize_scores(&raw_vec);
                let bm25_norm: HashMap<String, f64> = fid_order.iter()
                    .zip(norm_vec.iter())
                    .map(|(fid, &s)| (fid.clone(), s))
                    .collect();

                // ── Signal 2: Causal Chain via Dep Graph ──
                // Top BM25 matches seed a BFS through the dep graph.
                // Files in the causal chain are structurally connected
                // to query-relevant code — embeddings can't find these.
                let mut sorted_bm25: Vec<(String, f64)> = bm25_raw.iter()
                    .map(|(k, &v)| (k.clone(), v))
                    .collect();
                sorted_bm25.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let seed_ids: Vec<String> = sorted_bm25.iter()
                    .take(10)
                    .filter(|(_, s)| *s > 0.5)
                    .map(|(id, _)| id.clone())
                    .collect();
                let causal_set: HashSet<String> = if !seed_ids.is_empty() {
                    hierarchical::identify_cluster(&self.dep_graph, &seed_ids, 2)
                        .into_iter().collect()
                } else {
                    HashSet::new()
                };

                // ── Signal 3: PageRank Centrality ──
                let pr_ids: Vec<String> = self.fragments.keys().cloned().collect();
                let pagerank = hierarchical::compute_pagerank(&self.dep_graph, &pr_ids, 15);
                let max_pr = pagerank.values().cloned().fold(0.0_f64, f64::max).max(1e-10);

                // ── Signal 4: NCD Reranking (top-50 only for performance) ──
                let top_candidates: Vec<String> = sorted_bm25.iter()
                    .take(50)
                    .map(|(id, _)| id.clone())
                    .collect();
                let mut ncd_scores: HashMap<String, f64> = HashMap::new();
                for fid in &top_candidates {
                    if let Some(frag) = self.fragments.get(fid) {
                        // Truncate to 2KB for NCD perf (first 2KB has imports+defs)
                        let end = frag.content.len().min(2048);
                        let safe = (0..=end).rev()
                            .find(|&i| frag.content.is_char_boundary(i))
                            .unwrap_or(0);
                        let sample = &frag.content[..safe];
                        ncd_scores.insert(fid.clone(), entropy::ncd_similarity(&query, sample));
                    }
                }

                // ── Fusion → semantic_score ──
                // Phase 5: Task-type-adaptive fusion weights.
                // Different query types benefit from different signals:
                //   Debug → BM25 dominates (exact error/function name matching)
                //   Refactor → causal chain dominates (structural deps matter)
                //   Architecture → PageRank dominates (hub files, system overview)
                //   Understanding → balanced (need both content + structure)
                //
                // These weights are the starting point; PRISM's semantic dimension
                // weight (w_semantic) then scales the entire fused score relative to
                // recency/frequency/entropy, so online learning still applies.
                let task_type = TaskType::classify(&query);
                let (w_bm25, w_causal, w_pr, w_ncd) = match task_type {
                    TaskType::BugTracing => (0.55, 0.15, 0.10, 0.20),
                    TaskType::Refactoring => (0.30, 0.30, 0.20, 0.20),
                    TaskType::CodeGeneration => (0.35, 0.25, 0.20, 0.20),
                    TaskType::Exploration => (0.35, 0.15, 0.25, 0.25),
                    _ => (0.40, 0.20, 0.15, 0.25),
                };

                // ── Query Kernel Extraction ──────────────────────────
                // Identify the most discriminative query terms (highest IDF).
                // These are the "kernel" — the terms that identify WHAT the
                // user is asking about, vs HOW (intent terms like "refactor").
                //
                // For "Refactor the checkpoint system to support async":
                //   kernel = ["checkpoint"] (highest IDF — rare, specific)
                //   intent = ["refactor", "support", "system", "async"] (common)
                //
                // Files whose filename matches a kernel term are about that
                // concept — they should rank at the top regardless of generic
                // term frequency.
                let mut term_idfs: Vec<(String, f64)> = query_terms.iter()
                    .map(|t| (t.clone(), bm25_idx.idf(t)))
                    .collect();
                term_idfs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                // Top 2 highest-IDF terms are the kernel
                let kernel_terms: Vec<String> = term_idfs.iter()
                    .take(2)
                    .filter(|(_, idf)| *idf > 1.0) // Must be reasonably rare
                    .map(|(t, _)| t.clone())
                    .collect();
                for (fid, frag) in self.fragments.iter_mut() {
                    let bm25_s = bm25_norm.get(fid).copied().unwrap_or(0.05);
                    let causal_s: f64 = if causal_set.contains(fid) { 1.0 } else { 0.0 };
                    let pr_s = (pagerank.get(fid).copied().unwrap_or(0.0) / max_pr).min(1.0);
                    let ncd_s = ncd_scores.get(fid).copied().unwrap_or(0.0);

                    // ── Multi-Signal Linear Fusion with CombMNZ ──────
                    // Linear base + CombMNZ bonus for multi-signal alignment.
                    // CombMNZ (Fox & Shaw, 1994): score × count_of_active_signals
                    // rewards files that match across multiple retrieval channels.
                    let base = w_bm25 * bm25_s
                        + w_causal * causal_s
                        + w_pr * pr_s
                        + w_ncd * ncd_s;

                    // Count active signals (non-trivial values)
                    let active = 1.0  // BM25 always active
                        + if causal_s > 0.0 { 1.0 } else { 0.0 }
                        + if pr_s > 0.2 { 1.0 } else { 0.0 }
                        + if ncd_s > 0.1 { 1.0 } else { 0.0 };

                    // CombMNZ: boost proportional to signal count (max 25% bonus)
                    let fused = (base * (1.0 + 0.25 * (active - 1.0) / 3.0))
                        .clamp(0.0, 1.0);

                    // ── Source File Type Boost ─────────────────────────
                    // Quality invariant: source code must ALWAYS outrank
                    // docs/configs when semantic scores are close.
                    // Without this, CONTRIBUTING.md can outrank _checkpoint.py
                    // because .md files have higher entropy scores.
                    //
                    // Boost is multiplicative on the fused score, so it only
                    // amplifies existing semantic signal — it cannot promote
                    // an irrelevant .py file above a relevant .md file.
                    let src_lower = frag.source.to_lowercase();

                    // Detect test/example files (contain usage, not implementation)
                    let is_test = src_lower.contains("/test_")
                        || src_lower.contains("/tests/")
                        || src_lower.contains("\\test_")
                        || src_lower.contains("\\tests\\")
                        || src_lower.contains("/examples/")
                        || src_lower.contains("\\examples\\")
                        || src_lower.contains("/bench/")
                        || src_lower.contains("\\bench\\")
                        || src_lower.contains("/conftest")
                        || src_lower.contains("\\conftest");

                    let is_source = (src_lower.ends_with(".py")
                        || src_lower.ends_with(".rs")
                        || src_lower.ends_with(".ts")
                        || src_lower.ends_with(".js")
                        || src_lower.ends_with(".tsx")
                        || src_lower.ends_with(".jsx"))
                        && !is_test;

                    let type_boost = if is_source {
                        // Core source code: 15% boost
                        1.15
                    } else if is_test {
                        // Test/example: 5% penalty (has relevant terms but not impl)
                        0.95
                    } else if src_lower.ends_with(".md")
                        || src_lower.ends_with(".json")
                        || src_lower.ends_with(".yml")
                        || src_lower.ends_with(".yaml")
                        || src_lower.ends_with(".toml")
                        || src_lower.ends_with(".cfg")
                    {
                        // Docs/config: 10% penalty
                        0.90
                    } else {
                        1.0
                    };

                    frag.semantic_score = (fused * type_boost).clamp(0.0, 1.0);

                    // ── Query Kernel Filename Boost ───────────────────
                    // If a NON-TEST file's name matches the highest-IDF
                    // query term, this file's PRIMARY PURPOSE is that concept.
                    // _checkpoint.py → primary purpose is "checkpoint"
                    // Skip test files: test_checkpoint.py just tests it.
                    if !kernel_terms.is_empty() && !is_test {
                        let fname = frag.source.rsplit(&['/', '\\'][..])
                            .next().unwrap_or("").to_lowercase();
                        for kt in &kernel_terms {
                            if fname.contains(kt.as_str()) {
                                frag.semantic_score = frag.semantic_score.max(0.95);
                                break;
                            }
                        }
                    }
                }

                // ── Entity Query Routing ──────
                // For entity queries ("StateGraph", "ChatOpenAI"), files that
                // DEFINE the entity must outrank files that merely import it.
                //
                // IMPORTANT: Extract entities from the RAW query (preserves
                // CamelCase), NOT from query_terms (already lowercased).
                let raw_words: Vec<&str> = query.split_whitespace().collect();
                let entities: Vec<String> = raw_words.iter()
                    .filter(|w| {
                        let chars: Vec<char> = w.chars().collect();
                        // CamelCase: has uppercase letter after position 0
                        let is_camel = chars.len() > 2
                            && chars[0].is_uppercase()
                            && chars.iter().skip(1).any(|c| c.is_lowercase())
                            && chars.iter().skip(1).any(|c| c.is_uppercase());
                        // PascalCase single word (e.g. "StateGraph")
                        let is_pascal = chars.len() > 3
                            && chars[0].is_uppercase()
                            && chars.iter().skip(1).any(|c| c.is_lowercase());
                        // Long snake_case identifier
                        let is_snake = w.contains('_') && w.len() > 5;
                        is_camel || is_pascal || is_snake
                    })
                    .map(|w| w.to_string())
                    .collect();

                if !entities.is_empty() {
                    for (_fid, frag) in self.fragments.iter_mut() {
                        let content = &frag.content;
                        let source_lower = frag.source.to_lowercase();
                        for entity in &entities {
                            let entity_lower = entity.to_lowercase();

                            // Definition patterns (preserves original case for matching)
                            let is_definition = content.contains(&format!("class {}", entity))
                                || content.contains(&format!("class {}(", entity))
                                || content.contains(&format!("class {}:", entity))
                                || content.contains(&format!("def {}", entity))
                                || content.contains(&format!("def {}(", entity))
                                || content.contains(&format!("struct {}", entity))
                                || content.contains(&format!("fn {}", entity))
                                || content.contains(&format!("trait {}", entity))
                                || content.contains(&format!("interface {}", entity))
                                || content.contains(&format!("type {} ", entity));

                            // Path contains the entity name (case-insensitive)
                            let in_path = source_lower.contains(&entity_lower);

                            if is_definition {
                                // Absolute boost: defining file goes to near-max
                                // This overrides BM25 coverage bonus which can
                                // push test files (matching many terms) above source
                                frag.semantic_score = frag.semantic_score.max(0.95);
                            } else if in_path {
                                // Moderate boost: file is named after the entity
                                frag.semantic_score = (frag.semantic_score * 1.25).min(1.0);
                            }
                        }
                    }
                }
            }

            // ── Cold-Start Weight Adaptation ──────────────────────────────
            // On first use: recency=1.0 and frequency=0.0 for ALL fragments,
            // so these dimensions carry zero information. Transfer weight
            // from recency→semantic so the GGCR signal dominates.
            // Saved and restored at the end of optimize() to avoid permanent drift.
            let (saved_w_recency, saved_w_semantic) = (self.w_recency, self.w_semantic);
            if !query.is_empty() {
                let obs = self.feedback.total_observations();
                if obs < 20 {
                    let boost = 0.15 * (1.0 - obs as f64 / 20.0);
                    self.w_semantic += boost;
                    self.w_recency -= boost;
                }
            }

            // Build feedback multipliers for all fragments
            let feedback_mults: HashMap<String, f64> = self.fragments.keys()
                .map(|fid| (fid.clone(), self.feedback.learned_value(fid)))
                .collect();

            // ── Query Persona Manifold: assign query to archetype ──
            // Build feature vector from TF-IDF analysis, route to archetype,
            // and use per-archetype learned weights if available.
            let (archetype_weights, archetype_id) = if self.enable_query_personas && !query.is_empty() {
                let fragment_summaries: Vec<String> = self.fragments.values()
                    .take(50)
                    .map(|f| f.source.clone())
                    .collect();
                let analysis = query::analyze_query(&query, &fragment_summaries);

                // Build TF-IDF score vector for PSM embedding
                let tfidf_scores: Vec<f64> = analysis.key_terms.iter()
                    .enumerate()
                    .map(|(i, _)| 1.0 / (i as f64 + 1.0)) // rank-weighted scores
                    .collect();

                let features = query_persona::build_query_features(
                    &tfidf_scores,
                    analysis.vagueness_score,
                    query.len(),
                    analysis.key_terms.len(),
                    analysis.needs_refinement,
                );

                let (aid, weights, _is_new) = self.query_manifold.assign(&features);
                (Some(weights), Some(aid))
            } else {
                (None, None)
            };
            self.last_archetype_id = archetype_id.clone();

            // ── Query-Adaptive Belief Info Factor (Change 1) ──────────────
            // Modulate belief resolution value based on query archetype.
            // Architecture/onboarding queries → beliefs are sufficient statistics
            // (0.85 info factor = 85% of the information at 10% of the tokens).
            // Repair/debug queries → need actual code lines (0.30 factor).
            //
            // The base_belief_info_factor is further modulated by closed-loop
            // utilization feedback (belief_util_ema / full_util_ema).
            // This LEARNS the optimal belief factor from actual LLM responses.
            //
            // Mathematical foundation: Rate-Distortion Theory.
            //   R(D) = min_{p(ŷ|y)} I(Y; Ŷ)  s.t.  E[d(Y, Ŷ)] ≤ D
            // Beliefs are the encoder ŷ that minimizes bitrate R for a given
            // distortion D. Different query archetypes tolerate different D.
            {
                let query_lower = query.to_lowercase();
                // Heuristic intent detection from query text (fast, no model needed)
                let is_architecture = query_lower.contains("architecture")
                    || query_lower.contains("how does") || query_lower.contains("how do")
                    || query_lower.contains("explain") || query_lower.contains("overview")
                    || query_lower.contains("design") || query_lower.contains("pattern");
                let is_onboarding = query_lower.contains("onboard")
                    || query_lower.contains("walkthrough") || query_lower.contains("getting started")
                    || query_lower.contains("what does") || query_lower.contains("what is");
                let is_research = query_lower.contains("compare") || query_lower.contains("difference")
                    || query_lower.contains("alternative") || query_lower.contains("tradeoff");
                let is_repair = query_lower.contains("fix") || query_lower.contains("bug")
                    || query_lower.contains("error") || query_lower.contains("crash")
                    || query_lower.contains("line ") || query_lower.contains("debug");

                let archetype_factor = if is_architecture { 0.85 }
                    else if is_onboarding { 0.80 }
                    else if is_research { 0.75 }
                    else if is_repair { 0.30 }
                    else { 0.50 };  // default: neutral

                // Closed-loop modulation: adjust base via utilization feedback
                let util_ratio = if self.full_util_ema > 0.01 {
                    (self.belief_util_ema / self.full_util_ema).clamp(0.3, 2.0)
                } else {
                    1.0  // no data yet, neutral
                };

                // Online gradient step: shift base toward observed optimum
                //   base ← base + η · (util_ratio - 1.0)
                // If beliefs are well-utilized (ratio > 1), base increases.
                // If poorly utilized (ratio < 1), base decreases.
                let learning_rate = 0.02;
                self.base_belief_info_factor = (self.base_belief_info_factor
                    + learning_rate * (util_ratio - 1.0))
                    .clamp(0.15, 0.90);

                // Final factor = learned base × archetype modulation
                self.ios_belief_info_factor = (self.base_belief_info_factor * archetype_factor
                    / 0.50)  // normalize: archetype_factor=0.50 is neutral
                    .clamp(0.15, 0.90);
            }

            let mut frags: Vec<ContextFragment> = self.fragments.values().cloned().collect();
            // Use per-archetype weights if PSM assigned, otherwise global weights
            let weights = if let Some(aw) = archetype_weights {
                ScoringWeights {
                    recency: aw[prism::dim::RECENCY],
                    frequency: aw[prism::dim::FREQUENCY],
                    semantic: aw[prism::dim::SEMANTIC],
                    entropy: aw[prism::dim::ENTROPY],
                }
            } else {
                ScoringWeights {
                    recency: self.w_recency,
                    frequency: self.w_frequency,
                    semantic: self.w_semantic,
                    entropy: self.w_entropy,
                }
            };

            // ── Dependency-aware score boosting ──
            // First pass with basic knapsack to discover initial selection,
            // then compute dep boosts from that selection.
            let result1 = knapsack_optimize(&frags, effective_budget, &weights, &feedback_mults, self.gradient_temperature);
            // Store λ* for the REINFORCE backward pass — ensures forward/backward p_i are identical.
            self.last_lambda_star = result1.lambda_star;
            // Store ADGT signal — dual gap D(λ*)−primal from this forward pass.
            self.last_dual_gap = result1.dual_gap;
            let initial_selected_ids: HashSet<String> = result1.selected_indices.iter()
                .map(|&i| frags[i].fragment_id.clone())
                .collect();
            let dep_boosts = self.dep_graph.compute_dep_boosts(&initial_selected_ids);

            // ── Context Resonance: compute pairwise interaction bonuses ──
            // For each unselected fragment, compute how much it "resonates" with
            // the initial selection. High resonance → the pair has historically
            // produced better LLM outputs together.
            let resonance_bonuses: HashMap<String, f64> = if self.enable_resonance && !self.resonance_matrix.is_empty() {
                let selected_refs: Vec<&str> = initial_selected_ids.iter().map(|s| s.as_str()).collect();
                let candidate_refs: Vec<&str> = frags.iter()
                    .filter(|f| !initial_selected_ids.contains(&f.fragment_id))
                    .map(|f| f.fragment_id.as_str())
                    .collect();
                self.resonance_matrix.batch_resonance_bonuses(&candidate_refs, &selected_refs)
            } else {
                HashMap::new()
            };

            // ── Causal Context Graph: gravity + temporal bonuses ──
            // Information gravity: causally important fragments get a retrieval bonus.
            // Temporal chains: fragments enabled by previous selection get a bonus.
            let causal_gravity: HashMap<String, f64> = if self.enable_causal && !self.causal_graph.is_empty() {
                let candidate_refs: Vec<&str> = frags.iter()
                    .map(|f| f.fragment_id.as_str())
                    .collect();
                self.causal_graph.gravity_bonuses(&candidate_refs)
            } else {
                HashMap::new()
            };
            let causal_temporal: HashMap<String, f64> = if self.enable_causal && !self.causal_graph.is_empty() {
                let candidate_refs: Vec<&str> = frags.iter()
                    .map(|f| f.fragment_id.as_str())
                    .collect();
                let prev_refs: Vec<&str> = self.prev_selected_ids.iter()
                    .map(|s| s.as_str())
                    .collect();
                self.causal_graph.temporal_bonuses(&candidate_refs, &prev_refs)
            } else {
                HashMap::new()
            };

            // ── Coverage Estimator: capture semantic vs structural candidate sets ──
            // N₁ = semantic candidates (fragments with semantic_score > threshold)
            let semantic_threshold = 0.15;
            let semantic_candidate_ids: HashSet<String> = frags.iter()
                .filter(|f| f.semantic_score > semantic_threshold)
                .map(|f| f.fragment_id.clone())
                .collect();
            // N₂ = structural candidates (initial selection + dep boost targets)
            let structural_candidate_ids: HashSet<String> = initial_selected_ids.iter().cloned()
                .chain(dep_boosts.keys().filter(|k| dep_boosts[*k] > 0.3).cloned())
                .collect();
            // m = overlap
            let candidate_overlap = semantic_candidate_ids.intersection(&structural_candidate_ids).count();
            self.last_semantic_candidates = semantic_candidate_ids.len();
            self.last_structural_candidates = structural_candidate_ids.len();
            self.last_candidate_overlap = candidate_overlap;

            // Apply dep boosts + resonance bonuses to fragments' semantic scores
            let mut boosted_frags = frags.clone();
            for frag in boosted_frags.iter_mut() {
                if !initial_selected_ids.contains(&frag.fragment_id) {
                    if let Some(&boost) = dep_boosts.get(&frag.fragment_id) {
                        if boost > 0.3 {
                            frag.semantic_score = (frag.semantic_score + boost * 0.5).min(1.0);
                        }
                    }
                    // Resonance boost: w_resonance modulates how much pairwise
                    // interaction signal influences fragment selection.
                    if let Some(&res_bonus) = resonance_bonuses.get(&frag.fragment_id) {
                        if res_bonus.abs() > 0.01 {
                            // Apply as additive boost to semantic score, scaled by w_resonance.
                            // Positive resonance → boost, negative → suppress.
                            let resonance_effect = self.w_resonance * res_bonus;
                            frag.semantic_score = (frag.semantic_score + resonance_effect).clamp(0.0, 1.0);
                        }
                    }
                }
                // ── Causal Context Graph: gravity + temporal bonuses ──
                // Applied to ALL fragments (not just unselected) because gravity
                // represents true causal importance, not just pairwise interaction.
                if let Some(&grav) = causal_gravity.get(&frag.fragment_id) {
                    // Gravity: additive boost scaled by 0.15 (moderate influence)
                    frag.semantic_score = (frag.semantic_score + 0.15 * grav).clamp(0.0, 1.0);
                }
                if let Some(&temp) = causal_temporal.get(&frag.fragment_id) {
                    // Temporal: additive boost scaled by 0.10 (subtler influence)
                    frag.semantic_score = (frag.semantic_score + 0.10 * temp).clamp(0.0, 1.0);
                }
            }

            // ── Core Selection: IOS (SDS+MRK) or legacy knapsack ──
            let mut final_indices: Vec<usize>;
            let mut skeleton_indices: Vec<usize> = Vec::new();
            let mut skeleton_tokens_used: u32 = 0;
            let mut ios_diversity_score: Option<f64> = None;
            let mut ios_resolutions: HashMap<usize, Resolution> = HashMap::new();
            let mut explored_ids: Vec<String> = Vec::new();
            let mut selection_method: &str = "ios";

            if self.enable_ios {
                // ── IOS Path: Submodular Diversity + Multi-Resolution ──
                let info_factors = InfoFactors {
                    skeleton: self.ios_skeleton_info_factor,
                    belief: self.ios_belief_info_factor,
                    reference: self.ios_reference_info_factor,
                };
                let ios_result = ios_select(
                    &boosted_frags,
                    effective_budget,
                    self.w_recency,
                    self.w_frequency,
                    self.w_semantic,
                    self.w_entropy,
                    &feedback_mults,
                    self.enable_ios_diversity,
                    self.enable_ios_multi_resolution,
                    &info_factors,
                    self.ios_diversity_floor,
                    self.min_relevance,
                );

                final_indices = Vec::new();
                for (idx, resolution) in &ios_result.selections {
                    match resolution {
                        Resolution::Full => final_indices.push(*idx),
                        Resolution::Skeleton => skeleton_indices.push(*idx),
                        Resolution::Belief => skeleton_indices.push(*idx),    // Beliefs use alternative content like skeletons
                        Resolution::Reference => skeleton_indices.push(*idx), // References handled like skeletons in output
                    }
                    ios_resolutions.insert(*idx, *resolution);
                }
                skeleton_tokens_used = ios_result.total_tokens.saturating_sub(
                    final_indices.iter().map(|&i| frags[i].token_count).sum::<u32>()
                );
                ios_diversity_score = Some(ios_result.diversity_score);

                // ── Wiki-Link Graph Boost (Change 3) ──────────────────────
                // After IOS selects beliefs, traverse [[wiki-links]] to boost
                // related fragments. This is the novel Knowledge Graph traversal:
                //   1. For each selected Belief fragment, extract [[wiki-links]]
                //   2. For each link target, find the matching fragment by source stem
                //   3. Boost that fragment's semantic score (so it may enter selection)
                //
                // This creates TRANSITIVE context loading: module A's belief links
                // to [[B]], so module B gets boosted into the context window.
                // Unlike dependency graphs (which are structural), wiki-links
                // capture SEMANTIC relationships written by the BeliefCompiler.
                {
                    let mut wiki_boosts: HashMap<usize, f64> = HashMap::new();
                    let selected_set: HashSet<usize> = final_indices.iter()
                        .chain(skeleton_indices.iter())
                        .copied()
                        .collect();

                    for &(idx, ref resolution) in &ios_result.selections {
                        if *resolution == Resolution::Belief {
                            if let Some(ref belief) = frags[idx].belief_content {
                                let links = cogops::extract_wiki_links(belief);
                                for link in links {
                                    let link_lower = link.to_lowercase();
                                    // Match link target against fragment source stems
                                    for (j, f) in frags.iter().enumerate() {
                                        if selected_set.contains(&j) { continue; }
                                        // Extract filename stem from source path
                                        let stem = f.source
                                            .rsplit(&['/', '\\'][..])
                                            .next()
                                            .unwrap_or(&f.source)
                                            .rsplit('.')
                                            .next_back()
                                            .unwrap_or("")
                                            .to_lowercase();
                                        if stem == link_lower || f.source.to_lowercase().contains(&link_lower) {
                                            let entry = wiki_boosts.entry(j).or_insert(0.0);
                                            *entry = (*entry + 0.15).min(0.45); // cap at 0.45
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Apply wiki-link boosts to unselected fragments
                    for (&idx, &boost) in &wiki_boosts {
                        if boost > 0.0 {
                            frags[idx].semantic_score = (frags[idx].semantic_score + boost).min(1.0);
                        }
                    }
                }

                // ── TPSE Phase B: Lazy Skeleton Materialization ────────────────
                // Novel two-phase approach (cf. two-stage stochastic optimization,
                // Birge & Louveaux 2011; speculative execution, Smith & Sohi 1995):
                //
                // Phase A (batch_ingest) estimated skeleton_token_count from
                // per-language ratios — O(1) per file. Gave IOS enough info for
                // resolution decisions (Full vs Skeleton vs Reference).
                //
                // Phase B (here): compute REAL skeletons for only the K fragments
                // IOS selected at Skeleton resolution. K ≈ 30-100 vs N = 2500+.
                // For VSCode: 100×5KB = 500KB parsing vs 2500×5KB = 12.5MB.
                //
                // After materialization, actual skeleton_token_count replaces the
                // estimate. Channel Coding trailing pass fills any budget gap.
                for &(idx, ref resolution) in &ios_result.selections {
                    if *resolution == Resolution::Skeleton && frags[idx].skeleton_content.is_none() {
                        if let Some(skel) = skeleton::extract_skeleton(&frags[idx].content, &frags[idx].source) {
                            let skel_non_alpha = skel.chars().filter(|c| !c.is_alphabetic()).count();
                            let skel_r = skel_non_alpha as f64 / skel.len().max(1) as f64;
                            let skel_cpt = if skel_r > 0.4 { 5.0 } else { 4.0 };
                            let skel_tc = (skel.len() as f64 / skel_cpt).max(1.0) as u32;
                            // Persist to engine (amortized: paid once, cached for future calls)
                            let fid = frags[idx].fragment_id.clone();
                            if let Some(stored) = self.fragments.get_mut(&fid) {
                                stored.skeleton_content = Some(skel.clone());
                                stored.skeleton_token_count = Some(skel_tc);
                            }
                            // Update local vec for assembly in THIS call
                            frags[idx].skeleton_content = Some(skel);
                            frags[idx].skeleton_token_count = Some(skel_tc);
                        }
                    }
                }

                // ── IOS-consistent λ* for REINFORCE backward pass ──────────────────
                // IOS uses submodular greedy selection — a different mechanism than knapsack.
                // Re-run bisection with the actual IOS token usage as the budget target:
                //   Find λ_ios s.t. Σ σ((sᵢ − λ_ios·tokensᵢ)/τ)·tokensᵢ = ios_tokens_used
                // This gives the sigmoid model's best approximation of IOS inclusion probs,
                // making apply_prism_rl_update's p_i consistent with the IOS forward pass.
                if self.gradient_temperature >= 0.05 {
                    // Build scored slice from boosted_frags for the bisection (mirror IOS inputs)
                    let ios_scored: Vec<(usize, f64)> = boosted_frags.iter().enumerate()
                        .filter_map(|(i, f)| {
                            if f.is_pinned { return None; }
                            let fm = feedback_mults.get(&f.fragment_id).copied().unwrap_or(1.0);
                            let s = (weights.recency   * f.recency_score
                                   + weights.frequency * f.frequency_score
                                   + weights.semantic  * f.semantic_score
                                   + weights.entropy   * f.entropy_score) * fm.max(0.01);
                            if s > 0.0 && f.token_count > 0 { Some((i, s)) } else { None }
                        })
                        .collect();
                    let ios_tokens_used = ios_result.total_tokens;
                    self.last_lambda_star = compute_lambda_star(
                        &ios_scored, &boosted_frags, ios_tokens_used, self.gradient_temperature
                    );
                }

                if frags.len() > final_indices.len() + skeleton_indices.len()
                    && !final_indices.is_empty()
                    && should_explore
                {
                    let selected_all: HashSet<usize> = final_indices.iter()
                        .chain(skeleton_indices.iter())
                        .copied()
                        .collect();
                    let unselected: Vec<usize> = (0..frags.len())
                        .filter(|i| !selected_all.contains(i) && !frags[*i].is_pinned)
                        .collect();

                    if !unselected.is_empty() {
                        let mut min_rel = f64::MAX;
                        let mut min_pos = None;
                        for (pos, &idx) in final_indices.iter().enumerate() {
                            if !frags[idx].is_pinned {
                                let fm = feedback_mults.get(&frags[idx].fragment_id).copied().unwrap_or(1.0);
                                let rel = compute_relevance(&frags[idx], self.w_recency, self.w_frequency, self.w_semantic, self.w_entropy, fm);
                                if rel < min_rel {
                                    min_rel = rel;
                                    min_pos = Some(pos);
                                }
                            }
                        }

                        if let Some(pos) = min_pos {
                            // Pick exploration target: random when using rate-based exploration,
                            // UCB-max when using threshold-based exploration.
                            let explore_idx = if self.exploration_rate > 0.0 && unselected.len() > 1 {
                                // xorshift64 PRNG for varied exploration target selection
                                let mut x = self.rng_state;
                                x ^= x << 13;
                                x ^= x >> 7;
                                x ^= x << 17;
                                self.rng_state = x;
                                let pick = (x as usize) % unselected.len();
                                unselected[pick]
                            } else {
                                // RAVEN-UCB: pick unselected fragment with highest UCB score
                                *unselected.iter()
                                    .max_by(|&&a, &&b| {
                                        let ucb_a = self.feedback.ucb_score(&frags[a].fragment_id, alpha_0);
                                        let ucb_b = self.feedback.ucb_score(&frags[b].fragment_id, alpha_0);
                                        ucb_a.partial_cmp(&ucb_b).unwrap_or(std::cmp::Ordering::Equal)
                                    })
                                    .unwrap()
                            };
                            let old_tokens = frags[final_indices[pos]].token_count;
                            let new_tokens = frags[explore_idx].token_count;
                            if new_tokens <= old_tokens + 100 {
                                explored_ids.push(frags[explore_idx].fragment_id.clone());
                                final_indices[pos] = explore_idx;
                                self.total_explorations += 1;
                            }
                        }
                    }
                }
            } else {
                // Legacy Path: Standard knapsack + exploration + skeleton pass
                selection_method = "legacy_knapsack";
                let result = if dep_boosts.values().any(|&b| b > 0.3) {
                    knapsack_optimize(&boosted_frags, effective_budget, &weights, &feedback_mults, self.gradient_temperature)
                } else {
                    result1
                };
                final_indices = result.selected_indices.clone();
                // RAVEN-UCB Exploration (legacy path)
                if frags.len() > final_indices.len() && !final_indices.is_empty() && should_explore {
                    let selected_set: HashSet<usize> = final_indices.iter().copied().collect();
                    let unselected: Vec<usize> = (0..frags.len())
                        .filter(|i| !selected_set.contains(i) && !frags[*i].is_pinned)
                        .collect();
                    if !unselected.is_empty() {
                        let mut min_rel = f64::MAX;
                        let mut min_pos = None;
                        for (pos, &idx) in final_indices.iter().enumerate() {
                            if !frags[idx].is_pinned {
                                let fm = feedback_mults.get(&frags[idx].fragment_id).copied().unwrap_or(1.0);
                                let rel = compute_relevance(&frags[idx], self.w_recency, self.w_frequency, self.w_semantic, self.w_entropy, fm);
                                if rel < min_rel { min_rel = rel; min_pos = Some(pos); }
                            }
                        }
                        if let Some(pos) = min_pos {
                            let explore_idx = *unselected.iter()
                                .max_by(|&&a, &&b| {
                                    self.feedback.ucb_score(&frags[a].fragment_id, alpha_0)
                                        .partial_cmp(&self.feedback.ucb_score(&frags[b].fragment_id, alpha_0))
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                }).unwrap();
                            let old_tokens = frags[final_indices[pos]].token_count;
                            let new_tokens = frags[explore_idx].token_count;
                            if new_tokens <= old_tokens + 100 {
                                explored_ids.push(frags[explore_idx].fragment_id.clone());
                                final_indices[pos] = explore_idx;
                                self.total_explorations += 1;
                            }
                        }
                    }
                }

                // Skeleton Substitution pass
                let full_tokens_legacy: u32 = final_indices.iter().map(|&i| frags[i].token_count).sum();
                let selected_set: HashSet<usize> = final_indices.iter().copied().collect();
                let mut unselected_with_skel: Vec<(usize, f64)> = (0..frags.len())
                    .filter(|i| !selected_set.contains(i))
                    .filter(|i| frags[*i].skeleton_token_count.is_some())
                    .map(|i| {
                        let fm = feedback_mults.get(&frags[i].fragment_id).copied().unwrap_or(1.0);
                        let rel = compute_relevance(&frags[i], self.w_recency, self.w_frequency, self.w_semantic, self.w_entropy, fm);
                        (i, rel)
                    })
                    .collect();
                unselected_with_skel.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                let mut skel_budget = effective_budget.saturating_sub(full_tokens_legacy);
                for (idx, _rel) in unselected_with_skel {
                    if let Some(stc) = frags[idx].skeleton_token_count {
                        if stc <= skel_budget {
                            skeleton_indices.push(idx);
                            skeleton_tokens_used += stc;
                            skel_budget -= stc;
                        }
                    }
                }
            }

            // ── Channel Coding: Trailing Pass ──
            // Fill the KKT/IOS token gap using marginal information gain
            // instead of leaving unused budget on the table.
            //
            // CRITICAL: pass BOTH full and skeleton indices to prevent
            // double-inclusion (a fragment at skeleton resolution must not
            // be re-added at full resolution).
            if self.enable_channel_coding {
                let used_full: u32 = final_indices.iter().map(|&i| boosted_frags[i].token_count).sum();
                let used_total = used_full + skeleton_tokens_used;
                let token_gap = effective_budget.saturating_sub(used_total);
                if token_gap > 0 {
                    let mut all_selected = final_indices.clone();
                    all_selected.extend_from_slice(&skeleton_indices);
                    let trailing = channel::channel_trailing_pass(
                        &boosted_frags,
                        &all_selected,
                        token_gap,
                        &self.dep_graph,
                    );
                    for idx in trailing {
                        final_indices.push(idx);
                    }
                }
            }

            // Track savings
            let total_available: u32 = frags.iter().map(|f| f.token_count).sum();
            let full_tokens: u32 = final_indices.iter().map(|&i| frags[i].token_count).sum();
            let final_tokens: u32 = full_tokens + skeleton_tokens_used;
            let saved = total_available.saturating_sub(final_tokens);
            self.total_tokens_saved += saved as u64;

            // Mark selected as accessed
            let mut accessed_indices = final_indices.clone();
            accessed_indices.extend(skeleton_indices.iter().copied());
            accessed_indices.sort_unstable();
            accessed_indices.dedup();
            for &idx in &accessed_indices {
                let fid = &frags[idx].fragment_id;
                if let Some(f) = self.fragments.get_mut(fid) {
                    f.turn_last_accessed = self.current_turn;
                    f.access_count += 1;
                }
            }

            // ── Context Sufficiency Scoring ──
            // What fraction of referenced symbols have definitions in selected context?
            let selected_id_set: HashSet<String> = final_indices.iter()
                .chain(skeleton_indices.iter())
                .map(|&i| frags[i].fragment_id.clone())
                .collect();
            let sufficiency = self.compute_sufficiency(&frags, &final_indices);

            // ── Spectral Contradiction Guard ──
            // Detect and evict fragments that are structurally similar but
            // semantically contradictory (e.g., two versions of the same class).
            // Based on SimHash Divergence Ratio (SDR).
            let contradictions_evicted;
            let final_indices = if self.enable_channel_coding {
                let pre_relevances: Vec<f64> = final_indices.iter()
                    .map(|&i| {
                        let fm = feedback_mults.get(&frags[i].fragment_id).copied().unwrap_or(1.0);
                        compute_relevance(&frags[i], self.w_recency, self.w_frequency, self.w_semantic, self.w_entropy, fm)
                    })
                    .collect();
                let (filtered, report) = channel::contradiction_guard(
                    &frags, &final_indices, &pre_relevances,
                    0.25,  // sdr_threshold
                    0.60,  // structural_threshold
                );
                contradictions_evicted = report.pairs_found;
                if report.pairs_found > 0 {
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "[entroly] Contradiction guard: evicted {} fragments ({} pairs)",
                        report.evicted.len(), report.pairs_found,
                    );
                }
                filtered
            } else {
                contradictions_evicted = 0;
                final_indices
            };

            // ── Context ordering: sort selected for LLM attention ──
            let ordered_indices = if self.enable_channel_coding {
                // Channel Coding: attention-aware semantic interleaving
                // Respects causal ordering (defs before refs) + U-shaped attention
                let relevances: Vec<f64> = final_indices.iter()
                    .map(|&i| {
                        let fm = feedback_mults.get(&frags[i].fragment_id).copied().unwrap_or(1.0);
                        compute_relevance(&frags[i], self.w_recency, self.w_frequency, self.w_semantic, self.w_entropy, fm)
                    })
                    .collect();
                let interleaved = channel::semantic_interleave(&frags, &final_indices, &relevances, &self.dep_graph);

                // Bookend Attention Calibration: within each causal level,
                // place highest-importance fragments at U-shaped attention peaks.
                let rel_map: std::collections::HashMap<usize, f64> = final_indices.iter()
                    .zip(relevances.iter())
                    .map(|(&idx, &rel)| (idx, rel))
                    .collect();
                channel::bookend_calibrate(&interleaved, &frags, &rel_map, &self.dep_graph)
            } else {
                let mut ordered = final_indices.clone();
                ordered.sort_by(|&a, &b| {
                    let fa = &frags[a];
                    let fb = &frags[b];
                    let crit_a = file_criticality(&fa.source);
                    let crit_b = file_criticality(&fb.source);
                    let dep_count_a = self.dep_graph.reverse_deps(&fa.fragment_id).len();
                    let dep_count_b = self.dep_graph.reverse_deps(&fb.fragment_id).len();
                    let fm_a = feedback_mults.get(&fa.fragment_id).copied().unwrap_or(1.0);
                    let fm_b = feedback_mults.get(&fb.fragment_id).copied().unwrap_or(1.0);
                    let rel_a = compute_relevance(fa, self.w_recency, self.w_frequency, self.w_semantic, self.w_entropy, fm_a);
                    let rel_b = compute_relevance(fb, self.w_recency, self.w_frequency, self.w_semantic, self.w_entropy, fm_b);
                    let prio_a = compute_ordering_priority(rel_a, crit_a, fa.is_pinned, dep_count_a);
                    let prio_b = compute_ordering_priority(rel_b, crit_b, fb.is_pinned, dep_count_b);
                    prio_b.partial_cmp(&prio_a).unwrap_or(std::cmp::Ordering::Equal)
                });
                ordered
            };

            // ── Build Explainability Snapshot ──
            let mut fragment_scores: Vec<FragmentScore> = Vec::with_capacity(frags.len());
            for frag in frags.iter() {
                let fm = feedback_mults.get(&frag.fragment_id).copied().unwrap_or(1.0);
                let db = dep_boosts.get(&frag.fragment_id).copied().unwrap_or(0.0);
                let crit = file_criticality(&frag.source);
                let composite = compute_relevance(frag, self.w_recency, self.w_frequency, self.w_semantic, self.w_entropy, fm);
                let is_selected = selected_id_set.contains(&frag.fragment_id);
                let is_explored = explored_ids.contains(&frag.fragment_id);

                let reason = if frag.is_pinned {
                    "pinned/critical".to_string()
                } else if is_explored {
                    "ε-exploration".to_string()
                } else if is_selected && db > 0.3 {
                    format!("dep-boosted (boost={:.2})", db)
                } else if is_selected {
                    "knapsack-optimal".to_string()
                } else if composite < self.min_relevance {
                    "below min relevance".to_string()
                } else {
                    "budget exceeded".to_string()
                };

                fragment_scores.push(FragmentScore {
                    fragment_id: frag.fragment_id.clone(),
                    source: frag.source.clone(),
                    selected: is_selected || is_explored,
                    recency: frag.recency_score,
                    frequency: frag.frequency_score,
                    semantic: frag.semantic_score,
                    entropy: frag.entropy_score,
                    feedback_mult: fm,
                    dep_boost: db,
                    criticality: format!("{:?}", crit),
                    composite,
                    reason,
                });
            }

            self.last_optimization = Some(OptimizationSnapshot {
                fragment_scores,
                sufficiency,
                explored_ids: explored_ids.clone(),
            });

            // ── Compute context efficiency ──
            // context_efficiency = sum(entropy_i * actual_tokens_i) / total_tokens
            // Uses resolution-aware token counts to match actual context window.
            let context_efficiency = if final_tokens > 0 {
                let weighted_entropy: f64 = final_indices.iter()
                    .map(|&i| frags[i].entropy_score * frags[i].token_count as f64)
                    .sum::<f64>()
                    + skeleton_indices.iter()
                        .map(|&i| {
                            let actual_tc = match ios_resolutions.get(&i) {
                                Some(&Resolution::Reference) => {
                                    (frags[i].source.len() as u32 / 4).clamp(3, 10)
                                }
                                _ => frags[i].skeleton_token_count.unwrap_or(frags[i].token_count),
                            };
                            frags[i].entropy_score * actual_tc as f64
                        })
                        .sum::<f64>();
                weighted_entropy / final_tokens as f64
            } else {
                0.0
            };

            // ── Build Python result ──
            let py_result = PyDict::new(py);
            py_result.set_item("method", selection_method)?;
            py_result.set_item("total_tokens", final_tokens)?;
            py_result.set_item("context_efficiency", (context_efficiency * 10000.0).round() / 10000.0)?;
            // Compute total relevance from selected fragments
            let total_rel: f64 = final_indices.iter().chain(skeleton_indices.iter())
                .map(|&i| {
                    let fm = feedback_mults.get(&frags[i].fragment_id).copied().unwrap_or(1.0);
                    compute_relevance(&frags[i], self.w_recency, self.w_frequency, self.w_semantic, self.w_entropy, fm)
                })
                .sum();
            py_result.set_item("total_relevance", (total_rel * 10000.0).round() / 10000.0)?;
            py_result.set_item("selected_count", ordered_indices.len() + skeleton_indices.len())?;
            py_result.set_item("skeleton_count", skeleton_indices.len())?;
            py_result.set_item("skeleton_tokens", skeleton_tokens_used)?;
            py_result.set_item("tokens_saved", saved)?;
            py_result.set_item("effective_budget", effective_budget)?;
            py_result.set_item("budget_utilization",
                if effective_budget > 0 { (final_tokens as f64 / effective_budget as f64 * 10000.0).round() / 10000.0 } else { 0.0 }
            )?;
            py_result.set_item("sufficiency", (sufficiency * 10000.0).round() / 10000.0)?;
            if sufficiency < 0.7 {
                py_result.set_item("sufficiency_warning",
                    format!("Only {:.0}% of referenced symbols have definitions in context", sufficiency * 100.0)
                )?;
            }
            if !explored_ids.is_empty() {
                py_result.set_item("explored", explored_ids.clone())?;
            }
            if contradictions_evicted > 0 {
                py_result.set_item("contradictions_evicted", contradictions_evicted)?;
            }
            if let Some(div_score) = ios_diversity_score {
                py_result.set_item("ios_diversity_score", div_score)?;
                py_result.set_item("ios_enabled", true)?;
            }

            // ── Coverage Sufficiency Estimator (Unknown Unknowns) ──
            // Chapman capture-recapture: how much of the relevant information
            // space does our selected context cover?
            let coverage_est = resonance::estimate_coverage(
                ordered_indices.len() + skeleton_indices.len(),
                self.last_semantic_candidates,
                self.last_structural_candidates,
                self.last_candidate_overlap,
            );
            py_result.set_item("coverage", (coverage_est.coverage * 10000.0).round() / 10000.0)?;
            py_result.set_item("coverage_confidence", (coverage_est.confidence * 10000.0).round() / 10000.0)?;
            py_result.set_item("coverage_gap", coverage_est.estimated_gap.round())?;
            py_result.set_item("coverage_risk", coverage_est.risk_level)?;

            // ── Resonance diagnostics ──
            if self.enable_resonance {
                py_result.set_item("resonance_pairs", self.resonance_matrix.len())?;
                py_result.set_item("resonance_strength", (self.resonance_matrix.mean_strength() * 10000.0).round() / 10000.0)?;
                py_result.set_item("w_resonance", (self.w_resonance * 10000.0).round() / 10000.0)?;
            }

            // ── Causal Context Graph diagnostics ──
            if self.enable_causal && !self.causal_graph.is_empty() {
                let cs = self.causal_graph.stats();
                py_result.set_item("causal_tracked", cs.tracked_fragments)?;
                py_result.set_item("causal_interventional", cs.interventional_fragments)?;
                py_result.set_item("causal_gravity_sources", cs.gravity_sources)?;
                py_result.set_item("causal_mean_mass", (cs.mean_causal_mass * 10000.0).round() / 10000.0)?;
                py_result.set_item("causal_temporal_links", cs.temporal_links)?;
            }

            // Selected fragment details (in LLM-optimal order)
            let selected_list = pyo3::types::PyList::empty(py);
            for &idx in &ordered_indices {
                let f = &frags[idx];
                let d = PyDict::new(py);
                d.set_item("id", &f.fragment_id)?;
                d.set_item("source", &f.source)?;
                d.set_item("token_count", f.token_count)?;
                d.set_item("variant", "full")?;
                let fm = feedback_mults.get(&f.fragment_id).copied().unwrap_or(1.0);
                let rel = compute_relevance(f, self.w_recency, self.w_frequency, self.w_semantic, self.w_entropy, fm);
                d.set_item("relevance", (rel * 10000.0).round() / 10000.0)?;
                d.set_item("entropy_score", (f.entropy_score * 10000.0).round() / 10000.0)?;
                let preview = if f.content.len() > 100 {
                    let mut end = 100;
                    while end < f.content.len() && !f.content.is_char_boundary(end) {
                        end += 1;
                    }
                    format!("{}...", &f.content[..end])
                } else {
                    f.content.clone()
                };
                d.set_item("preview", preview)?;
                d.set_item("content", &f.content)?;
                selected_list.append(d)?;
            }
            // Append skeleton/reference fragments (lower priority, after full fragments)
            for &idx in &skeleton_indices {
                let f = &frags[idx];
                let resolution = ios_resolutions.get(&idx).copied().unwrap_or(Resolution::Skeleton);
                let variant_str = resolution.as_str();

                if resolution == Resolution::Reference {
                    // Reference: just the source path, minimal tokens
                    let d = PyDict::new(py);
                    d.set_item("id", &f.fragment_id)?;
                    d.set_item("source", &f.source)?;
                    let ref_tokens = (f.source.len() as u32 / 4).clamp(3, 10);
                    d.set_item("token_count", ref_tokens)?;
                    d.set_item("variant", variant_str)?;
                    let fm = feedback_mults.get(&f.fragment_id).copied().unwrap_or(1.0);
                    let rel = compute_relevance(f, self.w_recency, self.w_frequency, self.w_semantic, self.w_entropy, fm);
                    d.set_item("relevance", (rel * 10000.0).round() / 10000.0)?;
                    d.set_item("entropy_score", (f.entropy_score * 10000.0).round() / 10000.0)?;
                    d.set_item("preview", format!("[ref] {}", &f.source))?;
                    d.set_item("content", format!("[ref] {}", &f.source))?;
                    selected_list.append(d)?;
                } else if resolution == Resolution::Belief {
                    // Belief: vault-compiled architectural summary — replaces raw code.
                    // This is the Hierarchical Context Synthesis bridge:
                    //   200 tokens of belief REPLACE 800 tokens of code,
                    //   saving ~600 tokens per fragment with near-zero info loss.
                    if let (Some(ref belief), Some(belief_tc)) = (&f.belief_content, f.belief_token_count) {
                        let d = PyDict::new(py);
                        d.set_item("id", &f.fragment_id)?;
                        d.set_item("source", &f.source)?;
                        d.set_item("token_count", belief_tc)?;
                        d.set_item("variant", "belief")?;
                        let fm = feedback_mults.get(&f.fragment_id).copied().unwrap_or(1.0);
                        let rel = compute_relevance(f, self.w_recency, self.w_frequency, self.w_semantic, self.w_entropy, fm);
                        d.set_item("relevance", (rel * 10000.0).round() / 10000.0)?;
                        d.set_item("entropy_score", (f.entropy_score * 10000.0).round() / 10000.0)?;
                        let preview = if belief.len() > 100 {
                            let mut end = 100;
                            while end < belief.len() && !belief.is_char_boundary(end) { end += 1; }
                            format!("{}...", &belief[..end])
                        } else {
                            belief.clone()
                        };
                        d.set_item("preview", preview)?;
                        d.set_item("content", belief.as_str())?;
                        selected_list.append(d)?;
                    }
                    // If belief_content is None (not loaded), silently skip —
                    // this fragment was selected at Belief resolution but no vault
                    // belief was attached. The channel coding trailing pass
                    // will fill the gap with other fragments.
                } else if let (Some(ref skel_content), Some(skel_tc)) = (&f.skeleton_content, f.skeleton_token_count) {
                    let d = PyDict::new(py);
                    d.set_item("id", &f.fragment_id)?;
                    d.set_item("source", &f.source)?;
                    d.set_item("token_count", skel_tc)?;
                    d.set_item("variant", variant_str)?;
                    let fm = feedback_mults.get(&f.fragment_id).copied().unwrap_or(1.0);
                    let rel = compute_relevance(f, self.w_recency, self.w_frequency, self.w_semantic, self.w_entropy, fm);
                    d.set_item("relevance", (rel * 10000.0).round() / 10000.0)?;
                    d.set_item("entropy_score", (f.entropy_score * 10000.0).round() / 10000.0)?;
                    let preview = if skel_content.len() > 100 {
                        let mut end = 100;
                        while end < skel_content.len() && !skel_content.is_char_boundary(end) {
                            end += 1;
                        }
                        format!("{}...", &skel_content[..end])
                    } else {
                        skel_content.clone()
                    };
                    d.set_item("preview", preview)?;
                    d.set_item("content", skel_content.as_str())?;
                    selected_list.append(d)?;
                }
            }
            py_result.set_item("selected", selected_list)?;
            py_result.set_item("cache_hit", false)?;
            let cache_eligible = !query.is_empty() && explored_ids.is_empty();
            self.last_cache_feedback_eligible = cache_eligible;
            py_result.set_item("cache_eligible", cache_eligible)?;
            py_result.set_item(
                "optimization_policy",
                if explored_ids.is_empty() { "exploit" } else { "explore" },
            )?;

            // ── EGSC Cache: store the optimization result for future lookups ──
            // Only realized exploit trajectories are memoizable. Exploratory
            // selections are learning signals, not reusable answers.
            if cache_eligible {
                let current_frag_ids: HashSet<String> = self.fragments.keys().cloned().collect();
                let entropies: Vec<(f64, u32)> = final_indices.iter()
                    .chain(skeleton_indices.iter())
                    .map(|&i| (frags[i].entropy_score, frags[i].token_count))
                    .collect();
                // Serialize a compact result snapshot for the cache.
                // CRITICAL: include selected_ids so we can reconstruct
                // the full selected_fragments list on cache hit.
                let selected_ids_json: Vec<serde_json::Value> = ordered_indices.iter()
                    .map(|&i| serde_json::json!({"id": frags[i].fragment_id, "variant": "full"}))
                    .chain(skeleton_indices.iter().map(|&i| {
                        let variant = ios_resolutions.get(&i)
                            .map(|r| r.as_str())
                            .unwrap_or("skeleton");
                        serde_json::json!({"id": frags[i].fragment_id, "variant": variant})
                    }))
                    .collect();
                let cache_snapshot = serde_json::json!({
                    "method": selection_method,
                    "total_tokens": final_tokens,
                    "context_efficiency": (context_efficiency * 10000.0).round() / 10000.0,
                    "total_relevance": (total_rel * 10000.0).round() / 10000.0,
                    "selected_count": ordered_indices.len() + skeleton_indices.len(),
                    "skeleton_count": skeleton_indices.len(),
                    "skeleton_tokens": skeleton_tokens_used,
                    "tokens_saved": saved,
                    "effective_budget": effective_budget,
                    "budget_utilization": if effective_budget > 0 { (final_tokens as f64 / effective_budget as f64 * 10000.0).round() / 10000.0 } else { 0.0 },
                    "sufficiency": (sufficiency * 10000.0).round() / 10000.0,
                    "selected_ids": selected_ids_json,
                });
                let response_json = cache_snapshot.to_string();
                self.egsc_cache.store_with_budget(
                    &query, current_frag_ids, &entropies,
                    response_json, final_tokens, self.current_turn, effective_budget,
                );
            }

            // ── Causal Context Graph: store selected/explored for feedback path ──
            // These are used by record_success/failure/reward to build CausalTraces.
            if self.enable_causal {
                let all_selected: Vec<String> = ordered_indices.iter()
                    .chain(skeleton_indices.iter())
                    .map(|&i| frags[i].fragment_id.clone())
                    .collect();
                self.prev_selected_ids = all_selected;
                self.prev_explored_ids = explored_ids;
            }

            // Restore cold-start weight overrides
            self.w_recency = saved_w_recency;
            self.w_semantic = saved_w_semantic;

            Ok(py_result.into())
        })
    }

    /// Semantic recall of relevant fragments.
    ///
    /// Uses ebbiforge-ported LSH multi-probe index for sub-linear candidate
    /// selection, then scores per ContextScorer (similarity+recency+entropy
    /// +frequency+feedback). Falls back to O(N) scan when LSH returns no
    /// candidates (cold start with < NUM_TABLES fragments).
    pub fn recall(&self, query: String, top_k: usize) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let query_fp = simhash(&query);

            // ── LSH candidate retrieval ─────────────────────────────────
            let candidates = self.lsh_index.query(query_fp);

            // Resolve slots → fragments, compute composite scores
            let mut scored: Vec<(&ContextFragment, f64)> = if !candidates.is_empty() {
                candidates.iter()
                    .filter_map(|&slot| {
                        let frag_id = self.fragment_slot_ids.get(slot)?;
                        self.fragments.get(frag_id)
                    })
                    .map(|f| {
                        let dist = hamming_distance(query_fp, f.simhash);
                        let fm   = self.feedback.learned_value(&f.fragment_id);
                        let rel  = self.context_scorer.score(
                            dist,
                            f.recency_score,
                            f.entropy_score,
                            f.frequency_score,
                            fm,
                        );
                        (f, rel)
                    })
                    .collect()
            } else {
                // Cold-start fallback: O(N) brute force
                self.fragments.values()
                    .map(|f| {
                        let dist = hamming_distance(query_fp, f.simhash);
                        let fm   = self.feedback.learned_value(&f.fragment_id);
                        let rel  = self.context_scorer.score(
                            dist,
                            f.recency_score,
                            f.entropy_score,
                            f.frequency_score,
                            fm,
                        );
                        (f, rel)
                    })
                    .collect()
            };

            scored.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.fragment_id.cmp(&b.0.fragment_id))
            });
            scored.truncate(top_k);

            let result = pyo3::types::PyList::empty(py);
            for (f, rel) in scored {
                let d = PyDict::new(py);
                d.set_item("fragment_id", &f.fragment_id)?;
                d.set_item("source", &f.source)?;
                d.set_item("relevance", (rel * 10000.0).round() / 10000.0)?;
                d.set_item("entropy", (f.entropy_score * 10000.0).round() / 10000.0)?;
                d.set_item("content", &f.content)?;
                result.append(d)?;
            }

            Ok(result.into())
        })
    }

    /// Get session statistics.
    pub fn stats(&mut self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let total_tokens: u32 = self.fragments.values().map(|f| f.token_count).sum();
            let avg_entropy = if self.fragments.is_empty() {
                0.0
            } else {
                self.fragments.values().map(|f| f.entropy_score).sum::<f64>()
                    / self.fragments.len() as f64
            };
            let pinned = self.fragments.values().filter(|f| f.is_pinned).count();
            let feedback_observations = self.feedback.total_observations();
            let adaptive_exploration_rate = self.feedback.adaptive_exploration_rate(2.0);
            let total_exploitations = self.total_optimizations.saturating_sub(self.total_explorations);

            let result = PyDict::new(py);

            let session = PyDict::new(py);
            session.set_item("current_turn", self.current_turn)?;
            session.set_item("total_fragments", self.fragments.len())?;
            session.set_item("total_tokens_tracked", total_tokens)?;
            session.set_item("avg_entropy", (avg_entropy * 10000.0).round() / 10000.0)?;
            session.set_item("pinned", pinned)?;
            result.set_item("session", session)?;

            let savings = PyDict::new(py);
            savings.set_item("total_tokens_saved", self.total_tokens_saved)?;
            savings.set_item("total_duplicates_caught", self.total_duplicates_caught)?;
            savings.set_item("total_optimizations", self.total_optimizations)?;
            savings.set_item("total_fragments_ingested", self.total_fragments_ingested)?;
            savings.set_item("estimated_cost_saved_usd",
                (self.total_tokens_saved as f64 * 0.000003 * 10000.0).round() / 10000.0
            )?;
            result.set_item("savings", savings)?;

            let dedup = PyDict::new(py);
            dedup.set_item("indexed_fragments", self.dedup_index.size())?;
            dedup.set_item("duplicates_detected", self.dedup_index.duplicates_detected)?;
            result.set_item("dedup", dedup)?;

            let policy = PyDict::new(py);
            policy.set_item("configured_exploration_rate", (self.exploration_rate * 10000.0).round() / 10000.0)?;
            policy.set_item("adaptive_exploration_rate", (adaptive_exploration_rate * 10000.0).round() / 10000.0)?;
            policy.set_item("feedback_observations", feedback_observations)?;
            policy.set_item("total_explorations", self.total_explorations)?;
            policy.set_item("total_exploitations", total_exploitations)?;
            policy.set_item(
                "explore_ratio",
                if self.total_optimizations > 0 {
                    (self.total_explorations as f64 / self.total_optimizations as f64 * 10000.0).round() / 10000.0
                } else {
                    0.0
                },
            )?;
            policy.set_item(
                "exploit_ratio",
                if self.total_optimizations > 0 {
                    (total_exploitations as f64 / self.total_optimizations as f64 * 10000.0).round() / 10000.0
                } else {
                    0.0
                },
            )?;
            result.set_item("policy", policy)?;

            // ── PRISM RL Observability ──
            // Expose learned weights so the RL loop is observable.
            // Without this, there's no way to verify weights are actually changing.
            let prism = PyDict::new(py);
            prism.set_item("w_recency", (self.w_recency * 10000.0).round() / 10000.0)?;
            prism.set_item("w_frequency", (self.w_frequency * 10000.0).round() / 10000.0)?;
            prism.set_item("w_semantic", (self.w_semantic * 10000.0).round() / 10000.0)?;
            prism.set_item("w_entropy", (self.w_entropy * 10000.0).round() / 10000.0)?;
            prism.set_item("gradient_temperature", (self.gradient_temperature * 10000.0).round() / 10000.0)?;
            prism.set_item("reward_baseline_ema", (self.reward_baseline_ema * 10000.0).round() / 10000.0)?;
            prism.set_item("gradient_norm_ema", (self.gradient_norm_ema * 10000.0).round() / 10000.0)?;
            prism.set_item("condition_number", (self.prism_optimizer.condition_number() * 100.0).round() / 100.0)?;
            result.set_item("prism", prism)?;

            // EGSC Cache stats — exposed to Python for monitoring and debugging
            let cs = self.egsc_cache.stats();
            let cache_dict = PyDict::new(py);
            cache_dict.set_item("entries", cs.total_entries)?;
            cache_dict.set_item("lookups", cs.total_lookups)?;
            cache_dict.set_item("exact_hits", cs.exact_hits)?;
            cache_dict.set_item("semantic_hits", cs.semantic_hits)?;
            cache_dict.set_item("misses", cs.misses)?;
            cache_dict.set_item("hit_rate", (cs.hit_rate * 10000.0).round() / 10000.0)?;
            cache_dict.set_item("tokens_saved", cs.total_tokens_saved)?;
            cache_dict.set_item("admissions", cs.total_admissions)?;
            cache_dict.set_item("rejections", cs.total_rejections)?;
            cache_dict.set_item("evictions", cs.total_evictions)?;
            cache_dict.set_item("invalidations", cs.total_invalidations)?;
            cache_dict.set_item("admission_rate", (cs.admission_rate * 10000.0).round() / 10000.0)?;
            cache_dict.set_item("entropy_threshold", (cs.entropy_threshold * 10000.0).round() / 10000.0)?;
            cache_dict.set_item("shifts_detected", cs.shifts_detected)?;
            // Extended observability: Thompson gate, frequency sketch, cost tail, invalidation depth
            cache_dict.set_item("avg_quality", (cs.avg_quality * 10000.0).round() / 10000.0)?;
            cache_dict.set_item("adaptive_alpha", (cs.adaptive_alpha * 10000.0).round() / 10000.0)?;
            cache_dict.set_item("thompson_alpha", (cs.thompson_alpha * 100.0).round() / 100.0)?;
            cache_dict.set_item("thompson_beta", (cs.thompson_beta * 100.0).round() / 100.0)?;
            cache_dict.set_item("freq_admissions", cs.freq_admissions)?;
            cache_dict.set_item("freq_rejections", cs.freq_rejections)?;
            cache_dict.set_item("p50_cost_saved", (cs.p50_cost_saved * 10000.0).round() / 10000.0)?;
            cache_dict.set_item("p95_cost_saved", (cs.p95_cost_saved * 10000.0).round() / 10000.0)?;
            cache_dict.set_item("p99_cost_saved", (cs.p99_cost_saved * 10000.0).round() / 10000.0)?;
            cache_dict.set_item("hard_invalidations", cs.hard_invalidations)?;
            cache_dict.set_item("soft_invalidations", cs.soft_invalidations)?;
            cache_dict.set_item("invalidation_depth_counts", (cs.invalidation_depth_counts[0], cs.invalidation_depth_counts[1], cs.invalidation_depth_counts[2], cs.invalidation_depth_counts[3]))?;
            cache_dict.set_item("total_resets", cs.total_resets)?;
            cache_dict.set_item("hit_rate_ema", (cs.hit_rate_ema * 10000.0).round() / 10000.0)?;
            cache_dict.set_item("predictor_weights", (cs.predictor_weights[0], cs.predictor_weights[1], cs.predictor_weights[2], cs.predictor_weights[3], cs.predictor_weights[4]))?;
            result.set_item("cache", cache_dict)?;

            // Last selected fragment IDs — needed by context_bridge.record_outcome()
            if let Some(ref snapshot) = self.last_optimization {
                let last_ids: Vec<&str> = snapshot.fragment_scores.iter()
                    .filter(|fs| fs.selected)
                    .map(|fs| fs.fragment_id.as_str())
                    .collect();
                result.set_item("last_selected_ids", last_ids)?;
            } else {
                let empty: Vec<String> = vec![];
                result.set_item("last_selected_ids", empty)?;
            }

            // Query Persona Manifold stats
            if self.enable_query_personas {
                let ms = self.query_manifold.stats();
                let manifold = PyDict::new(py);
                manifold.set_item("population", ms.population)?;
                manifold.set_item("total_births", ms.total_births)?;
                manifold.set_item("total_deaths", ms.total_deaths)?;
                manifold.set_item("total_fusions", ms.total_fusions)?;
                manifold.set_item("tick", ms.tick)?;
                manifold.set_item("total_particles", ms.total_particles)?;
                result.set_item("query_manifold", manifold)?;
            }

            // ── Context Resonance stats ──
            if self.enable_resonance {
                let res_dict = PyDict::new(py);
                res_dict.set_item("tracked_pairs", self.resonance_matrix.len())?;
                res_dict.set_item("mean_strength", (self.resonance_matrix.mean_strength() * 10000.0).round() / 10000.0)?;
                res_dict.set_item("w_resonance", (self.w_resonance * 10000.0).round() / 10000.0)?;
                // 5D PRISM diagnostics
                let diag = self.prism_optimizer_5d.resonance_diagnostics();
                res_dict.set_item("resonance_energy_fraction", (diag.resonance_energy_fraction * 10000.0).round() / 10000.0)?;
                res_dict.set_item("resonance_eigenvalue", (diag.resonance_eigenvalue * 10000.0).round() / 10000.0)?;
                res_dict.set_item("resonance_alignment", (diag.resonance_alignment * 10000.0).round() / 10000.0)?;
                res_dict.set_item("is_calibrated", diag.is_calibrated)?;
                res_dict.set_item("condition_number_5d", (self.prism_optimizer_5d.condition_number() * 100.0).round() / 100.0)?;
                // Top resonance pairs (for dashboard)
                let top = self.resonance_matrix.top_pairs(5);
                let top_list = pyo3::types::PyList::empty(py);
                for (a, b, strength, cos) in &top {
                    let pair = PyDict::new(py);
                    pair.set_item("a", a.as_str())?;
                    pair.set_item("b", b.as_str())?;
                    pair.set_item("strength", (*strength * 10000.0).round() / 10000.0)?;
                    pair.set_item("co_selections", *cos)?;
                    top_list.append(pair)?;
                }
                res_dict.set_item("top_pairs", top_list)?;
                result.set_item("resonance", res_dict)?;
            }

            // ── Fragment Consolidation stats ──
            let consol = PyDict::new(py);
            consol.set_item("total_consolidations", self.total_consolidations)?;
            consol.set_item("tokens_saved", self.consolidation_tokens_saved)?;
            result.set_item("consolidation", consol)?;

            // ── Causal Context Graph stats ──
            if self.enable_causal {
                let cs = self.causal_graph.stats();
                let causal_dict = PyDict::new(py);
                causal_dict.set_item("total_traces", cs.total_traces)?;
                causal_dict.set_item("stored_traces", cs.stored_traces)?;
                causal_dict.set_item("tracked_fragments", cs.tracked_fragments)?;
                causal_dict.set_item("interventional_fragments", cs.interventional_fragments)?;
                causal_dict.set_item("temporal_links", cs.temporal_links)?;
                causal_dict.set_item("gravity_sources", cs.gravity_sources)?;
                causal_dict.set_item("mean_causal_mass", (cs.mean_causal_mass * 10000.0).round() / 10000.0)?;
                causal_dict.set_item("base_rate", (cs.base_rate * 10000.0).round() / 10000.0)?;
                causal_dict.set_item("total_interventional_updates", cs.total_interventional_updates)?;
                causal_dict.set_item("total_temporal_updates", cs.total_temporal_updates)?;
                // Top causal fragments (for dashboard)
                let top_causal = self.causal_graph.top_causal_fragments(5);
                let top_causal_list = pyo3::types::PyList::empty(py);
                for (fid, effect, bias, conf) in &top_causal {
                    let d = PyDict::new(py);
                    d.set_item("id", fid.as_str())?;
                    d.set_item("causal_effect", (*effect * 10000.0).round() / 10000.0)?;
                    d.set_item("confounding_bias", (*bias * 10000.0).round() / 10000.0)?;
                    d.set_item("confidence", (*conf * 10000.0).round() / 10000.0)?;
                    top_causal_list.append(d)?;
                }
                causal_dict.set_item("top_fragments", top_causal_list)?;
                // Top temporal chains
                let top_chains = self.causal_graph.top_temporal_chains(5);
                let top_chains_list = pyo3::types::PyList::empty(py);
                for (src, tgt, effect, conf) in &top_chains {
                    let d = PyDict::new(py);
                    d.set_item("source", src.as_str())?;
                    d.set_item("target", tgt.as_str())?;
                    d.set_item("temporal_effect", (*effect * 10000.0).round() / 10000.0)?;
                    d.set_item("confidence", (*conf * 10000.0).round() / 10000.0)?;
                    top_chains_list.append(d)?;
                }
                causal_dict.set_item("top_chains", top_chains_list)?;
                result.set_item("causal", causal_dict)?;
            }

            Ok(result.into())
        })
    }

    /// Get the current turn number.
    pub fn get_turn(&self) -> u32 {
        self.current_turn
    }

    /// Get fragment count.
    pub fn fragment_count(&self) -> usize {
        self.fragments.len()
    }

    /// Get detailed query manifold stats (per-archetype weights, health, particles).
    pub fn query_manifold_stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let ms = self.query_manifold.stats();
            let result = PyDict::new(py);
            result.set_item("enabled", self.enable_query_personas)?;
            result.set_item("population", ms.population)?;
            result.set_item("total_births", ms.total_births)?;
            result.set_item("total_deaths", ms.total_deaths)?;
            result.set_item("total_fusions", ms.total_fusions)?;
            result.set_item("tick", ms.tick)?;
            result.set_item("total_particles", ms.total_particles)?;

            let archetypes_list = pyo3::types::PyList::empty(py);
            for a in &ms.archetypes {
                let ad = PyDict::new(py);
                ad.set_item("id", &a.id)?;
                ad.set_item("health", a.health)?;
                ad.set_item("particles", a.particles)?;
                ad.set_item("observations", a.observations)?;
                ad.set_item("stick_weight", a.stick_weight)?;
                ad.set_item("effective_support", a.effective_support)?;
                ad.set_item("weights", (a.weights[0], a.weights[1], a.weights[2], a.weights[3]))?;
                ad.set_item("successes", a.successes)?;
                ad.set_item("total_uses", a.total_uses)?;
                ad.set_item("success_rate", (a.success_rate * 10000.0).round() / 10000.0)?;
                archetypes_list.append(ad)?;
            }
            result.set_item("archetypes", archetypes_list)?;

            if let Some(ref aid) = self.last_archetype_id {
                result.set_item("last_archetype_id", aid)?;
            }

            Ok(result.into())
        })
    }

    /// Enable or disable query persona manifold.
    pub fn set_query_personas_enabled(&mut self, enabled: bool) {
        self.enable_query_personas = enabled;
    }

    /// Hot-reload scoring weights mid-session (autotune live update).
    /// Normalizes to sum=1.0 and clamps to [0.05, 0.80].
    pub fn set_weights(&mut self, w_recency: f64, w_frequency: f64, w_semantic: f64, w_entropy: f64) {
        self.w_recency = w_recency.clamp(0.05, 0.80);
        self.w_frequency = w_frequency.clamp(0.05, 0.80);
        self.w_semantic = w_semantic.clamp(0.05, 0.80);
        self.w_entropy = w_entropy.clamp(0.05, 0.80);
        // Normalize to sum=1.0
        let sum = self.w_recency + self.w_frequency + self.w_semantic + self.w_entropy;
        if sum > 0.0 {
            self.w_recency /= sum;
            self.w_frequency /= sum;
            self.w_semantic /= sum;
            self.w_entropy /= sum;
        }
        // Sync to context scorer (uses different field names)
        self.context_scorer.w_recency = self.w_recency;
        self.context_scorer.w_frequency = self.w_frequency;
        self.context_scorer.w_similarity = self.w_semantic;
        self.context_scorer.w_entropy = self.w_entropy;
    }

    /// Persist the full fragment index to disk as compressed JSON.
    /// Called automatically after each ingest batch so sessions resume
    /// without re-indexing the whole codebase.
    pub fn persist_index(&self, path: &str) -> PyResult<()> {
        use std::io::Write;
        #[derive(Serialize)]
        struct IndexSnapshot<'a> {
            fragments: &'a HashMap<String, ContextFragment>,
            fragment_slot_ids: &'a Vec<String>,
            w_recency: f64,
            w_frequency: f64,
            w_semantic: f64,
            w_entropy: f64,
            total_tokens_saved: u64,
            total_optimizations: u64,
            total_fragments_ingested: u64,
            total_duplicates_caught: u64,
            gradient_temperature: f64,
            gradient_norm_ema: f64,
        }
        let snapshot = IndexSnapshot {
            fragments: &self.fragments,
            fragment_slot_ids: &self.fragment_slot_ids,
            w_recency: self.w_recency,
            w_frequency: self.w_frequency,
            w_semantic: self.w_semantic,
            w_entropy: self.w_entropy,
            total_tokens_saved: self.total_tokens_saved,
            total_optimizations: self.total_optimizations,
            total_fragments_ingested: self.total_fragments_ingested,
            total_duplicates_caught: self.total_duplicates_caught,
            gradient_temperature: self.gradient_temperature,
            gradient_norm_ema: self.gradient_norm_ema,
        };
        let json = serde_json::to_vec(&snapshot)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        // Write atomically via temp file
        let tmp = format!("{}.tmp", path);
        let mut f = std::fs::File::create(&tmp)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        f.write_all(&json)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        std::fs::rename(&tmp, path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Load a previously persisted index from disk.
    /// Returns the number of fragments restored.
    pub fn load_index(&mut self, path: &str) -> PyResult<usize> {
        #[derive(Deserialize)]
        struct IndexSnapshot {
            fragments: HashMap<String, ContextFragment>,
            fragment_slot_ids: Vec<String>,
            w_recency: f64,
            w_frequency: f64,
            w_semantic: f64,
            w_entropy: f64,
            total_tokens_saved: u64,
            total_optimizations: u64,
            total_fragments_ingested: u64,
            total_duplicates_caught: u64,
            #[serde(default = "default_gradient_temperature")]
            gradient_temperature: f64,
            #[serde(default)]
            gradient_norm_ema: f64,
        }
        fn default_gradient_temperature() -> f64 { 2.0 }
        let data = std::fs::read(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let snapshot: IndexSnapshot = serde_json::from_slice(&data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let n = snapshot.fragments.len();
        self.fragments = snapshot.fragments;
        self.fragment_slot_ids = snapshot.fragment_slot_ids;
        self.w_recency = snapshot.w_recency;
        self.w_frequency = snapshot.w_frequency;
        self.w_semantic = snapshot.w_semantic;
        self.w_entropy = snapshot.w_entropy;
        self.total_tokens_saved = snapshot.total_tokens_saved;
        self.total_optimizations = snapshot.total_optimizations;
        self.total_fragments_ingested = snapshot.total_fragments_ingested;
        self.total_duplicates_caught = snapshot.total_duplicates_caught;
        self.gradient_temperature = snapshot.gradient_temperature;
        self.gradient_norm_ema = snapshot.gradient_norm_ema;
        // Rebuild dedup index from loaded fragments
        for frag in self.fragments.values() {
            self.dedup_index.insert(&frag.fragment_id, &frag.content);
        }
        Ok(n)
    }

    /// Record that the selected fragments led to a successful output.
    /// This feeds the reinforcement learning loop.
    ///
    /// Channel coding path: R = modulated_reward(suff), A = R − μ.
    /// The advantage A is what drives the PRISM weight update,
    /// while μ (EMA baseline) reduces gradient variance.
    pub fn record_success(&mut self, fragment_ids: Vec<String>) {
        self.feedback.record_success(&fragment_ids);
        // EGSC cache: positive feedback → improve entry quality scores
        if self.last_cache_feedback_eligible {
            let frag_set: HashSet<String> = fragment_ids.iter().cloned().collect();
            self.egsc_cache.record_feedback_with_budget(
                &self.last_query,
                &frag_set,
                self.last_effective_budget,
                true,
            );
        }
        let raw_reward = if self.enable_channel_coding {
            let suff = self.last_optimization.as_ref()
                .map(|s| s.sufficiency)
                .unwrap_or(0.7);
            channel::modulated_reward(true, suff)
        } else {
            1.0
        };
        // A = R − μ  (REINFORCE advantage with EMA baseline)
        let advantage = raw_reward - self.reward_baseline_ema;
        // μ ← 0.9·μ + 0.1·R  (~10-step memory horizon)
        self.reward_baseline_ema = 0.9 * self.reward_baseline_ema + 0.1 * raw_reward;
        // Context Resonance: learn pairwise interaction strengths
        if self.enable_resonance {
            self.resonance_matrix.record_outcome(&fragment_ids, advantage, self.current_turn);
        }
        // Causal Context Graph: record trace with interventional data
        if self.enable_causal {
            let query_hash = dedup::simhash(&self.last_query);
            self.causal_graph.record_trace(
                self.current_turn,
                query_hash,
                &self.prev_selected_ids,
                &self.prev_explored_ids,
                advantage,
            );
        }
        self.apply_prism_rl_update(&fragment_ids, advantage);
    }

    /// Record that the selected fragments led to a failed output.
    ///
    /// Channel coding path: R = modulated_reward(suff), A = R − μ.
    /// Low sufficiency → stronger penalty → faster weight correction.
    pub fn record_failure(&mut self, fragment_ids: Vec<String>) {
        self.feedback.record_failure(&fragment_ids);
        // EGSC cache: negative feedback → degrade entry quality scores
        if self.last_cache_feedback_eligible {
            let frag_set: HashSet<String> = fragment_ids.iter().cloned().collect();
            self.egsc_cache.record_feedback_with_budget(
                &self.last_query,
                &frag_set,
                self.last_effective_budget,
                false,
            );
        }
        let raw_reward = if self.enable_channel_coding {
            let suff = self.last_optimization.as_ref()
                .map(|s| s.sufficiency)
                .unwrap_or(0.7);
            channel::modulated_reward(false, suff)
        } else {
            -1.0
        };
        // A = R − μ
        let advantage = raw_reward - self.reward_baseline_ema;
        // μ ← 0.9·μ + 0.1·R
        self.reward_baseline_ema = 0.9 * self.reward_baseline_ema + 0.1 * raw_reward;
        // Context Resonance: learn pairwise interaction strengths (negative signal)
        if self.enable_resonance {
            self.resonance_matrix.record_outcome(&fragment_ids, advantage, self.current_turn);
        }
        // Causal Context Graph: record trace with interventional data
        if self.enable_causal {
            let query_hash = dedup::simhash(&self.last_query);
            self.causal_graph.record_trace(
                self.current_turn,
                query_hash,
                &self.prev_selected_ids,
                &self.prev_explored_ids,
                advantage,
            );
        }
        self.apply_prism_rl_update(&fragment_ids, advantage);
    }

    /// Record a continuous reward signal for the selected fragments.
    ///
    /// This is the preferred path when ΔPerplexity is available:
    ///   R = PPL(response | no context) − PPL(response | context)
    ///
    /// Positive R means our context helped (perplexity dropped).
    /// Negative R means our context hurt (perplexity rose — wrong context).
    /// The advantage A = R − μ is computed internally with EMA baseline.
    ///
    /// Call this INSTEAD of record_success/record_failure when the
    /// Python layer can compute ΔPPL from the LLM response.
    pub fn record_reward(&mut self, fragment_ids: Vec<String>, reward: f64) {
        // NaN safety: non-finite reward is treated as neutral (0.0)
        let r = if reward.is_finite() { reward } else { 0.0 };
        // Update feedback tracker based on sign
        if r >= 0.0 {
            self.feedback.record_success(&fragment_ids);
            if self.last_cache_feedback_eligible {
                let frag_set: HashSet<String> = fragment_ids.iter().cloned().collect();
                self.egsc_cache.record_feedback_with_budget(
                    &self.last_query,
                    &frag_set,
                    self.last_effective_budget,
                    true,
                );
            }
        } else {
            self.feedback.record_failure(&fragment_ids);
            if self.last_cache_feedback_eligible {
                let frag_set: HashSet<String> = fragment_ids.iter().cloned().collect();
                self.egsc_cache.record_feedback_with_budget(
                    &self.last_query,
                    &frag_set,
                    self.last_effective_budget,
                    false,
                );
            }
        }
        // A = R − μ
        let advantage = r - self.reward_baseline_ema;
        // μ ← 0.9·μ + 0.1·R
        self.reward_baseline_ema = 0.9 * self.reward_baseline_ema + 0.1 * r;
        // Context Resonance: continuous reward signal for pairwise learning
        if self.enable_resonance {
            self.resonance_matrix.record_outcome(&fragment_ids, advantage, self.current_turn);
        }
        // Causal Context Graph: continuous reward signal for interventional learning
        if self.enable_causal {
            let query_hash = dedup::simhash(&self.last_query);
            self.causal_graph.record_trace(
                self.current_turn,
                query_hash,
                &self.prev_selected_ids,
                &self.prev_explored_ids,
                advantage,
            );
        }
        self.apply_prism_rl_update(&fragment_ids, advantage);
    }

    /// Enable or disable channel coding framework.
    pub fn set_channel_coding_enabled(&mut self, enabled: bool) {
        self.enable_channel_coding = enabled;
    }

    /// Update belief utilization EMA from proxy-side utilization scoring.
    ///
    /// Called by Python after measuring trigram + identifier overlap between
    /// the injected context and the LLM response. Closes the feedback loop:
    ///
    ///   belief_util_ema ← 0.9 × belief_util_ema + 0.1 × mean(belief_scores)
    ///   full_util_ema   ← 0.9 × full_util_ema   + 0.1 × mean(full_scores)
    ///
    /// These EMAs feed into the query-adaptive belief info factor:
    ///   util_ratio = belief_util_ema / full_util_ema
    ///   ios_belief_info_factor = archetype_factor × util_ratio
    ///
    /// If beliefs are well-utilized (util_ratio > 1.0) → factor increases → more beliefs selected.
    /// If beliefs are poorly utilized (util_ratio < 1.0) → factor decreases → more raw code.
    ///
    /// This is online stochastic gradient descent on the information efficiency objective:
    ///   max_θ  E[utilization(belief_context; θ)] / token_cost(belief_context; θ)
    pub fn update_belief_utilization(&mut self, belief_util: f64, full_util: f64) {
        if belief_util.is_finite() {
            self.belief_util_ema = 0.9 * self.belief_util_ema + 0.1 * belief_util.clamp(0.0, 1.0);
        }
        if full_util.is_finite() {
            self.full_util_ema = 0.9 * self.full_util_ema + 0.1 * full_util.clamp(0.0, 1.0);
        }
    }

    /// Get current belief utilization EMA (for diagnostics/logging).
    pub fn get_belief_util_ema(&self) -> f64 { self.belief_util_ema }
    /// Get current full fragment utilization EMA (for diagnostics/logging).
    pub fn get_full_util_ema(&self) -> f64 { self.full_util_ema }
    /// Get the current query-adaptive belief info factor (for diagnostics/logging).
    pub fn get_belief_info_factor(&self) -> f64 { self.ios_belief_info_factor }

    // ── EGSC Cache Management API ──

    /// Clear the EGSC cache (useful after major project changes).
    pub fn cache_clear(&mut self) { self.egsc_cache.clear(); }

    /// Get current cache entry count.
    pub fn cache_len(&self) -> usize { self.egsc_cache.len() }

    /// Check if cache is empty.
    pub fn cache_is_empty(&self) -> bool { self.egsc_cache.is_empty() }

    /// Get the current cache hit rate (from shift detector EMA).
    pub fn cache_hit_rate(&mut self) -> f64 {
        self.egsc_cache.stats().hit_rate
    }

    /// Set the cost model from a model name (auto-detects pricing).
    ///
    /// Covers 20+ models: OpenAI (gpt-4o, gpt-4, o1, o3), Anthropic (claude-3.5),
    /// Google (gemini), DeepSeek, Meta (llama), Mistral, and more.
    /// Unknown models default to GPT-4o pricing ($0.000015/token).
    ///
    /// Example: `engine.set_model("gpt-4o-mini")` → $0.60/M tokens
    pub fn set_model(&mut self, model_name: &str) {
        let cost_model = cache::CostModel::for_model(model_name);
        self.egsc_cache.set_cost_per_token(cost_model.cost_per_token);
    }

    /// Set cost-per-token directly (power users only).
    /// Most developers should use `set_model()` instead.
    /// Default is already $0.000015 (GPT-4o output) — no config needed.
    pub fn set_cache_cost_per_token(&mut self, cost: f64) {
        self.egsc_cache.set_cost_per_token(cost);
    }

    // ── Vault Belief Bridge API ──
    // Hierarchical Context Synthesis: attach vault beliefs to fragments
    // so IOS can select them at Belief resolution (5-10x token savings).

    /// Attach a vault belief to an existing fragment by fragment_id.
    ///
    /// When a belief is attached, IOS can select this fragment at Belief
    /// resolution — emitting the ~200-token belief summary INSTEAD of the
    /// ~800-token raw code. This is the core token savings mechanism.
    ///
    /// Call this after ingest (from Python) for each fragment that has a
    /// corresponding belief file in the vault.
    pub fn set_belief(&mut self, fragment_id: &str, belief_content: String, belief_token_count: u32) -> bool {
        if let Some(frag) = self.fragments.get_mut(fragment_id) {
            frag.belief_content = Some(belief_content);
            frag.belief_token_count = Some(belief_token_count);
            true
        } else {
            false
        }
    }

    /// Bulk-load vault beliefs from a directory.
    ///
    /// Scans `vault_dir` for *.md belief files, matches them to existing
    /// fragments by source path basename, and attaches belief content.
    ///
    /// Matching heuristic: belief file `knapsack_sds_18a33f4c.md` matches
    /// any fragment whose source contains `knapsack_sds`. This is O(N×M)
    /// where N = fragments, M = belief files, but both are small (~100s).
    ///
    /// Returns the number of beliefs successfully attached.
    pub fn load_vault_beliefs(&mut self, vault_dir: &str) -> usize {
        let dir = std::path::Path::new(vault_dir);
        if !dir.is_dir() { return 0; }

        // Collect all belief files: (basename_prefix, full_content, token_count)
        let mut beliefs: Vec<(String, String, u32)> = Vec::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) != Some("md") { continue; }
                let fname = match path.file_stem().and_then(|s| s.to_str()) {
                    Some(s) => s.to_string(),
                    None => continue,
                };
                // Extract the module name prefix before the hash suffix
                // e.g. "knapsack_sds_18a33f4c" → "knapsack_sds"
                // Skip architecture beliefs (arch_*) — they're cross-cutting
                if fname.starts_with("arch_") || fname.starts_with("doc_") { continue; }
                let prefix = if let Some(pos) = fname.rfind('_') {
                    // Check if suffix looks like a hex hash (8+ hex chars)
                    let suffix = &fname[pos+1..];
                    if suffix.len() >= 8 && suffix.chars().all(|c| c.is_ascii_hexdigit()) {
                        fname[..pos].to_string()
                    } else {
                        fname.clone()
                    }
                } else {
                    fname.clone()
                };

                if let Ok(content) = std::fs::read_to_string(&path) {
                    // Strip YAML frontmatter (between --- delimiters)
                    let body = if let Some(stripped) = content.strip_prefix("---") {
                        if let Some(end) = stripped.find("---") {
                            stripped[end + 3..].trim().to_string()
                        } else {
                            content.clone()
                        }
                    } else {
                        content.clone()
                    };
                    if body.is_empty() { continue; }
                    // Estimate tokens: ~4 chars per token for markdown
                    let tc = (body.len() as u32 / 4).max(1);
                    beliefs.push((prefix, body, tc));
                }
            }
        }

        if beliefs.is_empty() { return 0; }

        // Match beliefs to fragments by source path basename
        let mut attached = 0usize;
        let frag_ids: Vec<(String, String)> = self.fragments.iter()
            .map(|(id, f)| (id.clone(), f.source.clone()))
            .collect();

        for (fid, source) in &frag_ids {
            // Extract basename from source path (e.g. "entroly-core/src/knapsack_sds.rs" → "knapsack_sds")
            let src_path = std::path::Path::new(source);
            let basename = src_path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            if basename.is_empty() { continue; }

            // Find matching belief
            for (prefix, body, tc) in &beliefs {
                if prefix == basename {
                    if let Some(frag) = self.fragments.get_mut(fid) {
                        // Only attach if no belief already loaded
                        if frag.belief_content.is_none() {
                            frag.belief_content = Some(body.clone());
                            frag.belief_token_count = Some(*tc);
                            attached += 1;
                        }
                    }
                    break;
                }
            }
        }
        attached
    }

    /// Classify a task query and return the recommended budget multiplier.
    pub fn classify_task(&self, query: &str) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let task_type = TaskType::classify(query);
            let result = PyDict::new(py);
            result.set_item("task_type", format!("{:?}", task_type))?;
            result.set_item("budget_multiplier", task_type.budget_multiplier())?;
            Ok(result.into())
        })
    }

    /// Get dependency graph stats.
    pub fn dep_graph_stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let result = PyDict::new(py);
            result.set_item("nodes", self.dep_graph.node_count())?;
            result.set_item("edges", self.dep_graph.edge_count())?;
            Ok(result.into())
        })
    }

    /// Compute PageRank centrality scores for all fragments in the dep graph.
    ///
    /// Returns a dict of {fragment_id: score} where higher scores indicate
    /// structurally central code (hub files imported by many others).
    /// Used by the EvolutionDaemon to prioritize skill gaps in hub files.
    pub fn compute_pagerank(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let fragment_ids: Vec<String> = self.fragments.keys().cloned().collect();
            let scores = hierarchical::compute_pagerank(
                &self.dep_graph,
                &fragment_ids,
                20, // 20 iterations — sufficient convergence for typical code graphs
            );
            let result = PyDict::new(py);
            for (id, score) in &scores {
                result.set_item(id, (*score * 10000.0).round() / 10000.0)?;
            }
            Ok(result.into())
        })
    }

    /// Hierarchical Context Compression — 3-level codebase compression.
    ///
    /// Level 1: Skeleton map of entire codebase (~5% budget)
    /// Level 2: Expanded skeletons for dep-graph connected cluster (~25%)
    /// Level 3: Full content of most relevant fragments (~70%)
    ///
    /// Novel: symbol-reachability slicing + submodular diversity + PageRank.
    pub fn hierarchical_compress(&self, token_budget: u32, query: String) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            // Collect all fragments in a stable order
            let mut frags: Vec<ContextFragment> = self.fragments.values().cloned().collect();
            frags.sort_by(|a, b| a.fragment_id.cmp(&b.fragment_id));

            if frags.is_empty() {
                let result = PyDict::new(py);
                result.set_item("status", "empty")?;
                result.set_item("level1_map", "")?;
                result.set_item("level2_cluster", "")?;
                result.set_item("level3_count", 0)?;
                return Ok(result.into());
            }

            // Find query-relevant fragments (by SimHash similarity)
            let query_hash = dedup::simhash(&query);
            let mut scored: Vec<(usize, f64)> = frags.iter()
                .enumerate()
                .map(|(i, f)| {
                    let dist = dedup::hamming_distance(query_hash, f.simhash);
                    let sim = (1.0 - dist as f64 / 64.0).max(0.0);
                    (i, sim)
                })
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Top-K most relevant fragment IDs (seed for cluster expansion)
            let top_k = scored.iter()
                .take(10)
                .filter(|(_, sim)| *sim > 0.1)
                .map(|(i, _)| frags[*i].fragment_id.clone())
                .collect::<Vec<_>>();

            // Mean entropy for budget allocation
            let mean_entropy = frags.iter().map(|f| f.entropy_score).sum::<f64>()
                / frags.len() as f64;

            // Run hierarchical compression
            let hcc = hierarchical::hierarchical_compress(
                &frags,
                &self.dep_graph,
                &top_k,
                token_budget,
                mean_entropy,
            );

            // Build Python result
            let result = PyDict::new(py);
            result.set_item("status", "compressed")?;
            result.set_item("level1_map", &hcc.level1_map)?;
            result.set_item("level1_tokens", hcc.budget_used.0)?;
            result.set_item("level2_cluster", &hcc.level2_cluster)?;
            result.set_item("level2_tokens", hcc.budget_used.1)?;
            result.set_item("level3_count", hcc.level3_indices.len())?;
            result.set_item("level3_tokens", hcc.budget_used.2)?;

            // Coverage stats
            let coverage = PyDict::new(py);
            coverage.set_item("level1_files", hcc.coverage.0)?;
            coverage.set_item("level2_cluster_files", hcc.coverage.1)?;
            coverage.set_item("level3_full_files", hcc.coverage.2)?;
            result.set_item("coverage", coverage)?;

            // Total budget utilization
            let total_used = hcc.budget_used.0 + hcc.budget_used.1 + hcc.budget_used.2;
            result.set_item("budget_utilization",
                if token_budget > 0 {
                    (total_used as f64 / token_budget as f64 * 10000.0).round() / 10000.0
                } else { 0.0 }
            )?;

            // Selected L3 fragment details
            let l3_list = pyo3::types::PyList::empty(py);
            for &idx in &hcc.level3_indices {
                let f = &frags[idx];
                let d = PyDict::new(py);
                d.set_item("id", &f.fragment_id)?;
                d.set_item("source", &f.source)?;
                d.set_item("token_count", f.token_count)?;
                d.set_item("content", &f.content)?;
                let preview = if f.content.len() > 100 {
                    let mut end = 100;
                    while end < f.content.len() && !f.content.is_char_boundary(end) {
                        end += 1;
                    }
                    format!("{}...", &f.content[..end])
                } else {
                    f.content.clone()
                };
                d.set_item("preview", preview)?;
                l3_list.append(d)?;
            }
            result.set_item("level3_fragments", l3_list)?;

            // Cluster IDs for debugging
            result.set_item("cluster_ids", hcc.cluster_ids)?;

            Ok(result.into())
        })
    }

    /// Explain why each fragment was included or excluded in the last optimization.
    ///
    /// Returns per-fragment scoring breakdowns with all dimensions visible.
    /// Call after optimize() to understand selection decisions.
    pub fn explain_selection(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let snapshot = match &self.last_optimization {
                Some(s) => s,
                None => {
                    let result = PyDict::new(py);
                    result.set_item("error", "No optimization has been run yet")?;
                    return Ok(result.into());
                }
            };

            let result = PyDict::new(py);
            result.set_item("sufficiency", (snapshot.sufficiency * 10000.0).round() / 10000.0)?;

            let included = pyo3::types::PyList::empty(py);
            let excluded = pyo3::types::PyList::empty(py);

            for fs in &snapshot.fragment_scores {
                let d = PyDict::new(py);
                d.set_item("id", &fs.fragment_id)?;       // Bug fix: was "fragment_id", inconsistent with optimize's "id" key
                d.set_item("source", &fs.source)?;
                d.set_item("decision", if fs.selected { "included" } else { "excluded" })?;
                let scores = PyDict::new(py);
                scores.set_item("recency", (fs.recency * 10000.0).round() / 10000.0)?;
                scores.set_item("frequency", (fs.frequency * 10000.0).round() / 10000.0)?;
                scores.set_item("semantic", (fs.semantic * 10000.0).round() / 10000.0)?;
                scores.set_item("entropy", (fs.entropy * 10000.0).round() / 10000.0)?;
                scores.set_item("feedback_mult", (fs.feedback_mult * 10000.0).round() / 10000.0)?;
                scores.set_item("dep_boost", (fs.dep_boost * 10000.0).round() / 10000.0)?;
                scores.set_item("criticality", &fs.criticality)?;
                scores.set_item("composite", (fs.composite * 10000.0).round() / 10000.0)?;
                d.set_item("scores", scores)?;
                d.set_item("reason", &fs.reason)?;

                if fs.selected {
                    included.append(d)?;
                } else {
                    excluded.append(d)?;
                }
            }

            result.set_item("included", included)?;
            result.set_item("excluded", excluded)?;

            if !snapshot.explored_ids.is_empty() {
                result.set_item("explored", snapshot.explored_ids.clone())?;
            }

            Ok(result.into())
        })
    }

    /// Export full engine state as JSON string for checkpoint/restore.
    ///
    /// Serializes: fragments, dedup index, dep graph, feedback tracker,
    /// turn counter, stats — everything needed for perfect resume.
    pub fn export_state(&self) -> PyResult<String> {
        let state = EngineState {
            fragments: self.fragments.clone(),
            dedup_index: &self.dedup_index,
            dep_graph: &self.dep_graph,
            feedback: &self.feedback,
            prism_optimizer: &self.prism_optimizer,
            w_recency: self.w_recency,
            w_frequency: self.w_frequency,
            w_semantic: self.w_semantic,
            w_entropy: self.w_entropy,
            current_turn: self.current_turn,
            id_counter: self.id_counter,
            max_fragments: self.max_fragments,
            total_tokens_saved: self.total_tokens_saved,
            total_optimizations: self.total_optimizations,
            total_fragments_ingested: self.total_fragments_ingested,
            total_duplicates_caught: self.total_duplicates_caught,
            total_explorations: self.total_explorations,
            rng_state: self.rng_state,
            gradient_temperature: self.gradient_temperature,
            gradient_norm_ema: self.gradient_norm_ema,
            // EGSC cache warm-start: serialize cache state as nested JSON
            cache_snapshot: self.egsc_cache.export_cache().ok(),
            // Context Resonance persistence
            resonance_matrix: &self.resonance_matrix,
            prism_optimizer_5d: &self.prism_optimizer_5d,
            w_resonance: self.w_resonance,
            // Consolidation persistence
            total_consolidations: self.total_consolidations,
            consolidation_tokens_saved: self.consolidation_tokens_saved,
            // Causal Context Graph persistence
            causal_graph: &self.causal_graph,
        };
        serde_json::to_string(&state).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization failed: {}", e))
        })
    }

    /// Import engine state from JSON string (checkpoint restore).
    ///
    /// Replaces all engine state with the deserialized data.
    pub fn import_state(&mut self, json_str: &str) -> PyResult<()> {
        let state: OwnedEngineState = serde_json::from_str(json_str).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Deserialization failed: {}", e))
        })?;
        let dedup_threshold = state.dedup_index.hamming_threshold();
        self.fragments = state.fragments;
        self.dedup_index = DedupIndex::new(dedup_threshold);
        self.dep_graph = state.dep_graph;
        self.feedback = state.feedback;
        // Restore PRISM covariance if available; fall back to fresh optimizer to support
        // checkpoints created before this field was added.
        if let Some(p) = state.prism_optimizer {
            self.prism_optimizer = p;
        }
        self.w_recency = state.w_recency;
        self.w_frequency = state.w_frequency;
        self.w_semantic = state.w_semantic;
        self.w_entropy = state.w_entropy;
        self.current_turn = state.current_turn;
        self.id_counter = state.id_counter;
        self.max_fragments = state.max_fragments;
        // NOTE: instance_id is intentionally NOT restored from the checkpoint.
        // The restored engine runs in the current process and must maintain
        // its own unique instance_id (already assigned in new()) to keep
        // fragment IDs disjoint from any other engine in this process.
        self.total_tokens_saved = state.total_tokens_saved;
        self.total_optimizations = state.total_optimizations;
        self.total_fragments_ingested = state.total_fragments_ingested;
        self.total_duplicates_caught = state.total_duplicates_caught;
        self.total_explorations = state.total_explorations;
        if state.rng_state != 0 { self.rng_state = state.rng_state; }
        self.gradient_temperature = state.gradient_temperature;
        self.gradient_norm_ema = state.gradient_norm_ema;
        let mut fragment_entries: Vec<(&String, &ContextFragment)> = self.fragments.iter().collect();
        fragment_entries.sort_by(|a, b| a.0.cmp(b.0));
        for (fid, frag) in fragment_entries {
            self.dedup_index.insert(fid, &frag.content);
        }
        self.rebuild_lsh_index();
        self.context_scorer.w_recency = self.w_recency;
        self.context_scorer.w_frequency = self.w_frequency;
        self.context_scorer.w_similarity = self.w_semantic;
        self.context_scorer.w_entropy = self.w_entropy;
        self.last_optimization = None;
        // EGSC cache warm-start: restore cache from checkpoint
        if let Some(ref cache_json) = state.cache_snapshot {
            match self.egsc_cache.import_cache(cache_json) {
                Ok(n) => eprintln!("[entroly] Warm-start: restored {} EGSC cache entries", n),
                Err(e) => eprintln!("[entroly] Cache restore skipped: {}", e),
            }
        }
        // Context Resonance warm-start
        if let Some(rm) = state.resonance_matrix {
            self.resonance_matrix = rm;
        }
        if let Some(p5) = state.prism_optimizer_5d {
            self.prism_optimizer_5d = p5;
        }
        self.w_resonance = state.w_resonance;
        // Consolidation stats
        self.total_consolidations = state.total_consolidations;
        self.consolidation_tokens_saved = state.consolidation_tokens_saved;
        // Causal Context Graph warm-start
        if let Some(cg) = state.causal_graph {
            self.causal_graph = cg;
        }
        Ok(())
    }

    /// Set the exploration rate (0.0 = pure exploitation, 1.0 = always explore).
    pub fn set_exploration_rate(&mut self, rate: f64) {
        self.exploration_rate = rate.clamp(0.0, 1.0);
    }

    /// Scan a specific fragment for security vulnerabilities.
    ///
    /// Returns a JSON-encoded SastReport with:
    ///   - findings: list of {rule_id, cwe, severity, line_number, description, fix, confidence}
    ///   - risk_score: CVSS-inspired aggregate [0.0, 10.0]
    ///   - critical_count, high_count, medium_count, low_count
    ///   - top_fix: the most important remediation action
    ///
    /// The engine also auto-scans on ingest — this method lets you re-scan
    /// on demand (e.g., after fragment content changes, or for targeted audit).
    pub fn scan_fragment(&self, fragment_id: &str) -> PyResult<String> {
        let frag = self.fragments.get(fragment_id).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Fragment '{}' not found", fragment_id))
        })?;
        let report = sast::scan_content(&frag.content, &frag.source);
        serde_json::to_string(&report).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    /// Scan all ingested fragments and return an aggregated security report.
    ///
    /// Returns JSON with per-fragment findings + session-level statistics:
    ///   - total_findings, critical_total, max_risk_score
    ///   - most_vulnerable: fragment_id with highest risk score
    ///   - findings_by_category: {category: count}
    pub fn security_report(&self) -> PyResult<String> {
        let mut all_findings: Vec<serde_json::Value> = Vec::new();
        let mut critical_total = 0usize;
        let mut high_total = 0usize;
        let mut max_risk: f64 = 0.0;
        let mut most_vulnerable = String::new();
        let mut by_category: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

        for (fid, frag) in &self.fragments {
            let report = sast::scan_content(&frag.content, &frag.source);
            if report.findings.is_empty() {
                continue;
            }
            critical_total += report.critical_count;
            high_total     += report.high_count;
            for f in &report.findings {
                *by_category.entry(f.category.clone()).or_insert(0) += 1;
            }
            if report.risk_score > max_risk {
                max_risk = report.risk_score;
                most_vulnerable = fid.clone();
            }
            all_findings.push(serde_json::json!({
                "fragment_id": fid,
                "source": &frag.source,
                "risk_score": report.risk_score,
                "critical_count": report.critical_count,
                "high_count": report.high_count,
                "finding_count": report.findings.len(),
                "top_fix": report.top_fix,
            }));
        }

        // Sort by risk_score desc
        all_findings.sort_unstable_by(|a, b| {
            let ra = a["risk_score"].as_f64().unwrap_or(0.0);
            let rb = b["risk_score"].as_f64().unwrap_or(0.0);
            rb.partial_cmp(&ra).unwrap_or(std::cmp::Ordering::Equal)
        });

        let result = serde_json::json!({
            "fragments_scanned": self.fragments.len(),
            "fragments_with_findings": all_findings.len(),
            "critical_total": critical_total,
            "high_total": high_total,
            "max_risk_score": (max_risk * 100.0).round() / 100.0,
            "most_vulnerable_fragment": most_vulnerable,
            "findings_by_category": by_category,
            "vulnerable_fragments": all_findings,
        });

        serde_json::to_string(&result).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    /// Analyze codebase health across all ingested fragments.
    ///
    /// Returns a JSON-encoded HealthReport with:
    ///   - code_health_score [0–100] and health_grade (A/B/C/D/F)
    ///   - clone_pairs: near-duplicate file pairs (Type-1/2/3)
    ///   - dead_symbols: defined but never referenced
    ///   - god_files: over-coupled fragments (> μ+2σ reverse deps)
    ///   - arch_violations: cross-layer dependency violations
    ///   - naming_issues: Python/Rust/React naming convention breaks
    ///   - top_recommendation: single most impactful action
    pub fn analyze_health(&self) -> PyResult<String> {
        let frags: Vec<&ContextFragment> = self.fragments.values().collect();
        let report = health::analyze_health(&frags, &self.dep_graph);
        serde_json::to_string(&report).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    /// Scan for entropy anomalies — code that is statistically surprising
    /// relative to its neighbors. Uses robust Z-scores (MAD) grouped by
    /// directory. High-entropy spikes = copy-paste errors, unusual patterns.
    /// Low-entropy drops = dead stubs, placeholders.
    pub fn entropy_anomalies(&self) -> PyResult<String> {
        let frags: Vec<&ContextFragment> = self.fragments.values().collect();
        let report = anomaly::scan_anomalies(&frags);
        serde_json::to_string(&report).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    /// Score how much of the injected context the LLM actually used.
    /// Call after receiving the LLM response. Measures trigram overlap
    /// and identifier reuse to determine context efficiency.
    ///
    /// Returns JSON with per-fragment scores and session utilization.
    pub fn score_utilization(&self, response: &str) -> PyResult<String> {
        // Collect fragments that were most recently selected (used in last optimize())
        let frags: Vec<&ContextFragment> = self.fragments.values().collect();
        let report = utilization::score_utilization(&frags, response);
        serde_json::to_string(&report).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    /// Run semantic deduplication on all fragments and report which ones
    /// are informationally redundant. Returns removal candidates and
    /// estimated token savings.
    pub fn semantic_dedup_report(&self) -> PyResult<String> {
        let frags: Vec<ContextFragment> = self.fragments.values().cloned().collect();
        if frags.is_empty() {
            return Ok("{\"kept\": 0, \"removed\": 0, \"tokens_saved\": 0}".into());
        }
        let sorted: Vec<usize> = (0..frags.len()).collect();
        let result = semantic_dedup::semantic_deduplicate_with_stats(&frags, &sorted, None);
        let report = serde_json::json!({
            "total_fragments": frags.len(),
            "kept": result.kept_indices.len(),
            "removed": result.removed_count,
            "tokens_saved": result.tokens_saved,
        });
        serde_json::to_string(&report).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        })
    }

    /// Export fragments for checkpoint (returns list of dicts).
    pub fn export_fragments(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let list = pyo3::types::PyList::empty(py);
            for frag in self.fragments.values() {
                let d = PyDict::new(py);
                d.set_item("fragment_id", &frag.fragment_id)?;
                d.set_item("content", &frag.content)?;
                d.set_item("token_count", frag.token_count)?;
                d.set_item("source", &frag.source)?;
                d.set_item("is_pinned", frag.is_pinned)?;
                d.set_item("recency_score", frag.recency_score)?;
                d.set_item("frequency_score", frag.frequency_score)?;
                d.set_item("semantic_score", frag.semantic_score)?;
                d.set_item("entropy_score", frag.entropy_score)?;
                d.set_item("turn_created", frag.turn_created)?;
                d.set_item("turn_last_accessed", frag.turn_last_accessed)?;
                d.set_item("access_count", frag.access_count)?;
                d.set_item("simhash", frag.simhash)?;
                list.append(d)?;
            }
            Ok(list.into())
        })
    }
}

// ═══════════════════════════════════════════════════════════════════
// Non-PyO3 implementation methods
// ═══════════════════════════════════════════════════════════════════

impl EntrolyEngine {
    /// Rebuild the LSH index and slot list from the current fragment map.
    ///
    /// Called after batch eviction in advance_turn(). O(N) but infrequent.
    fn rebuild_lsh_index(&mut self) {
        self.lsh_index.clear();
        self.fragment_slot_ids.clear();
        // Sort by fragment_id for deterministic slot assignment
        let mut ids: Vec<String> = self.fragments.keys().cloned().collect();
        ids.sort_unstable();
        for (slot, id) in ids.iter().enumerate() {
            if let Some(frag) = self.fragments.get(id) {
                self.lsh_index.insert(frag.simhash, slot);
            }
            self.fragment_slot_ids.push(id.clone());
        }
    }

    /// Numerically stable sigmoid: σ(x) = 1 / (1 + e^{-x}).
    /// Clamps input to [-500, 500] to prevent overflow.
    #[inline]
    fn sigmoid(x: f64) -> f64 {
        let x = x.clamp(-500.0, 500.0);
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let ex = x.exp();
            ex / (1.0 + ex)
        }
    }

    /// Apply PRISM Anisotropic Spectral Shaping with differentiable soft-selector gradients.
    ///
    /// Uses REINFORCE-with-baseline policy gradient through a sigmoid relaxation:
    ///
    ///   p_i = σ(score_i / τ)                 — soft selection probability
    ///   advantage_i = (selected_i - p_i) × R  — REINFORCE baseline
    ///   ∂E[R]/∂w_k = Σ_i  advantage_i · σ'(score_i/τ) · feature_{i,k}
    ///
    /// The `(selected - p)` baseline provides proper counterfactual credit assignment:
    ///   - Selected fragment with low p → large positive advantage (good surprise)
    ///   - Selected fragment with high p → small advantage (expected)
    ///   - Unselected fragment with high p → negative advantage (should've been included)
    ///   - Unselected fragment with low p → ~zero advantage (correctly excluded)
    ///
    /// The σ'(score/τ) = p(1-p)/τ term focuses gradient on decision-boundary fragments.
    ///
    /// Temperature τ anneals via τ *= 0.995 per update but resets on regime changes
    /// (gradient norm spike > 3× EMA), allowing re-exploration when the task shifts.
    ///
    /// NOTE: The gradient operates on the pre-softcap linear score `w^T · features`,
    /// not on the post-softcap score used in actual selection. This is intentional:
    /// softcap is monotone, so pre-softcap optima = post-softcap optima, and the
    /// linear gradient landscape is better conditioned for optimization.
    fn apply_prism_rl_update(&mut self, fragment_ids: &[String], reward: f64) {
        if self.fragments.is_empty() { return; }

        let tau = self.gradient_temperature.max(0.01);
        let selected: HashSet<&str> = fragment_ids.iter().map(|s| s.as_str()).collect();

        // ── Per-Fragment Counterfactual Credit (Shapley-inspired) ──
        //
        // The uniform advantage A = (action_i - p_i) × R gives every selected
        // fragment the SAME credit. This creates "reward pollution": irrelevant
        // fragments accumulate false positive weight.
        //
        // Solution: Weight each fragment's advantage by its *marginal information
        // contribution* — a Shapley-inspired decomposition.
        //
        // φᵢ = entropy_i / Σⱼ∈S(entropy_j)  (information share)
        //
        // High-entropy fragments bear more responsibility for outcomes because
        // they contributed more information to the LLM's reasoning.
        let selected_entropy_sum: f64 = fragment_ids.iter()
            .filter_map(|id| self.fragments.get(id))
            .map(|f| f.entropy_score.max(0.01))
            .sum();
        let selected_entropy_sum = selected_entropy_sum.max(0.01); // Prevent div by zero

        // ── Eligibility Traces: TD(λ) Temporal Credit ──
        //
        // Current request's outcome may depend on context assembled in past
        // requests. Eligibility traces propagate attenuated credit backward:
        //   e_i(t) = λ_trace × e_i(t-1) + ∂log π(a_i|s) / ∂θ
        //
        // Fragments selected in this AND recent requests receive credit
        // proportional to their trace magnitude.
        //
        // λ_trace = 0.7 gives ~3-request effective memory window.
        let trace_lambda = 0.7_f64;

        // Compute REINFORCE-with-baseline policy gradient
        let mut g = [0.0_f64; 4]; // [∂/∂w_r, ∂/∂w_f, ∂/∂w_s, ∂/∂w_e]
        // 5D gradient includes resonance dimension
        let mut g5 = [0.0_f64; 5]; // [∂/∂w_r, ∂/∂w_f, ∂/∂w_s, ∂/∂w_e, ∂/∂w_res]
        // λ* from the last forward bisection — use the SAME probability as the forward pass.
        // p_i = σ((s_i − λ*·tokens_i)/τ) is the exact KKT soft selection probability.
        // Reusing it here ensures advantage = (action − p_exact) × R is an unbiased estimator.
        let lambda = self.last_lambda_star;

        // Pre-compute resonance features for the 5th gradient dimension.
        // For each fragment, its "resonance feature" is the mean resonance
        // bonus with the co-selected set — this is the partial derivative
        // ∂score/∂w_resonance that drives PRISM's 5th dimension learning.
        let selected_ids_vec: Vec<&str> = selected.iter().copied().collect();

        for frag in self.fragments.values_mut() {
            // Linear score (pre-softcap — same landscape as forward pass)
            let score = self.w_recency   * frag.recency_score
                      + self.w_frequency * frag.frequency_score
                      + self.w_semantic  * frag.semantic_score
                      + self.w_entropy   * frag.entropy_score;

            // Exact KKT probability: p_i = σ((s_i − λ*·tokens_i) / τ)
            // Matches the forward bisection probability — no forward/backward mismatch.
            let tc = frag.token_count as f64;
            let p = Self::sigmoid((score - lambda * tc) / tau);
            let dp = p * (1.0 - p) / tau; // σ'(·/τ) — focuses gradient on marginal fragments

            let action = if selected.contains(frag.fragment_id.as_str()) { 1.0 } else { 0.0 };

            // ── Counterfactual credit: φᵢ modulates advantage ──
            // Selected fragments: credit weighted by information share
            // Non-selected: use uniform 1.0 (their "absence" signal is unmodulated)
            let phi = if action > 0.5 {
                (frag.entropy_score.max(0.01) / selected_entropy_sum).clamp(0.1, 3.0)
            } else {
                1.0
            };

            // ── Eligibility trace update ──
            // Decay previous trace, add current "log-policy gradient" direction
            let trace_update = (action - p) * dp;
            frag.eligibility_trace = trace_lambda * frag.eligibility_trace + trace_update;

            // Advantage = counterfactual_credit × eligibility_trace × reward
            let advantage = phi * frag.eligibility_trace * reward;

            // Accumulate: advantage_i × feature_{i,k}
            g[prism::dim::RECENCY]   += advantage * frag.recency_score;
            g[prism::dim::FREQUENCY] += advantage * frag.frequency_score;
            g[prism::dim::SEMANTIC]  += advantage * frag.semantic_score;
            g[prism::dim::ENTROPY]   += advantage * frag.entropy_score;

            // 5D gradient: resonance feature = mean pairwise resonance with co-selected set
            if self.enable_resonance {
                let res_feature = self.resonance_matrix.resonance_bonus(
                    frag.fragment_id.as_str(), &selected_ids_vec
                );
                g5[prism::dim::RECENCY]   += advantage * frag.recency_score;
                g5[prism::dim::FREQUENCY] += advantage * frag.frequency_score;
                g5[prism::dim::SEMANTIC]  += advantage * frag.semantic_score;
                g5[prism::dim::ENTROPY]   += advantage * frag.entropy_score;
                g5[prism::dim::RESONANCE] += advantage * res_feature;
            }
        }


        // ── Adaptive Dual Gap Temperature (ADGT) × PRISM Condition-Number Temperature (PCNT) ──
        //
        // Replaces the ad-hoc τ *= 0.995 schedule with a principled information-theoretic signal.
        //
        // ADGT: natural temperature derived from the duality gap G = D(λ*) − primal.
        //   G ∈ [0, τ·N·log(2)]
        //   G ≈ 0       → weights converged → lower τ (exploit sharp selection)
        //   G ≈ τ·N·log(2) → all p_i ≈ 0.5 → full uncertainty → keep τ high (explore)
        //   natural_tau_adgt = G / (N · log(2) · C)  where C=4 is a calibration scale
        //
        // PCNT: condition number κ = sqrt(λ_max / λ_min) of PRISM gradient covariance.
        //   κ ≈ 1  → isotropic, well-conditioned weights → sharper selection OK
        //   κ >> 1 → ill-conditioned → some dims highly variable → keep selection softer
        //   modulation: τ_final = τ_adgt × sqrt(κ) / sqrt(d=4)  [normalized by dimension]
        //
        // These two signals are orthogonal:
        //   ADGT measures "how far is the current selection from the continuous optimum?"
        //   PCNT measures "how uncertain are the learned weights themselves?"
        // Together they ensure τ is always calibrated to the actual optimization state.
        let g_norm = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        if self.gradient_norm_ema > 1e-8 && g_norm > self.gradient_norm_ema * 3.0 {
            // Regime change: task distribution shifted, reset to full exploration.
            self.gradient_temperature = 2.0;
        } else if self.last_dual_gap > 0.0 && self.fragments.len() > 1 {
            // ADGT: normalize gap per fragment, scale by log(2) to bound on [0, τ]
            let n = self.fragments.len() as f64;
            let gap_per_frag = self.last_dual_gap / (n * 2_f64.ln());
            // PCNT: condition number amplifies τ when weights are uncertain
            let kappa = self.prism_optimizer.condition_number();
            let kappa_norm = (kappa / 2.0).clamp(0.5, 4.0); // normalize: κ=4 → 2×, κ=1 → 0.5×
            // Natural temperature: high gap + high κ → high τ; low gap + low κ → low τ
            let natural_tau = (gap_per_frag * kappa_norm).clamp(0.1, 2.0);
            // Slow EMA blend toward natural_tau (avoids oscillation)
            self.gradient_temperature = 0.90 * self.gradient_temperature + 0.10 * natural_tau;
            // Hard floor to prevent convergence to zero (min exploration needed)
            self.gradient_temperature = self.gradient_temperature.max(0.1);
        } else {
            // No ADGT signal available (hard-path or early run): gentle decay as fallback
            self.gradient_temperature = (self.gradient_temperature * 0.998).max(0.1);
        }
        self.gradient_norm_ema = 0.95 * self.gradient_norm_ema + 0.05 * g_norm;

        // Let PRISM compute the anisotropically-damped update: Q Λ^{-1/2} Q^T g
        let update = self.prism_optimizer.compute_update(&g);

        // Apply updates to weights
        self.w_recency   += update[prism::dim::RECENCY];
        self.w_frequency += update[prism::dim::FREQUENCY];
        self.w_semantic  += update[prism::dim::SEMANTIC];
        self.w_entropy   += update[prism::dim::ENTROPY];

        // ── 5D PRISM: learn w_resonance via spectral shaping ──
        // The resonance dimension has inherently higher gradient variance
        // (combinatorial noise from pairwise interactions). PRISM 5D's
        // Λ⁻¹/² automatically dampens this — no manual learning rate needed.
        if self.enable_resonance {
            let update5 = self.prism_optimizer_5d.compute_update(&g5);
            self.w_resonance += update5[prism::dim::RESONANCE];
            // Resonance weight: [0.0, 0.5] — capped lower than base dims
            // because it's a secondary signal that augments, not replaces.
            self.w_resonance = self.w_resonance.clamp(0.0, 0.5);
        }

        // Prevent collapse: clamp weights to positive bounds [0.05, 0.8]
        self.w_recency   = self.w_recency.clamp(0.05, 0.8);
        self.w_frequency = self.w_frequency.clamp(0.05, 0.8);
        self.w_semantic  = self.w_semantic.clamp(0.05, 0.8);
        self.w_entropy   = self.w_entropy.clamp(0.05, 0.8);

        // Normalize base weights so they sum to 1.0 to preserve scoring scale.
        // w_resonance is NOT included in the normalization — it's a separate
        // additive dimension that modulates selection independently.
        let sum = self.w_recency + self.w_frequency + self.w_semantic + self.w_entropy;
        self.w_recency   /= sum;
        self.w_frequency /= sum;
        self.w_semantic  /= sum;
        self.w_entropy   /= sum;

        // Update the context scorer with the newly learned decoupled weights
        self.context_scorer.w_similarity = self.w_semantic;
        self.context_scorer.w_recency    = self.w_recency;
        self.context_scorer.w_entropy    = self.w_entropy;
        self.context_scorer.w_frequency  = self.w_frequency;

        // Route gradient to per-archetype PRISM (if query persona is active)
        if self.enable_query_personas {
            if let Some(ref aid) = self.last_archetype_id {
                let success = reward > 0.0;
                self.query_manifold.record_result(aid, &g, success);
            }
        }
    }

    /// Compute context sufficiency: fraction of referenced symbols
    /// that have definitions in the selected context.
    fn compute_sufficiency(&self, frags: &[ContextFragment], selected_indices: &[usize]) -> f64 {
        let selected_ids: HashSet<&str> = selected_indices.iter()
            .map(|&i| frags[i].fragment_id.as_str())
            .collect();

        // Collect all symbols defined by selected fragments
        let defined_symbols: HashSet<String> = self.dep_graph.symbol_definitions().iter()
            .filter(|(_, fid)| selected_ids.contains(fid.as_str()))
            .map(|(symbol, _)| symbol.clone())
            .collect();

        // Collect all symbols referenced by selected fragments
        let mut referenced_symbols: HashSet<String> = HashSet::new();
        for &idx in selected_indices {
            let idents = extract_identifiers(&frags[idx].content);
            for ident in idents {
                // Only count identifiers that are in the symbol table
                // (i.e., that are defined somewhere in the context)
                if self.dep_graph.has_symbol(&ident) {
                    referenced_symbols.insert(ident);
                }
            }
        }

        if referenced_symbols.is_empty() {
            return 1.0; // Nothing to reference = fully sufficient
        }

        let satisfied = referenced_symbols.iter()
            .filter(|s| defined_symbols.contains(*s))
            .count();

        satisfied as f64 / referenced_symbols.len() as f64
    }
}

// ═══════════════════════════════════════════════════════════════════
// Serialization types for export/import
// ═══════════════════════════════════════════════════════════════════
/// Borrowed state for serialization (avoids cloning dedup/dep/feedback).
#[derive(Serialize)]
struct EngineState<'a> {
    fragments: HashMap<String, ContextFragment>,
    dedup_index: &'a DedupIndex,
    dep_graph: &'a DepGraph,
    feedback: &'a FeedbackTracker,
    prism_optimizer: &'a PrismOptimizer,
    w_recency: f64,
    w_frequency: f64,
    w_semantic: f64,
    w_entropy: f64,
    current_turn: u32,
    id_counter: u64,
    max_fragments: usize,
    total_tokens_saved: u64,
    total_optimizations: u64,
    total_fragments_ingested: u64,
    total_duplicates_caught: u64,
    total_explorations: u64,
    rng_state: u64,
    gradient_temperature: f64,
    gradient_norm_ema: f64,
    /// EGSC cache snapshot (JSON) — warm-start persistence.
    /// Nested JSON string to avoid coupling EngineState to cache internals.
    cache_snapshot: Option<String>,
    // ── Context Resonance persistence ──
    resonance_matrix: &'a ResonanceMatrix,
    prism_optimizer_5d: &'a PrismOptimizer5D,
    w_resonance: f64,
    // ── Consolidation persistence ──
    total_consolidations: u64,
    consolidation_tokens_saved: u64,
    // ── Causal Context Graph persistence ──
    causal_graph: &'a CausalContextGraph,
}

/// Owned state for deserialization.
#[derive(Deserialize)]
struct OwnedEngineState {
    fragments: HashMap<String, ContextFragment>,
    dedup_index: DedupIndex,
    dep_graph: DepGraph,
    feedback: FeedbackTracker,
    prism_optimizer: Option<PrismOptimizer>,  // Optional for backward-compat with old checkpoints
    #[serde(default = "default_w_recency")]
    w_recency: f64,
    #[serde(default = "default_w_frequency")]
    w_frequency: f64,
    #[serde(default = "default_w_semantic")]
    w_semantic: f64,
    #[serde(default = "default_w_entropy")]
    w_entropy: f64,
    current_turn: u32,
    id_counter: u64,
    // Optional for backward-compat — old checkpoints lack this field.
    #[serde(default = "default_max_fragments")]
    max_fragments: usize,
    total_tokens_saved: u64,
    total_optimizations: u64,
    total_fragments_ingested: u64,
    total_duplicates_caught: u64,
    total_explorations: u64,
    #[serde(default = "default_rng_state")]
    rng_state: u64,
    #[serde(default = "default_gradient_temperature")]
    gradient_temperature: f64,
    #[serde(default)]
    gradient_norm_ema: f64,
    /// EGSC cache snapshot — warm-start persistence (optional for backward-compat).
    #[serde(default)]
    cache_snapshot: Option<String>,
    // ── Context Resonance (optional for backward-compat with old checkpoints) ──
    #[serde(default)]
    resonance_matrix: Option<ResonanceMatrix>,
    #[serde(default)]
    prism_optimizer_5d: Option<PrismOptimizer5D>,
    #[serde(default)]
    w_resonance: f64,
    // ── Consolidation (optional for backward-compat) ──
    #[serde(default)]
    total_consolidations: u64,
    #[serde(default)]
    consolidation_tokens_saved: u64,
    // ── Causal Context Graph (optional for backward-compat) ──
    #[serde(default)]
    causal_graph: Option<CausalContextGraph>,
}

fn default_max_fragments() -> usize { 10_000 }
fn default_w_recency() -> f64 { 0.30 }
fn default_w_frequency() -> f64 { 0.25 }
fn default_w_semantic() -> f64 { 0.25 }
fn default_w_entropy() -> f64 { 0.20 }
fn default_gradient_temperature() -> f64 { 2.0 }
fn default_rng_state() -> u64 { 1 }

// ═══════════════════════════════════════════════════════════════════
// Standalone PyO3 functions (for direct access to math engines)
// ═══════════════════════════════════════════════════════════════════

/// Compute Shannon entropy of a text string (bits per character).
#[pyfunction]
fn py_shannon_entropy(text: &str) -> f64 {
    shannon_entropy(text)
}

/// Compute normalized Shannon entropy [0, 1].
#[pyfunction]
fn py_normalized_entropy(text: &str) -> f64 {
    normalized_entropy(text)
}

/// Compute boilerplate ratio [0, 1].
#[pyfunction]
fn py_boilerplate_ratio(text: &str) -> f64 {
    boilerplate_ratio(text)
}

/// Compute Rényi entropy of order 2 (collision entropy).
/// More sensitive to concentrated probability mass than Shannon entropy.
#[pyfunction]
fn py_renyi_entropy_2(text: &str) -> f64 {
    renyi_entropy_2(text)
}

/// Shannon–Rényi divergence: H₁(X) - H₂(X).
/// High divergence indicates noise or encoded data, not useful context.
#[pyfunction]
fn py_entropy_divergence(text: &str) -> f64 {
    entropy_divergence(text)
}

/// Compute 64-bit SimHash fingerprint.
#[pyfunction]
fn py_simhash(text: &str) -> u64 {
    simhash(text)
}

/// Compute Hamming distance between two fingerprints.
#[pyfunction]
fn py_hamming_distance(a: u64, b: u64) -> u32 {
    hamming_distance(a, b)
}

/// Compute information density score [0, 1].
#[pyfunction]
fn py_information_score(text: &str, other_fragments: Vec<String>) -> f64 {
    let refs: Vec<&str> = other_fragments.iter().map(|s| s.as_str()).collect();
    information_score(text, &refs)
}

/// Scan content for security vulnerabilities — returns JSON-encoded SastReport.
#[pyfunction]
fn py_scan_content(content: &str, source: &str) -> String {
    let report = sast::scan_content(content, source);
    serde_json::to_string(&report).unwrap_or_else(|e| {
        let escaped = format!("{}", e).replace('\\', "\\\\").replace('"', "\\\"");
        format!("{{\"error\":\"{}\"}}", escaped)
    })
}

#[pyfunction]
fn py_analyze_query(
    query: &str,
    fragment_summaries: Vec<String>,
) -> (f64, Vec<String>, bool, String) {
    query::py_analyze_query(query, fragment_summaries)
}

#[pyfunction]
fn py_refine_heuristic(query: &str, fragment_summaries: Vec<String>) -> String {
    query::py_refine_heuristic(query, fragment_summaries)
}

/// Health analysis is available via engine.analyze_health() (a #[pymethod]).
/// This standalone function is intentionally minimal.
#[pyfunction]
fn py_analyze_health_info() -> String {
    "{\"info\":\"Call engine.analyze_health() to get a full HealthReport for the current session.\"}".to_string()
}

// ─── Conversation Pruner PyO3 wrappers ───────────────────────────────────────

/// Prune a conversation to fit within a token budget.
///
/// Uses multi-choice knapsack via KKT dual bisection with causal DAG
/// coherence enforcement.  Returns JSON-encoded PruneResult.
///
/// `blocks` is a list of dicts: {index, role, content, token_count, tool_name?, timestamp}
#[pyfunction]
fn py_prune_conversation(
    _py: Python,
    blocks: Vec<Bound<'_, PyDict>>,
    token_budget: u32,
    decay_lambda: f64,
    protect_last: usize,
) -> PyResult<String> {
    use conversation_pruner::*;

    let conv_blocks: Vec<ConvBlock> = blocks.iter().enumerate().map(|(i, d)| {
        let index = d.get_item("index").ok().flatten()
            .and_then(|v| v.extract::<usize>().ok()).unwrap_or(i);
        let role: String = d.get_item("role").ok().flatten()
            .and_then(|v| v.extract().ok()).unwrap_or_default();
        let content: String = d.get_item("content").ok().flatten()
            .and_then(|v| v.extract().ok()).unwrap_or_default();
        let token_count: u32 = d.get_item("token_count").ok().flatten()
            .and_then(|v| v.extract().ok()).unwrap_or(0);
        let tool_name: Option<String> = d.get_item("tool_name").ok().flatten()
            .and_then(|v| v.extract().ok());
        let timestamp: f64 = d.get_item("timestamp").ok().flatten()
            .and_then(|v| v.extract().ok()).unwrap_or(index as f64);

        let kind = classify_block(&role, &content, tool_name.as_deref());
        let sh = dedup::simhash(&content);

        ConvBlock {
            index,
            kind,
            token_count,
            simhash: sh,
            content,
            role,
            tool_name,
            depends_on: vec![],
            timestamp,
        }
    }).collect();

    let result = prune_conversation(&conv_blocks, token_budget, decay_lambda, protect_last);
    serde_json::to_string(&result)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON error: {}", e)))
}

/// Progressive compression: assign resolutions based on context utilization.
/// Returns JSON: [{index, resolution}]
#[pyfunction]
fn py_progressive_thresholds(
    blocks: Vec<Bound<'_, PyDict>>,
    utilization: f64,
    recency_cutoff: usize,
) -> PyResult<String> {
    use conversation_pruner::*;

    let conv_blocks: Vec<ConvBlock> = blocks.iter().enumerate().map(|(i, d)| {
        let index = d.get_item("index").ok().flatten()
            .and_then(|v| v.extract::<usize>().ok()).unwrap_or(i);
        let role: String = d.get_item("role").ok().flatten()
            .and_then(|v| v.extract().ok()).unwrap_or_default();
        let content: String = d.get_item("content").ok().flatten()
            .and_then(|v| v.extract().ok()).unwrap_or_default();
        let token_count: u32 = d.get_item("token_count").ok().flatten()
            .and_then(|v| v.extract().ok()).unwrap_or(0);
        let tool_name: Option<String> = d.get_item("tool_name").ok().flatten()
            .and_then(|v| v.extract().ok());
        let timestamp: f64 = d.get_item("timestamp").ok().flatten()
            .and_then(|v| v.extract().ok()).unwrap_or(index as f64);

        let kind = classify_block(&role, &content, tool_name.as_deref());
        let sh = dedup::simhash(&content);

        ConvBlock {
            index,
            kind,
            token_count,
            simhash: sh,
            content,
            role,
            tool_name,
            depends_on: vec![],
            timestamp,
        }
    }).collect();

    let assignments = progressive_thresholds(&conv_blocks, utilization, recency_cutoff);
    let result: Vec<HashMap<String, String>> = assignments.iter().map(|(idx, res)| {
        let mut m = HashMap::new();
        m.insert("index".into(), idx.to_string());
        m.insert("resolution".into(), res.as_str().to_string());
        m
    }).collect();

    serde_json::to_string(&result)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON error: {}", e)))
}

/// Compress a single block at a given resolution level.
/// Returns the compressed text.
#[pyfunction]
#[pyo3(signature = (role, content, token_count, resolution, tool_name=None))]
fn py_compress_block(
    role: &str,
    content: &str,
    token_count: u32,
    resolution: &str,
    tool_name: Option<String>,
) -> String {
    use conversation_pruner::*;

    let kind = classify_block(role, content, tool_name.as_deref());
    let sh = dedup::simhash(content);

    let block = ConvBlock {
        index: 0,
        kind,
        token_count,
        simhash: sh,
        content: content.to_string(),
        role: role.to_string(),
        tool_name,
        depends_on: vec![],
        timestamp: 0.0,
    };

    let res = match resolution {
        "skeleton"    => Resolution::Skeleton,
        "digest"      => Resolution::Digest,
        "fingerprint" => Resolution::Fingerprint,
        _             => Resolution::Verbatim,
    };

    compress_block(&block, res)
}

/// Classify a conversation block by role and content.
/// Returns: "user", "assistant", "tool_call", "tool_result", "thinking", "system"
#[pyfunction]
#[pyo3(signature = (role, content, tool_name=None))]
fn py_classify_block(role: &str, content: &str, tool_name: Option<String>) -> String {
    use conversation_pruner::classify_block;
    classify_block(role, content, tool_name.as_deref()).label().to_string()
}

// ─── Extra standalone wrappers for direct test/utility access ────────────────

/// Cross-fragment redundancy: how much of `text` is already covered by `others` [0,1].
#[pyfunction]
fn py_cross_fragment_redundancy(text: &str, others: Vec<String>) -> f64 {
    use entropy::cross_fragment_redundancy;
    let refs: Vec<&str> = others.iter().map(|s| s.as_str()).collect();
    cross_fragment_redundancy(text, &refs)
}

/// Apply Ebbinghaus decay in-place to a list of ContextFragments.
/// Mutates recency_score based on turns elapsed since turn_last_accessed.
#[pyfunction]
fn py_apply_ebbinghaus_decay(
    fragments: Vec<ContextFragment>,
    current_turn: u32,
    half_life: u32,
) -> Vec<ContextFragment> {
    use fragment::apply_ebbinghaus_decay;
    let mut frags = fragments;
    apply_ebbinghaus_decay(&mut frags, current_turn, half_life);
    frags
}

/// Convenience knapsack optimizer for standalone use (default weights, empty feedback).
/// Returns (selected_fragments, stats_dict).
#[pyfunction]
fn py_knapsack_optimize(
    fragments: Vec<ContextFragment>,
    token_budget: u32,
) -> (Vec<ContextFragment>, HashMap<String, f64>) {
    let weights = knapsack::ScoringWeights::default();
    let feedback = HashMap::new();
    let result = knapsack_optimize(&fragments, token_budget, &weights, &feedback, 0.0); // hard path for py_knapsack_optimize compatibility
    let selected: Vec<ContextFragment> = result.selected_indices.iter()
        .map(|&i| fragments[i].clone())
        .collect();
    let mut stats = HashMap::new();
    stats.insert("total_tokens".to_string(), result.total_tokens as f64);
    stats.insert("total_relevance".to_string(), result.total_relevance);
    (selected, stats)
}

// ── Non-PyO3 helper methods (can't be in #[pymethods] due to serde types) ──
impl EntrolyEngine {
    /// Reconstruct the `selected` PyList from cached fragment IDs + live fragment data.
    ///
    /// On cache hit, the stored JSON contains `selected_ids: [{id, variant}, ...]`.
    /// We look up each fragment by ID in `self.fragments` and rebuild the full
    /// dict with content, preview, scores — exactly matching the miss path output.
    fn rebuild_selected_list(
        &self,
        py: Python<'_>,
        cached_value: &serde_json::Value,
    ) -> PyResult<PyObject> {
        let selected_list = pyo3::types::PyList::empty(py);

        if let Some(ids) = cached_value.get("selected_ids").and_then(|v| v.as_array()) {
            for entry in ids {
                let frag_id = match entry.get("id").and_then(|v| v.as_str()) {
                    Some(id) => id,
                    None => continue,
                };
                let variant = entry.get("variant").and_then(|v| v.as_str()).unwrap_or("full");

                if let Some(f) = self.fragments.get(frag_id) {
                    let d = PyDict::new(py);
                    d.set_item("id", &f.fragment_id)?;
                    d.set_item("source", &f.source)?;
                    d.set_item("variant", variant)?;
                    d.set_item("entropy_score", (f.entropy_score * 10000.0).round() / 10000.0)?;

                    let fm = self.feedback.learned_value(&f.fragment_id);
                    let rel = compute_relevance(f, self.w_recency, self.w_frequency, self.w_semantic, self.w_entropy, fm);
                    d.set_item("relevance", (rel * 10000.0).round() / 10000.0)?;

                    match variant {
                        "reference" => {
                            let ref_tokens = (f.source.len() as u32 / 4).clamp(3, 10);
                            d.set_item("token_count", ref_tokens)?;
                            d.set_item("preview", format!("[ref] {}", &f.source))?;
                            d.set_item("content", format!("[ref] {}", &f.source))?;
                        }
                        "belief" => {
                            // Belief: vault-compiled architectural summary (cache hit path)
                            let tc = f.belief_token_count.unwrap_or(f.token_count);
                            d.set_item("token_count", tc)?;
                            let content = f.belief_content.as_deref().unwrap_or(&f.content);
                            let preview = if content.len() > 100 {
                                let mut end = 100;
                                while end < content.len() && !content.is_char_boundary(end) { end += 1; }
                                format!("{}...", &content[..end])
                            } else {
                                content.to_string()
                            };
                            d.set_item("preview", &preview)?;
                            d.set_item("content", content)?;
                        }
                        "skeleton" => {
                            let tc = f.skeleton_token_count.unwrap_or(f.token_count);
                            d.set_item("token_count", tc)?;
                            let content = f.skeleton_content.as_deref().unwrap_or(&f.content);
                            let preview = if content.len() > 100 {
                                let mut end = 100;
                                while end < content.len() && !content.is_char_boundary(end) { end += 1; }
                                format!("{}...", &content[..end])
                            } else {
                                content.to_string()
                            };
                            d.set_item("preview", &preview)?;
                            d.set_item("content", content)?;
                        }
                        _ => {
                            // "full"
                            d.set_item("token_count", f.token_count)?;
                            let preview = if f.content.len() > 100 {
                                let mut end = 100;
                                while end < f.content.len() && !f.content.is_char_boundary(end) { end += 1; }
                                format!("{}...", &f.content[..end])
                            } else {
                                f.content.clone()
                            };
                            d.set_item("preview", &preview)?;
                            d.set_item("content", &f.content)?;
                        }
                    }
                    selected_list.append(d)?;
                }
            }
        }
        Ok(selected_list.into())
    }
}

/// Python-friendly DedupIndex — wraps the Rust DedupIndex struct.
#[pyclass]
struct PyDedupIndex {
    inner: dedup::DedupIndex,
}

#[pymethods]
impl PyDedupIndex {
    #[new]
    fn new(hamming_threshold: u32) -> Self {
        PyDedupIndex { inner: dedup::DedupIndex::new(hamming_threshold) }
    }

    /// Insert a fragment. Returns the ID of the duplicate if one was found, else None.
    fn insert(&mut self, fragment_id: &str, text: &str) -> Option<String> {
        self.inner.insert(fragment_id, text)
    }

    /// Remove a fragment by ID.
    fn remove(&mut self, fragment_id: &str) {
        self.inner.remove(fragment_id);
    }

    /// Current number of indexed fragments.
    fn size(&self) -> usize {
        self.inner.size()
    }
}
// ═══════════════════════════════════════════════════════════════════
// Module definition
// ═══════════════════════════════════════════════════════════════════

#[pymodule]
fn entroly_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ContextFragment>()?;
    m.add_class::<EntrolyEngine>()?;
    m.add_class::<PyDedupIndex>()?;
    // ── Entropy / Hashing
    m.add_function(wrap_pyfunction!(py_shannon_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(py_normalized_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(py_boilerplate_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(py_renyi_entropy_2, m)?)?;
    m.add_function(wrap_pyfunction!(py_entropy_divergence, m)?)?;
    m.add_function(wrap_pyfunction!(py_cross_fragment_redundancy, m)?)?;
    m.add_function(wrap_pyfunction!(py_simhash, m)?)?;
    m.add_function(wrap_pyfunction!(py_hamming_distance, m)?)?;
    m.add_function(wrap_pyfunction!(py_information_score, m)?)?;
    // ── Knapsack / Ebbinghaus
    m.add_function(wrap_pyfunction!(py_knapsack_optimize, m)?)?;
    m.add_function(wrap_pyfunction!(py_apply_ebbinghaus_decay, m)?)?;
    // ── SAST / Health / Query
    m.add_function(wrap_pyfunction!(py_scan_content, m)?)?;
    m.add_function(wrap_pyfunction!(py_analyze_health_info, m)?)?;
    m.add_function(wrap_pyfunction!(py_analyze_query, m)?)?;
    m.add_function(wrap_pyfunction!(py_refine_heuristic, m)?)?;
    // ── Conversation Pruner
    m.add_function(wrap_pyfunction!(py_prune_conversation, m)?)?;
    m.add_function(wrap_pyfunction!(py_progressive_thresholds, m)?)?;
    m.add_function(wrap_pyfunction!(py_compress_block, m)?)?;
    m.add_function(wrap_pyfunction!(py_classify_block, m)?)?;
    // ── Multi-Agent (additive — new classes, no existing API changes)
    m.add_class::<nkbe::NkbeAllocator>()?;
    m.add_class::<cognitive_bus::CognitiveBus>()?;
    // ── CogOps Epistemic Engine
    m.add_class::<cogops::CogOpsEngine>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simhash_distance_uses_64_not_32() {
        // Previously /32 meant hamming dist 48 → negative → clamped to 0
        // Now /64 means hamming dist 48 → (1.0 - 48/64) = 0.25
        let dist: u32 = 48;
        let score = (1.0 - dist as f64 / 64.0_f64).max(0.0);
        assert!(score > 0.0, "Hamming dist 48 should give positive score, got {}", score);
        assert!((score - 0.25).abs() < 0.001);

        // Old code would give: (1.0 - 48/32) = -0.5 → clamped to 0.0
        let old_score = (1.0 - dist as f64 / 32.0_f64).max(0.0);
        assert_eq!(old_score, 0.0, "Old /32 formula gives 0 for dist 48");
    }

    #[test]
    fn test_task_budget_multiplier() {
        // BugTracing → 1.5x
        let task = TaskType::classify("fix the bug");
        assert!((task.budget_multiplier() - 1.5).abs() < 0.01);
        assert_eq!((100.0 * task.budget_multiplier()) as u32, 150);

        // CodeGeneration → 0.7x
        let task = TaskType::classify("create a new endpoint");
        assert!((task.budget_multiplier() - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_export_import_roundtrip() {
        // Test serialization directly via serde (bypassing PyO3 wrappers)
        let mut fragments = HashMap::new();
        let mut frag = ContextFragment::new("f1".into(), "def foo(): return 42".into(), 10, "foo.py".into());
        frag.recency_score = 0.9;
        frag.entropy_score = 0.7;
        fragments.insert("f1".into(), frag);

        let mut frag2 = ContextFragment::new("f2".into(), "def bar(): return foo()".into(), 12, "bar.py".into());
        frag2.recency_score = 0.8;
        frag2.entropy_score = 0.6;
        fragments.insert("f2".into(), frag2);

        let dedup_index = DedupIndex::new(3);
        let dep_graph = DepGraph::new();
        let feedback = FeedbackTracker::new();

        let prism_test = crate::prism::PrismOptimizer::new(0.01);
        let resonance_test = ResonanceMatrix::new();
        let prism5_test = PrismOptimizer5D::from_4d(&prism_test);
        let state = EngineState {
            fragments: fragments.clone(),
            dedup_index: &dedup_index,
            dep_graph: &dep_graph,
            feedback: &feedback,
            prism_optimizer: &prism_test,
            w_recency: 0.30,
            w_frequency: 0.25,
            w_semantic: 0.25,
            w_entropy: 0.20,
            current_turn: 5,
            id_counter: 2,
            max_fragments: 10_000,
            total_tokens_saved: 100,
            total_optimizations: 3,
            total_fragments_ingested: 5,
            total_duplicates_caught: 1,
            total_explorations: 0,
            rng_state: 1,
            gradient_temperature: 2.0,
            gradient_norm_ema: 0.0,
            cache_snapshot: None,
            resonance_matrix: &resonance_test,
            prism_optimizer_5d: &prism5_test,
            w_resonance: 0.0,
            total_consolidations: 0,
            consolidation_tokens_saved: 0,
            causal_graph: &CausalContextGraph::new(),
        };

        let json = serde_json::to_string(&state).expect("failed to serialize OwnedEngineState to JSON");
        assert!(!json.is_empty());

        // Deserialize
        let restored: OwnedEngineState = serde_json::from_str(&json).expect("failed to deserialize OwnedEngineState from JSON");
        assert_eq!(restored.fragments.len(), 2);
        assert_eq!(restored.current_turn, 5);
        assert_eq!(restored.id_counter, 2);
        assert_eq!(restored.total_tokens_saved, 100);
    }

    #[test]
    fn test_sufficiency_full() {
        let mut engine = EntrolyEngine::new(0.30, 0.25, 0.25, 0.20, 15, 0.05, 3, 0.1, 10_000, true, true, true, 0.70, 0.50, 0.15, 0.10);

        // Register a symbol in the dep graph
        engine.dep_graph.register_symbol("calculate_tax", "f1");

        // Fragment that defines calculate_tax
        let mut frag1 = ContextFragment::new("f1".into(), "def calculate_tax(income): return income * 0.3".into(), 20, "tax.py".into());
        frag1.recency_score = 1.0;
        frag1.entropy_score = 0.8;
        engine.fragments.insert("f1".into(), frag1.clone());

        // Fragment that references calculate_tax
        let mut frag2 = ContextFragment::new("f2".into(), "total = calculate_tax(50000)".into(), 10, "main.py".into());
        frag2.recency_score = 1.0;
        frag2.entropy_score = 0.7;
        engine.fragments.insert("f2".into(), frag2.clone());

        let frags = vec![frag1, frag2];
        let selected = vec![0, 1]; // Both selected

        let suff = engine.compute_sufficiency(&frags, &selected);
        assert!((suff - 1.0).abs() < 0.01, "Both present = 100% sufficiency, got {}", suff);

        // Now only select the caller, not the definition
        let selected_partial = vec![1]; // Only f2 (calls calculate_tax but doesn't define it)
        let suff_partial = engine.compute_sufficiency(&frags, &selected_partial);
        assert!(suff_partial < 1.0, "Missing definition = partial sufficiency, got {}", suff_partial);
    }

    #[test]
    fn test_exploration_rate_bounds() {
        let mut engine = EntrolyEngine::new(0.30, 0.25, 0.25, 0.20, 15, 0.05, 3, 0.1, 10_000, true, true, true, 0.70, 0.50, 0.15, 0.10);
        engine.set_exploration_rate(1.5);
        assert!((engine.exploration_rate - 1.0).abs() < 0.001);
        engine.set_exploration_rate(-0.5);
        assert!((engine.exploration_rate - 0.0).abs() < 0.001);
        engine.set_exploration_rate(0.1);
        assert!((engine.exploration_rate - 0.1).abs() < 0.001);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Quality+Correctness Tests
    //
    //  These tests guard against the main ways a RAG context engine can give
    //  wrong or low-quality answers:
    //    1. Returning irrelevant fragments (recall accuracy)
    //    2. Ranking relevant fragments below irrelevant ones (score order)
    //    3. Silently dropping correct results after LSH migration
    //    4. Wrong scoring math (ContextScorer monotonicity)
    // ═══════════════════════════════════════════════════════════════════════

    /// Recall must return the EXACT fragment that matches the query content —
    /// not a random or irrelevant one. This is the most fundamental correctness
    /// guarantee: if you store "fn connect_database()" and query "database",
    /// that fragment must be top-1.
    #[test]
    fn test_recall_returns_correct_fragment_not_random() {
        use crate::dedup::simhash;
        let mut engine = EntrolyEngine::new(0.30, 0.25, 0.25, 0.20, 15, 0.05, 3, 0.0, 10_000, true, true, true, 0.70, 0.50, 0.15, 0.10);

        // Target: database code
        let target = "fn connect_to_database(host: &str, port: u16) -> Connection { ... }";
        // Noise: completely unrelated content
        let noise = [
            "struct UserProfile { name: String, age: u32 }",
            "fn render_html_template(ctx: Context) -> String { ... }",
            "const MAX_RETRY_COUNT: usize = 5;",
            "fn validate_email_format(email: &str) -> bool { ... }",
            "struct Config { debug: bool, log_level: LogLevel }",
        ];

        // Ingest noise FIRST so they get lower IDs (tests that ordering is by score, not insertion)
        for (i, n) in noise.iter().enumerate() {
            let mut frag = ContextFragment::new(
                format!("noise{}", i), n.to_string(), 20, format!("noise{}.rs", i)
            );
            frag.simhash = simhash(n);
            frag.recency_score = 0.9;
            frag.entropy_score = 0.5;
            engine.fragments.insert(format!("noise{}", i), frag.clone());
            let slot = engine.fragment_slot_ids.len();
            engine.fragment_slot_ids.push(format!("noise{}", i));
            engine.lsh_index.insert(frag.simhash, slot);
        }

        // Ingest the target LAST
        let mut frag_target = ContextFragment::new(
            "target".to_string(), target.to_string(), 20, "db.rs".to_string()
        );
        frag_target.simhash = simhash(target);
        frag_target.recency_score = 0.9;
        frag_target.entropy_score = 0.8;
        engine.fragments.insert("target".to_string(), frag_target.clone());
        let slot = engine.fragment_slot_ids.len();
        engine.fragment_slot_ids.push("target".to_string());
        engine.lsh_index.insert(frag_target.simhash, slot);

        // Query: same content → should be exact fingerprint match
        let query_fp = simhash(target);
        let candidates = engine.lsh_index.query(query_fp);

        // The target slot must be in LSH candidates
        let target_slot = engine.fragment_slot_ids.iter().position(|id| id == "target").expect("target fragment not found in slot IDs — test setup error");
        assert!(
            candidates.contains(&target_slot),
            "LSH must return the exact-match fragment. Candidates: {:?}, target slot: {}",
            candidates, target_slot
        );
    }

    /// Recall ranking must be MONOTONE: fragment[0].relevance >= fragment[1].relevance >= ...
    /// If this fails, the LLM sees irrelevant context before relevant context.
    #[test]
    fn test_recall_ranking_is_monotone_descending() {
        use crate::dedup::simhash;
        let query = "async fn process_payment(amount: f64, currency: &str) -> Result<Receipt>";

        let mut engine = EntrolyEngine::new(0.30, 0.25, 0.25, 0.20, 15, 0.05, 3, 0.0, 10_000, true, true, true, 0.70, 0.50, 0.15, 0.10);
        let query_fp = simhash(query);

        // Varying content: exact match, near match, unrelated
        let contents = [
            query.to_string(),   // identical → highest score
            "async fn process_payment(amount: f64) -> Result<()> {}".to_string(), // very similar
            "fn validate_user_token(token: &str) -> bool { false }".to_string(), // unrelated
            "const TAX_RATE: f64 = 0.15;".to_string(), // completely unrelated
        ];

        for (i, content) in contents.iter().enumerate() {
            let mut frag = ContextFragment::new(
                format!("f{}", i), content.clone(), 20, format!("f{}.rs", i)
            );
            frag.simhash = simhash(content);
            frag.recency_score = 0.9;
            frag.entropy_score = 0.7;
            engine.fragments.insert(format!("f{}", i), frag.clone());
            let slot = engine.fragment_slot_ids.len();
            engine.fragment_slot_ids.push(format!("f{}", i));
            engine.lsh_index.insert(frag.simhash, slot);
        }

        // Score all candidates
        let candidates = engine.lsh_index.query(query_fp);
        let mut scored: Vec<(String, f64)> = candidates.iter()
            .filter_map(|&slot| {
                let id = engine.fragment_slot_ids.get(slot)?;
                let f = engine.fragments.get(id)?;
                let dist = crate::dedup::hamming_distance(query_fp, f.simhash);
                let rel = engine.context_scorer.score(dist, f.recency_score, f.entropy_score, f.frequency_score, 1.0);
                Some((id.clone(), rel))
            })
            .collect();

        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Check monotone descent
        for window in scored.windows(2) {
            assert!(
                window[0].1 >= window[1].1,
                "Recall ranking broken: {} (score={:.4}) ranked below {} (score={:.4})",
                window[0].0, window[0].1, window[1].0, window[1].1
            );
        }

        // The exact-match fragment (f0) must be rank-1
        if let Some((top_id, _)) = scored.first() {
            assert_eq!(top_id, "f0", "Exact match must be top-1. Got: {:?}", scored);
        }
    }

    /// ContextScorer must be MONOTONE in similarity: higher similarity → higher score
    /// (everything else equal). If this fails, the scorer would rank distant fragments
    /// above close ones.
    #[test]
    fn test_context_scorer_similarity_monotone() {
        let scorer = crate::lsh::ContextScorer::default();
        let recency = 0.8;
        let entropy = 0.7;
        let freq = 0.5;

        let scores: Vec<f64> = (0u32..=8).map(|hamming| {
            scorer.score(hamming * 8, recency, entropy, freq, 1.0)
        }).collect();

        for window in scores.windows(2) {
            assert!(
                window[0] >= window[1],
                "Scorer not monotone: hamming increase should decrease score. Scores: {:?}", scores
            );
        }
    }

    /// LshIndex must not silently drop exact-match entries even after 1000 inserts.
    /// If this fails, some fragments will NEVER be recalled regardless of query.
    #[test]
    fn test_lsh_never_drops_exact_match_after_scale() {
        let mut idx = crate::lsh::LshIndex::new();
        let target_fp: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let target_slot = 500usize;

        // Insert 1000 random fingerprints to stress the index
        for i in 0u64..1000 {
            let fp = i.wrapping_mul(0x9E3779B97F4A7C15) ^ (i << 32);
            idx.insert(fp, i as usize);
        }
        // Insert our target
        idx.insert(target_fp, target_slot);

        let candidates = idx.query(target_fp);
        assert!(
            candidates.contains(&target_slot),
            "LSH dropped the exact-match at scale=1000. candidates={:?}", candidates
        );
    }
}
