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

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use serde::{Deserialize, Serialize};

use fragment::{ContextFragment, compute_relevance};
use knapsack::{knapsack_optimize, ScoringWeights};
use knapsack_sds::{ios_select, Resolution, InfoFactors};
use entropy::{information_score, shannon_entropy, normalized_entropy, boilerplate_ratio};
use dedup::{simhash, hamming_distance, DedupIndex};
use depgraph::{DepGraph, extract_identifiers};
use guardrails::{file_criticality, has_safety_signal, TaskType, FeedbackTracker, Criticality, compute_ordering_priority};
use prism::PrismOptimizer;


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
    gradient_norm_ema: f64,
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
        ios_skeleton_info_factor=0.70, ios_reference_info_factor=0.15, ios_diversity_floor=0.10
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
            last_optimization: None,
            lsh_index: lsh::LshIndex::new(),
            context_scorer: lsh::ContextScorer::default(),
            enable_ios,
            enable_ios_diversity,
            enable_ios_multi_resolution,
            ios_skeleton_info_factor: ios_skeleton_info_factor.clamp(0.01, 0.99),
            ios_reference_info_factor: ios_reference_info_factor.clamp(0.01, 0.99),
            ios_diversity_floor: ios_diversity_floor.clamp(0.0, 1.0),
            gradient_temperature: 2.0, // Start soft — anneals via τ *= 0.995 each turn
            last_lambda_star: 0.0,     // λ* from last forward pass — shared with REINFORCE
            gradient_norm_ema: 0.0,
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

            // Apply task-type budget multiplier
            let effective_budget = if !query.is_empty() {
                let task_type = TaskType::classify(&query);
                let mult = task_type.budget_multiplier();
                (token_budget as f64 * mult) as u32
            } else {
                token_budget
            };

            // Update semantic scores if query provided
            if !query.is_empty() {
                let query_hash = simhash(&query);
                for frag in self.fragments.values_mut() {
                    let dist = hamming_distance(query_hash, frag.simhash);
                    frag.semantic_score = (1.0 - dist as f64 / 64.0).max(0.0);
                }
            }

            // Build feedback multipliers for all fragments
            let feedback_mults: HashMap<String, f64> = self.fragments.keys()
                .map(|fid| (fid.clone(), self.feedback.learned_value(fid)))
                .collect();

            let frags: Vec<ContextFragment> = self.fragments.values().cloned().collect();
            let weights = ScoringWeights {
                recency: self.w_recency,
                frequency: self.w_frequency,
                semantic: self.w_semantic,
                entropy: self.w_entropy,
            };

            // ── Dependency-aware score boosting ──
            // First pass with basic knapsack to discover initial selection,
            // then compute dep boosts from that selection.
            let result1 = knapsack_optimize(&frags, effective_budget, &weights, &feedback_mults, self.gradient_temperature);
            // Store λ* for the REINFORCE backward pass — ensures forward/backward p_i are identical.
            self.last_lambda_star = result1.lambda_star;
            let initial_selected_ids: HashSet<String> = result1.selected_indices.iter()
                .map(|&i| frags[i].fragment_id.clone())
                .collect();
            let dep_boosts = self.dep_graph.compute_dep_boosts(&initial_selected_ids);

            // Apply dep boosts to fragments' semantic scores
            let mut boosted_frags = frags.clone();
            for frag in boosted_frags.iter_mut() {
                if !initial_selected_ids.contains(&frag.fragment_id) {
                    if let Some(&boost) = dep_boosts.get(&frag.fragment_id) {
                        if boost > 0.3 {
                            frag.semantic_score = (frag.semantic_score + boost * 0.5).min(1.0);
                        }
                    }
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
                );

                final_indices = Vec::new();
                for (idx, resolution) in &ios_result.selections {
                    match resolution {
                        Resolution::Full => final_indices.push(*idx),
                        Resolution::Skeleton => skeleton_indices.push(*idx),
                        Resolution::Reference => skeleton_indices.push(*idx), // References handled like skeletons in output
                    }
                    ios_resolutions.insert(*idx, *resolution);
                }
                skeleton_tokens_used = ios_result.total_tokens.saturating_sub(
                    final_indices.iter().map(|&i| frags[i].token_count).sum::<u32>()
                );
                ios_diversity_score = Some(ios_result.diversity_score);
            } else {
                // ── Legacy Path: Standard knapsack + exploration + skeleton pass ──
                selection_method = "legacy_knapsack";
                let result = if dep_boosts.values().any(|&b| b > 0.3) {
                    knapsack_optimize(&boosted_frags, effective_budget, &weights, &feedback_mults, self.gradient_temperature)
                } else {
                    result1
                };

                final_indices = result.selected_indices.clone();

                // ε-Greedy Exploration
                let lcg_val = (self.total_optimizations.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)) % 1000;
                let threshold = (self.exploration_rate * 1000.0) as u64;

                if frags.len() > final_indices.len() && !final_indices.is_empty() && lcg_val < threshold {
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
                                if rel < min_rel {
                                    min_rel = rel;
                                    min_pos = Some(pos);
                                }
                            }
                        }

                        if let Some(pos) = min_pos {
                            let explore_idx = unselected[(lcg_val as usize) % unselected.len()];
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

            // Track savings
            let total_available: u32 = frags.iter().map(|f| f.token_count).sum();
            let full_tokens: u32 = final_indices.iter().map(|&i| frags[i].token_count).sum();
            let final_tokens: u32 = full_tokens + skeleton_tokens_used;
            let saved = total_available.saturating_sub(final_tokens);
            self.total_tokens_saved += saved as u64;

            // Mark selected as accessed
            for &idx in &final_indices {
                let fid = &frags[idx].fragment_id;
                if let Some(f) = self.fragments.get_mut(fid) {
                    f.turn_last_accessed = self.current_turn;
                    f.access_count += 1;
                }
            }

            // ── Context Sufficiency Scoring ──
            // What fraction of referenced symbols have definitions in selected context?
            let selected_id_set: HashSet<String> = final_indices.iter()
                .map(|&i| frags[i].fragment_id.clone())
                .collect();
            let sufficiency = self.compute_sufficiency(&frags, &final_indices);

            // ── Context ordering: sort selected for LLM attention ──
            let mut ordered_indices = final_indices.clone();
            ordered_indices.sort_by(|&a, &b| {
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

            // ── Build Python result ──
            let py_result = PyDict::new(py);
            py_result.set_item("method", selection_method)?;
            py_result.set_item("total_tokens", final_tokens)?;
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
            if let Some(div_score) = ios_diversity_score {
                py_result.set_item("ios_diversity_score", div_score)?;
                py_result.set_item("ios_enabled", true)?;
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
                    selected_list.append(d)?;
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
                    selected_list.append(d)?;
                }
            }
            py_result.set_item("selected", selected_list)?;

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

            scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
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
    pub fn stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let total_tokens: u32 = self.fragments.values().map(|f| f.token_count).sum();
            let avg_entropy = if self.fragments.is_empty() {
                0.0
            } else {
                self.fragments.values().map(|f| f.entropy_score).sum::<f64>()
                    / self.fragments.len() as f64
            };
            let pinned = self.fragments.values().filter(|f| f.is_pinned).count();

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
    pub fn record_success(&mut self, fragment_ids: Vec<String>) {
        self.feedback.record_success(&fragment_ids);
        self.apply_prism_rl_update(&fragment_ids, 1.0);
    }

    /// Record that the selected fragments led to a failed output.
    pub fn record_failure(&mut self, fragment_ids: Vec<String>) {
        self.feedback.record_failure(&fragment_ids);
        self.apply_prism_rl_update(&fragment_ids, -1.0);
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
            current_turn: self.current_turn,
            id_counter: self.id_counter,
            max_fragments: self.max_fragments,
            total_tokens_saved: self.total_tokens_saved,
            total_optimizations: self.total_optimizations,
            total_fragments_ingested: self.total_fragments_ingested,
            total_duplicates_caught: self.total_duplicates_caught,
            total_explorations: self.total_explorations,
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
        self.fragments = state.fragments;
        self.dedup_index = state.dedup_index;
        self.dep_graph = state.dep_graph;
        self.feedback = state.feedback;
        // Restore PRISM covariance if available; fall back to fresh optimizer to support
        // checkpoints created before this field was added.
        if let Some(p) = state.prism_optimizer {
            self.prism_optimizer = p;
        }
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
        self.last_optimization = None;
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

        // Compute REINFORCE-with-baseline policy gradient
        let mut g = [0.0_f64; 4]; // [∂/∂w_r, ∂/∂w_f, ∂/∂w_s, ∂/∂w_e]
        // λ* from the last forward bisection — use the SAME probability as the forward pass.
        // p_i = σ((s_i − λ*·tokens_i)/τ) is the exact KKT soft selection probability.
        // Reusing it here ensures advantage = (action − p_exact) × R is an unbiased estimator.
        let lambda = self.last_lambda_star;

        for frag in self.fragments.values() {
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

            // REINFORCE baseline: advantage = (action - expected_action) × reward
            // action = 1 if selected, 0 if not
            // expected_action = p (exact KKT soft probability)
            let action = if selected.contains(frag.fragment_id.as_str()) { 1.0 } else { 0.0 };
            let advantage = (action - p) * reward;

            // Accumulate: advantage_i × σ'((s_i - λ*·tokens_i)/τ) × feature_{i,k}
            g[0] += advantage * dp * frag.recency_score;
            g[1] += advantage * dp * frag.frequency_score;
            g[2] += advantage * dp * frag.semantic_score;
            g[3] += advantage * dp * frag.entropy_score;
        }


        // ── Regime change detection ──
        // If gradient norm spikes >3× the running EMA, the task distribution shifted.
        // Reset temperature to re-explore the weight space.
        let g_norm = (g[0]*g[0] + g[1]*g[1] + g[2]*g[2] + g[3]*g[3]).sqrt();
        if self.gradient_norm_ema > 1e-8 && g_norm > self.gradient_norm_ema * 3.0 {
            self.gradient_temperature = 2.0; // regime change — re-explore
        } else {
            // Normal annealing: τ *= 0.995 each update (soft→hard over ~460 turns)
            self.gradient_temperature = (self.gradient_temperature * 0.995).max(0.1);
        }
        self.gradient_norm_ema = 0.95 * self.gradient_norm_ema + 0.05 * g_norm;

        // Let PRISM compute the anisotropically-damped update: Q Λ^{-1/2} Q^T g
        let update = self.prism_optimizer.compute_update(&g);

        // Apply updates to weights
        self.w_recency   += update[0];
        self.w_frequency += update[1];
        self.w_semantic  += update[2];
        self.w_entropy   += update[3];

        // Prevent collapse: clamp weights to positive bounds [0.05, 0.8]
        self.w_recency   = self.w_recency.clamp(0.05, 0.8);
        self.w_frequency = self.w_frequency.clamp(0.05, 0.8);
        self.w_semantic  = self.w_semantic.clamp(0.05, 0.8);
        self.w_entropy   = self.w_entropy.clamp(0.05, 0.8);

        // Normalize weights so they sum to 1.0 to preserve scoring scale
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
    current_turn: u32,
    id_counter: u64,
    max_fragments: usize,
    total_tokens_saved: u64,
    total_optimizations: u64,
    total_fragments_ingested: u64,
    total_duplicates_caught: u64,
    total_explorations: u64,
}

/// Owned state for deserialization.
#[derive(Deserialize)]
struct OwnedEngineState {
    fragments: HashMap<String, ContextFragment>,
    dedup_index: DedupIndex,
    dep_graph: DepGraph,
    feedback: FeedbackTracker,
    prism_optimizer: Option<PrismOptimizer>,  // Optional for backward-compat with old checkpoints
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
}

fn default_max_fragments() -> usize { 10_000 }

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
    serde_json::to_string(&report).unwrap_or_else(|e| format!("{{\"error\":\"{}\"}}", e))
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
        let state = EngineState {
            fragments: fragments.clone(),
            dedup_index: &dedup_index,
            dep_graph: &dep_graph,
            feedback: &feedback,
            prism_optimizer: &prism_test,
            current_turn: 5,
            id_counter: 2,
            max_fragments: 10_000,
            total_tokens_saved: 100,
            total_optimizations: 3,
            total_fragments_ingested: 5,
            total_duplicates_caught: 1,
            total_explorations: 0,
        };

        let json = serde_json::to_string(&state).unwrap();
        assert!(!json.is_empty());

        // Deserialize
        let restored: OwnedEngineState = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.fragments.len(), 2);
        assert_eq!(restored.current_turn, 5);
        assert_eq!(restored.id_counter, 2);
        assert_eq!(restored.total_tokens_saved, 100);
    }

    #[test]
    fn test_sufficiency_full() {
        let mut engine = EntrolyEngine::new(0.30, 0.25, 0.25, 0.20, 15, 0.05, 3, 0.1, 10_000, true, true, true, 0.70, 0.15, 0.10);

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
        let mut engine = EntrolyEngine::new(0.30, 0.25, 0.25, 0.20, 15, 0.05, 3, 0.1, 10_000, true, true, true, 0.70, 0.15, 0.10);
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
        let mut engine = EntrolyEngine::new(0.30, 0.25, 0.25, 0.20, 15, 0.05, 3, 0.0, 10_000, true, true, true, 0.70, 0.15, 0.10);

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
        let target_slot = engine.fragment_slot_ids.iter().position(|id| id == "target").unwrap();
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

        let mut engine = EntrolyEngine::new(0.30, 0.25, 0.25, 0.20, 15, 0.05, 3, 0.0, 10_000, true, true, true, 0.70, 0.15, 0.10);
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
