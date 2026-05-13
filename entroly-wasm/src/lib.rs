//! Entroly WebAssembly Module
//!
//! Exposes the full Entroly engine to JavaScript/TypeScript via wasm-bindgen.
//! Identical algorithms to the Python build — same knapsack, entropy scoring,
//! EGSC caching, dependency graph, RL feedback — with JSON-based I/O.
//!
//! Usage (JS/TS):
//!   import init, { WasmEntrolyEngine } from 'entroly';
//!   await init();
//!   const engine = new WasmEntrolyEngine();
//!   engine.ingest("code.py", "def hello(): ...", 50, false);
//!   const result = engine.optimize(4096, "find auth bugs");

// The wasm crate intentionally mirrors the native Rust crate so the JS package
// exposes the same API shape. Some structs and helpers are used only through
// wasm-bindgen or kept for parity with native checkpoint formats, which makes
// Rust's local dead-code analysis noisier than the product surface.
#![allow(dead_code, unused_imports)]
#![allow(
    clippy::if_same_then_else,
    clippy::manual_is_multiple_of,
    clippy::new_without_default
)]

mod anomaly;
mod cache;
mod causal;
mod channel;
mod cognitive_bus;
mod conversation_pruner;
mod dedup;
mod depgraph;
mod entropy;
mod fragment;
mod guardrails;
mod health;
mod hierarchical;
mod knapsack;
mod knapsack_sds;
mod lsh;
mod nkbe;
mod prism;
mod query;
mod query_persona;
mod resonance;
mod sast;
mod semantic_dedup;
mod skeleton;
mod utilization;

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use wasm_bindgen::prelude::*;

/// Convert any Serialize type → JsValue via JSON string roundtrip.
/// serde_wasm_bindgen::to_value() does NOT handle dynamic serde_json::Value correctly
/// (produces empty objects). This generic helper works for both typed structs
/// (#[derive(Serialize)]) and dynamic serde_json::Value objects.
fn json_to_js<T: serde::Serialize>(val: &T) -> JsValue {
    let s = serde_json::to_string(val).unwrap_or_else(|_| "null".to_string());
    js_sys::JSON::parse(&s).unwrap_or(JsValue::NULL)
}

use cache::{CacheLookup, EgscCache, EgscConfig};
use causal::CausalContextGraph;
use dedup::{hamming_distance, simhash, DedupIndex};
use depgraph::{extract_identifiers, DepGraph};
use entropy::{
    boilerplate_ratio, entropy_divergence, information_score, normalized_entropy, renyi_entropy_2,
    shannon_entropy,
};
use fragment::{apply_ebbinghaus_decay, compute_relevance, ContextFragment};
use guardrails::{
    compute_ordering_priority, file_criticality, has_safety_signal, Criticality, FeedbackTracker,
    TaskType,
};
use knapsack::{compute_lambda_star, knapsack_optimize, ScoringWeights};
use knapsack_sds::{ios_select, InfoFactors, Resolution};
use prism::PrismOptimizer;
use prism::PrismOptimizer5D;
use query_persona::QueryPersonaManifold;
use resonance::ResonanceMatrix;

// ═══════════════════════════════════════════════════════════════════
// Serializable result types (returned as JSON to JS)
// ═══════════════════════════════════════════════════════════════════

#[derive(Serialize)]
struct FragmentScore {
    fragment_id: String,
    content: String,
    token_count: u32,
    source: String,
    relevance: f64,
    entropy_score: f64,
    selected: bool,
}

#[derive(Serialize)]
struct OptimizeResult {
    selected: Vec<FragmentScore>,
    total_tokens: u32,
    token_budget: u32,
    fragments_considered: usize,
    cache_hit: bool,
}

#[derive(Serialize)]
struct IngestResult {
    fragment_id: String,
    token_count: u32,
    entropy_score: f64,
    is_duplicate: bool,
    duplicate_of: Option<String>,
    status: String,
    criticality: String,
    is_pinned: bool,
    total_fragments: usize,
    has_skeleton: bool,
    skeleton_token_count: Option<u32>,
}

#[derive(Serialize)]
struct CacheStatsView {
    total_entries: usize,
    hit_rate: f64,
    total_tokens_saved: u64,
    exact_hits: u64,
    semantic_hits: u64,
}

#[derive(Serialize)]
struct EngineStats {
    total_fragments: usize,
    total_tokens: u32,
    current_turn: u32,
    cache: CacheStatsView,
}

/// Snapshot of the last optimization for explainability.
struct OptimizationSnapshot {
    fragment_scores: Vec<ExplainScore>,
    sufficiency: f64,
    explored_ids: Vec<String>,
}

struct ExplainScore {
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

// ═══════════════════════════════════════════════════════════════════
// WasmEntrolyEngine — Full feature parity with entroly-core
// ═══════════════════════════════════════════════════════════════════

#[wasm_bindgen]
pub struct WasmEntrolyEngine {
    fragments: HashMap<String, ContextFragment>,
    fragment_slot_ids: Vec<String>,
    dedup_index: DedupIndex,
    dep_graph: DepGraph,
    feedback: FeedbackTracker,
    current_turn: u32,

    // Scoring weights
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

    // Fragment ID generation
    instance_id: u64,
    id_counter: u64,
    max_fragments: usize,

    // Exploration
    total_explorations: u64,
    exploration_rate: f64,
    rng_state: u64,

    // Explainability
    last_optimization: Option<OptimizationSnapshot>,

    // LSH index
    lsh_index: lsh::LshIndex,
    context_scorer: lsh::ContextScorer,

    // Prism optimizer
    prism_optimizer: PrismOptimizer,

    // IOS
    enable_ios: bool,
    enable_ios_diversity: bool,
    enable_ios_multi_resolution: bool,
    ios_skeleton_info_factor: f64,
    ios_reference_info_factor: f64,
    ios_diversity_floor: f64,

    // Gradient-based learning
    gradient_temperature: f64,
    last_lambda_star: f64,
    last_dual_gap: f64,
    gradient_norm_ema: f64,

    // Query Persona Manifold
    query_manifold: QueryPersonaManifold,
    last_archetype_id: Option<String>,
    enable_query_personas: bool,

    // Channel Coding
    enable_channel_coding: bool,
    reward_baseline_ema: f64,

    // EGSC Cache
    egsc_cache: EgscCache,
    last_query: String,
    last_effective_budget: u32,
    last_cache_feedback_eligible: bool,

    // Context Resonance
    resonance_matrix: ResonanceMatrix,
    prism_optimizer_5d: PrismOptimizer5D,
    w_resonance: f64,
    enable_resonance: bool,

    // Coverage Sufficiency Estimator
    last_semantic_candidates: usize,
    last_structural_candidates: usize,
    last_candidate_overlap: usize,

    // Fragment Consolidation
    total_consolidations: u64,
    consolidation_tokens_saved: u64,
    consolidation_hamming_threshold: u32,

    // Causal Context Graph
    causal_graph: CausalContextGraph,
    enable_causal: bool,
    prev_selected_ids: Vec<String>,
    prev_explored_ids: Vec<String>,

    // Legacy compat
    hamming_threshold: u32,
}

#[wasm_bindgen]
impl WasmEntrolyEngine {
    /// Create a new Entroly engine with default parameters.
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmEntrolyEngine {
        // Generate instance_id (simple counter for wasm — no multi-threading)
        static mut INSTANCE_COUNTER: u64 = 0;
        let instance_id = unsafe {
            INSTANCE_COUNTER += 1;
            let raw = INSTANCE_COUNTER;
            let mut x = raw.wrapping_add(0x9e3779b97f4a7c15);
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            x ^ (x >> 31)
        };

        let w_recency = 0.30;
        let w_frequency = 0.25;
        let w_semantic = 0.25;
        let w_entropy = 0.20;

        WasmEntrolyEngine {
            fragments: HashMap::new(),
            fragment_slot_ids: Vec::new(),
            dedup_index: DedupIndex::new(3),
            dep_graph: DepGraph::new(),
            feedback: FeedbackTracker::new(),
            current_turn: 0,
            w_recency,
            w_frequency,
            w_semantic,
            w_entropy,
            decay_half_life: 15,
            min_relevance: 0.05,
            total_tokens_saved: 0,
            total_optimizations: 0,
            total_fragments_ingested: 0,
            total_duplicates_caught: 0,
            instance_id,
            id_counter: 0,
            max_fragments: 10000,
            total_explorations: 0,
            exploration_rate: 0.1,
            rng_state: instance_id | 1,
            last_optimization: None,
            lsh_index: lsh::LshIndex::new(),
            context_scorer: lsh::ContextScorer::default(),
            prism_optimizer: PrismOptimizer::new(0.01),
            enable_ios: true,
            enable_ios_diversity: true,
            enable_ios_multi_resolution: true,
            ios_skeleton_info_factor: 0.70,
            ios_reference_info_factor: 0.15,
            ios_diversity_floor: 0.10,
            gradient_temperature: 2.0,
            last_lambda_star: 0.0,
            last_dual_gap: 0.0,
            gradient_norm_ema: 0.0,
            query_manifold: QueryPersonaManifold::new(
                [w_recency, w_frequency, w_semantic, w_entropy],
                1.0,
                0.25,
            ),
            last_archetype_id: None,
            enable_query_personas: true,
            enable_channel_coding: true,
            reward_baseline_ema: 0.0,
            egsc_cache: EgscCache::default(),
            last_query: String::new(),
            last_effective_budget: 0,
            last_cache_feedback_eligible: false,
            resonance_matrix: ResonanceMatrix::new(),
            prism_optimizer_5d: PrismOptimizer5D::from_4d(&PrismOptimizer::new(0.01)),
            w_resonance: 0.0,
            enable_resonance: true,
            last_semantic_candidates: 0,
            last_structural_candidates: 0,
            last_candidate_overlap: 0,
            total_consolidations: 0,
            consolidation_tokens_saved: 0,
            consolidation_hamming_threshold: 8,
            causal_graph: CausalContextGraph::new(),
            enable_causal: true,
            prev_selected_ids: Vec::new(),
            prev_explored_ids: Vec::new(),
            hamming_threshold: 3,
        }
    }

    /// Ingest a context fragment into the engine.
    /// Pipeline: tokens → SimHash → dedup → entropy → criticality → depgraph → store
    #[wasm_bindgen]
    pub fn ingest(
        &mut self,
        content: String,
        source: String,
        token_count: u32,
        is_pinned: bool,
    ) -> JsValue {
        self.total_fragments_ingested += 1;

        let tc = if token_count == 0 {
            let non_alpha = content.chars().filter(|c| !c.is_alphabetic()).count();
            let ratio = non_alpha as f64 / content.len().max(1) as f64;
            let chars_per_token = if ratio > 0.4 { 5.0 } else { 4.0 };
            (content.len() as f64 / chars_per_token).max(1.0) as u32
        } else {
            token_count
        };

        // Enforce max_fragments cap
        if self.fragments.len() >= self.max_fragments {
            let result = serde_json::json!({
                "status": "rejected",
                "reason": "max_fragments cap reached",
                "max_fragments": self.max_fragments,
            });
            return json_to_js(&result);
        }

        // Generate globally-unique fragment ID
        self.id_counter += 1;
        let frag_id = format!("f{:08x}_{:06x}", self.instance_id as u32, self.id_counter);

        // Check for duplicates
        if let Some(dup_id) = self.dedup_index.insert(&frag_id, &content) {
            self.total_duplicates_caught += 1;
            self.total_tokens_saved += tc as u64;

            let max_freq = self
                .fragments
                .values()
                .map(|f| f.access_count)
                .max()
                .unwrap_or(1);
            if let Some(existing) = self.fragments.get_mut(&dup_id) {
                existing.access_count += 1;
                existing.turn_last_accessed = self.current_turn;
                existing.frequency_score = (existing.access_count as f64
                    / max_freq.max(existing.access_count) as f64)
                    .min(1.0);
            }

            let result = IngestResult {
                fragment_id: frag_id,
                token_count: tc,
                entropy_score: 0.0,
                is_duplicate: true,
                duplicate_of: Some(dup_id),
                status: "duplicate".into(),
                criticality: "Normal".into(),
                is_pinned,
                total_fragments: self.fragments.len(),
                has_skeleton: false,
                skeleton_token_count: None,
            };
            return json_to_js(&result);
        }

        // Compute entropy score
        let mut sorted_frags: Vec<&ContextFragment> = self.fragments.values().collect();
        sorted_frags.sort_by(|a, b| a.fragment_id.cmp(&b.fragment_id));
        let other_contents: Vec<String> = sorted_frags
            .iter()
            .take(50)
            .map(|f| f.content.clone())
            .collect();
        let other_refs: Vec<&str> = other_contents.iter().map(|s| s.as_str()).collect();
        let entropy = information_score(&content, &other_refs);

        // Criticality check
        let criticality = file_criticality(&source);
        let has_safety = has_safety_signal(&content);
        let effective_pinned = is_pinned
            || criticality == Criticality::Safety
            || criticality == Criticality::Critical
            || has_safety;

        let fp = simhash(&content);
        let effective_entropy = if criticality >= Criticality::Important {
            entropy.max(0.5)
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

        // Hierarchical fragmentation: extract skeleton
        if let Some(skel) = skeleton::extract_skeleton(&content, &source) {
            let skel_non_alpha = skel.chars().filter(|c| !c.is_alphabetic()).count();
            let skel_ratio = skel_non_alpha as f64 / skel.len().max(1) as f64;
            let skel_cpt = if skel_ratio > 0.4 { 5.0 } else { 4.0 };
            let skel_tc = (skel.len() as f64 / skel_cpt).max(1.0) as u32;
            frag.skeleton_content = Some(skel);
            frag.skeleton_token_count = Some(skel_tc);
        }

        // Auto-link dependencies
        self.dep_graph.auto_link(&frag_id, &content);

        let has_skeleton = frag.skeleton_content.is_some();
        let skel_tc_for_result = frag.skeleton_token_count;

        self.fragments.insert(frag_id.clone(), frag);

        // Insert into LSH index
        let slot = self.fragment_slot_ids.len();
        self.fragment_slot_ids.push(frag_id.clone());
        self.lsh_index.insert(fp, slot);

        // EGSC cache: depth-weighted DAG invalidation
        let mut stale_closure: HashSet<String> = std::iter::once(frag_id.clone()).collect();
        let mut depth_weights: HashMap<String, u32> = HashMap::new();
        depth_weights.insert(frag_id.clone(), 0);
        let mut bfs_queue: std::collections::VecDeque<(String, u32)> = self
            .dep_graph
            .reverse_deps(&frag_id)
            .into_iter()
            .map(|id| (id, 1))
            .collect();
        while let Some((id, depth)) = bfs_queue.pop_front() {
            if depth > 3 || stale_closure.contains(&id) {
                continue;
            }
            stale_closure.insert(id.clone());
            depth_weights.insert(id.clone(), depth);
            for rev in self.dep_graph.reverse_deps(&id) {
                if !stale_closure.contains(&rev) {
                    bfs_queue.push_back((rev, depth + 1));
                }
            }
        }
        let _invalidated = self
            .egsc_cache
            .invalidate_weighted(&stale_closure, &depth_weights);

        let result = IngestResult {
            fragment_id: frag_id,
            token_count: tc,
            entropy_score: (effective_entropy * 10000.0).round() / 10000.0,
            is_duplicate: false,
            duplicate_of: None,
            status: "ingested".into(),
            criticality: format!("{:?}", criticality),
            is_pinned: effective_pinned,
            total_fragments: self.fragments.len(),
            has_skeleton,
            skeleton_token_count: skel_tc_for_result,
        };
        json_to_js(&result)
    }

    /// Full optimization pipeline: IOS + PRISM + Channel Coding + Resonance + Causal.
    #[wasm_bindgen]
    pub fn optimize(&mut self, token_budget: u32, query: String) -> JsValue {
        self.total_optimizations += 1;
        self.last_query = query.clone();
        self.last_cache_feedback_eligible = false;

        // Task-type budget multiplier
        let effective_budget = if !query.is_empty() {
            let task_type = TaskType::classify(&query);
            (token_budget as f64 * task_type.budget_multiplier()) as u32
        } else {
            token_budget
        };
        self.last_effective_budget = effective_budget;

        // ── RAVEN-UCB Exploration ──
        let alpha_0 = 2.0_f64;
        let should_explore = if self.exploration_rate > 0.0 {
            let mut x = self.rng_state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.rng_state = x;
            (x % 10000) as f64 / 10000.0 < self.exploration_rate
        } else {
            false
        };

        // ── EGSC Cache lookup ──
        if !query.is_empty() && !should_explore {
            let current_frag_ids: HashSet<String> = self.fragments.keys().cloned().collect();
            match self
                .egsc_cache
                .lookup_with_budget(&query, &current_frag_ids, effective_budget)
            {
                CacheLookup::ExactHit {
                    response,
                    tokens_saved,
                } => {
                    self.total_tokens_saved += tokens_saved as u64;
                    if let Ok(cached) = serde_json::from_str::<serde_json::Value>(&response) {
                        let mut result = cached.clone();
                        if let Some(obj) = result.as_object_mut() {
                            obj.insert("cache_hit".into(), serde_json::json!(true));
                            obj.insert("cache_hit_type".into(), serde_json::json!("exact"));
                            obj.insert(
                                "cache_tokens_saved".into(),
                                serde_json::json!(tokens_saved),
                            );
                        }
                        self.last_cache_feedback_eligible = true;
                        return json_to_js(&result);
                    }
                }
                CacheLookup::SemanticHit {
                    response,
                    tokens_saved,
                    ..
                } => {
                    self.total_tokens_saved += tokens_saved as u64;
                    if let Ok(cached) = serde_json::from_str::<serde_json::Value>(&response) {
                        let mut result = cached.clone();
                        if let Some(obj) = result.as_object_mut() {
                            obj.insert("cache_hit".into(), serde_json::json!(true));
                            obj.insert("cache_hit_type".into(), serde_json::json!("semantic"));
                            obj.insert(
                                "cache_tokens_saved".into(),
                                serde_json::json!(tokens_saved),
                            );
                        }
                        self.last_cache_feedback_eligible = true;
                        return json_to_js(&result);
                    }
                }
                CacheLookup::Miss => {}
            }
        }

        // Update semantic scores from query
        if !query.is_empty() {
            let query_hash = simhash(&query);
            for frag in self.fragments.values_mut() {
                let dist = hamming_distance(query_hash, frag.simhash);
                frag.semantic_score = (1.0 - dist as f64 / 64.0).max(0.0);
            }
        }

        // Feedback multipliers
        let feedback_mults: HashMap<String, f64> = self
            .fragments
            .keys()
            .map(|fid| (fid.clone(), self.feedback.learned_value(fid)))
            .collect();

        // ── Query Persona Manifold ──
        let (archetype_weights, archetype_id) = if self.enable_query_personas && !query.is_empty() {
            let fragment_summaries: Vec<String> = self
                .fragments
                .values()
                .take(50)
                .map(|f| f.source.clone())
                .collect();
            let analysis = query::analyze_query(&query, &fragment_summaries);
            let tfidf_scores: Vec<f64> = analysis
                .key_terms
                .iter()
                .enumerate()
                .map(|(i, _)| 1.0 / (i as f64 + 1.0))
                .collect();
            let features = query_persona::build_query_features(
                &tfidf_scores,
                analysis.vagueness_score,
                query.len(),
                analysis.key_terms.len(),
                analysis.needs_refinement,
            );
            let (aid, weights, _) = self.query_manifold.assign(&features);
            (Some(weights), Some(aid))
        } else {
            (None, None)
        };
        self.last_archetype_id = archetype_id;

        let frags: Vec<ContextFragment> = self.fragments.values().cloned().collect();
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

        // ── Initial knapsack for dep boosts + λ* ──
        let result1 = knapsack_optimize(
            &frags,
            effective_budget,
            &weights,
            &feedback_mults,
            self.gradient_temperature,
        );
        self.last_lambda_star = result1.lambda_star;
        self.last_dual_gap = result1.dual_gap;
        let initial_selected_ids: HashSet<String> = result1
            .selected_indices
            .iter()
            .map(|&i| frags[i].fragment_id.clone())
            .collect();
        let dep_boosts = self.dep_graph.compute_dep_boosts(&initial_selected_ids);

        // ── Context Resonance bonuses ──
        let resonance_bonuses: HashMap<String, f64> =
            if self.enable_resonance && !self.resonance_matrix.is_empty() {
                let selected_refs: Vec<&str> =
                    initial_selected_ids.iter().map(|s| s.as_str()).collect();
                let candidate_refs: Vec<&str> = frags
                    .iter()
                    .filter(|f| !initial_selected_ids.contains(&f.fragment_id))
                    .map(|f| f.fragment_id.as_str())
                    .collect();
                self.resonance_matrix
                    .batch_resonance_bonuses(&candidate_refs, &selected_refs)
            } else {
                HashMap::new()
            };

        // ── Causal graph bonuses ──
        let causal_gravity: HashMap<String, f64> =
            if self.enable_causal && !self.causal_graph.is_empty() {
                let refs: Vec<&str> = frags.iter().map(|f| f.fragment_id.as_str()).collect();
                self.causal_graph.gravity_bonuses(&refs)
            } else {
                HashMap::new()
            };
        let causal_temporal: HashMap<String, f64> = if self.enable_causal
            && !self.causal_graph.is_empty()
        {
            let refs: Vec<&str> = frags.iter().map(|f| f.fragment_id.as_str()).collect();
            let prev_refs: Vec<&str> = self.prev_selected_ids.iter().map(|s| s.as_str()).collect();
            self.causal_graph.temporal_bonuses(&refs, &prev_refs)
        } else {
            HashMap::new()
        };

        // ── Coverage Estimator ──
        let semantic_threshold = 0.15;
        let semantic_candidate_ids: HashSet<String> = frags
            .iter()
            .filter(|f| f.semantic_score > semantic_threshold)
            .map(|f| f.fragment_id.clone())
            .collect();
        let structural_candidate_ids: HashSet<String> = initial_selected_ids
            .iter()
            .cloned()
            .chain(dep_boosts.keys().filter(|k| dep_boosts[*k] > 0.3).cloned())
            .collect();
        let candidate_overlap = semantic_candidate_ids
            .intersection(&structural_candidate_ids)
            .count();
        self.last_semantic_candidates = semantic_candidate_ids.len();
        self.last_structural_candidates = structural_candidate_ids.len();
        self.last_candidate_overlap = candidate_overlap;

        // Apply dep boosts + resonance + causal to fragments
        let mut boosted_frags = frags.clone();
        for frag in boosted_frags.iter_mut() {
            if !initial_selected_ids.contains(&frag.fragment_id) {
                if let Some(&boost) = dep_boosts.get(&frag.fragment_id) {
                    if boost > 0.3 {
                        frag.semantic_score = (frag.semantic_score + boost * 0.5).min(1.0);
                    }
                }
                if let Some(&res_bonus) = resonance_bonuses.get(&frag.fragment_id) {
                    if res_bonus.abs() > 0.01 {
                        frag.semantic_score =
                            (frag.semantic_score + self.w_resonance * res_bonus).clamp(0.0, 1.0);
                    }
                }
            }
            if let Some(&grav) = causal_gravity.get(&frag.fragment_id) {
                frag.semantic_score = (frag.semantic_score + 0.15 * grav).clamp(0.0, 1.0);
            }
            if let Some(&temp) = causal_temporal.get(&frag.fragment_id) {
                frag.semantic_score = (frag.semantic_score + 0.10 * temp).clamp(0.0, 1.0);
            }
        }

        // ── Core Selection: IOS or legacy knapsack ──
        let mut final_indices: Vec<usize>;
        let mut skeleton_indices: Vec<usize> = Vec::new();
        let mut skeleton_tokens_used: u32 = 0;
        let mut ios_diversity_score: Option<f64> = None;
        let mut ios_resolutions: HashMap<usize, Resolution> = HashMap::new();
        let mut explored_ids: Vec<String> = Vec::new();
        let mut selection_method: &str = "ios";

        if self.enable_ios {
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
                self.min_relevance,
            );

            final_indices = Vec::new();
            for (idx, resolution) in &ios_result.selections {
                match resolution {
                    Resolution::Full => final_indices.push(*idx),
                    Resolution::Skeleton | Resolution::Reference => skeleton_indices.push(*idx),
                }
                ios_resolutions.insert(*idx, *resolution);
            }
            skeleton_tokens_used = ios_result.total_tokens.saturating_sub(
                final_indices
                    .iter()
                    .map(|&i| frags[i].token_count)
                    .sum::<u32>(),
            );
            ios_diversity_score = Some(ios_result.diversity_score);

            // IOS-consistent λ* for REINFORCE
            if self.gradient_temperature >= 0.05 {
                let ios_scored: Vec<(usize, f64)> = boosted_frags
                    .iter()
                    .enumerate()
                    .filter_map(|(i, f)| {
                        if f.is_pinned {
                            return None;
                        }
                        let fm = feedback_mults.get(&f.fragment_id).copied().unwrap_or(1.0);
                        let s = (weights.recency * f.recency_score
                            + weights.frequency * f.frequency_score
                            + weights.semantic * f.semantic_score
                            + weights.entropy * f.entropy_score)
                            * fm.max(0.01);
                        if s > 0.0 && f.token_count > 0 {
                            Some((i, s))
                        } else {
                            None
                        }
                    })
                    .collect();
                self.last_lambda_star = compute_lambda_star(
                    &ios_scored,
                    &boosted_frags,
                    ios_result.total_tokens,
                    self.gradient_temperature,
                );
            }

            // Exploration swap (IOS path)
            if frags.len() > final_indices.len() + skeleton_indices.len()
                && !final_indices.is_empty()
                && should_explore
            {
                let selected_all: HashSet<usize> = final_indices
                    .iter()
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
                            let fm = feedback_mults
                                .get(&frags[idx].fragment_id)
                                .copied()
                                .unwrap_or(1.0);
                            let rel = compute_relevance(
                                &frags[idx],
                                self.w_recency,
                                self.w_frequency,
                                self.w_semantic,
                                self.w_entropy,
                                fm,
                            );
                            if rel < min_rel {
                                min_rel = rel;
                                min_pos = Some(pos);
                            }
                        }
                    }
                    if let Some(pos) = min_pos {
                        let explore_idx = if self.exploration_rate > 0.0 && unselected.len() > 1 {
                            let mut x = self.rng_state;
                            x ^= x << 13;
                            x ^= x >> 7;
                            x ^= x << 17;
                            self.rng_state = x;
                            unselected[(x as usize) % unselected.len()]
                        } else {
                            *unselected
                                .iter()
                                .max_by(|&&a, &&b| {
                                    self.feedback
                                        .ucb_score(&frags[a].fragment_id, alpha_0)
                                        .partial_cmp(
                                            &self
                                                .feedback
                                                .ucb_score(&frags[b].fragment_id, alpha_0),
                                        )
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                })
                                .unwrap()
                        };
                        if frags[explore_idx].token_count
                            <= frags[final_indices[pos]].token_count + 100
                        {
                            explored_ids.push(frags[explore_idx].fragment_id.clone());
                            final_indices[pos] = explore_idx;
                            self.total_explorations += 1;
                        }
                    }
                }
            }
        } else {
            // Legacy knapsack path
            selection_method = "legacy_knapsack";
            let result = if dep_boosts.values().any(|&b| b > 0.3) {
                knapsack_optimize(
                    &boosted_frags,
                    effective_budget,
                    &weights,
                    &feedback_mults,
                    self.gradient_temperature,
                )
            } else {
                result1
            };
            final_indices = result.selected_indices.clone();

            // Legacy exploration
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
                            let fm = feedback_mults
                                .get(&frags[idx].fragment_id)
                                .copied()
                                .unwrap_or(1.0);
                            let rel = compute_relevance(
                                &frags[idx],
                                self.w_recency,
                                self.w_frequency,
                                self.w_semantic,
                                self.w_entropy,
                                fm,
                            );
                            if rel < min_rel {
                                min_rel = rel;
                                min_pos = Some(pos);
                            }
                        }
                    }
                    if let Some(pos) = min_pos {
                        let explore_idx = *unselected
                            .iter()
                            .max_by(|&&a, &&b| {
                                self.feedback
                                    .ucb_score(&frags[a].fragment_id, alpha_0)
                                    .partial_cmp(
                                        &self.feedback.ucb_score(&frags[b].fragment_id, alpha_0),
                                    )
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .unwrap();
                        if frags[explore_idx].token_count
                            <= frags[final_indices[pos]].token_count + 100
                        {
                            explored_ids.push(frags[explore_idx].fragment_id.clone());
                            final_indices[pos] = explore_idx;
                            self.total_explorations += 1;
                        }
                    }
                }
            }

            // Skeleton substitution pass
            let full_tokens_legacy: u32 = final_indices.iter().map(|&i| frags[i].token_count).sum();
            let selected_set: HashSet<usize> = final_indices.iter().copied().collect();
            let mut unselected_with_skel: Vec<(usize, f64)> = (0..frags.len())
                .filter(|i| !selected_set.contains(i) && frags[*i].skeleton_token_count.is_some())
                .map(|i| {
                    let fm = feedback_mults
                        .get(&frags[i].fragment_id)
                        .copied()
                        .unwrap_or(1.0);
                    (
                        i,
                        compute_relevance(
                            &frags[i],
                            self.w_recency,
                            self.w_frequency,
                            self.w_semantic,
                            self.w_entropy,
                            fm,
                        ),
                    )
                })
                .collect();
            unselected_with_skel.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut skel_budget = effective_budget.saturating_sub(full_tokens_legacy);
            for (idx, _) in unselected_with_skel {
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
        if self.enable_channel_coding {
            let used_full: u32 = final_indices
                .iter()
                .map(|&i| boosted_frags[i].token_count)
                .sum();
            let token_gap = effective_budget.saturating_sub(used_full + skeleton_tokens_used);
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

        // Mark accessed
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

        // Sufficiency
        let selected_id_set: HashSet<String> = final_indices
            .iter()
            .chain(skeleton_indices.iter())
            .map(|&i| frags[i].fragment_id.clone())
            .collect();
        let sufficiency = self.compute_sufficiency(&frags, &final_indices);

        // ── Spectral Contradiction Guard ──
        let contradictions_evicted;
        let final_indices = if self.enable_channel_coding {
            let pre_relevances: Vec<f64> = final_indices
                .iter()
                .map(|&i| {
                    let fm = feedback_mults
                        .get(&frags[i].fragment_id)
                        .copied()
                        .unwrap_or(1.0);
                    compute_relevance(
                        &frags[i],
                        self.w_recency,
                        self.w_frequency,
                        self.w_semantic,
                        self.w_entropy,
                        fm,
                    )
                })
                .collect();
            let (filtered, report) =
                channel::contradiction_guard(&frags, &final_indices, &pre_relevances, 0.25, 0.60);
            contradictions_evicted = report.pairs_found;
            filtered
        } else {
            contradictions_evicted = 0;
            final_indices
        };

        // ── Context ordering ──
        let ordered_indices = if self.enable_channel_coding {
            let relevances: Vec<f64> = final_indices
                .iter()
                .map(|&i| {
                    let fm = feedback_mults
                        .get(&frags[i].fragment_id)
                        .copied()
                        .unwrap_or(1.0);
                    compute_relevance(
                        &frags[i],
                        self.w_recency,
                        self.w_frequency,
                        self.w_semantic,
                        self.w_entropy,
                        fm,
                    )
                })
                .collect();
            let interleaved =
                channel::semantic_interleave(&frags, &final_indices, &relevances, &self.dep_graph);
            let rel_map: std::collections::HashMap<usize, f64> = final_indices
                .iter()
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
                let rel_a = compute_relevance(
                    fa,
                    self.w_recency,
                    self.w_frequency,
                    self.w_semantic,
                    self.w_entropy,
                    fm_a,
                );
                let rel_b = compute_relevance(
                    fb,
                    self.w_recency,
                    self.w_frequency,
                    self.w_semantic,
                    self.w_entropy,
                    fm_b,
                );
                let prio_a = compute_ordering_priority(rel_a, crit_a, fa.is_pinned, dep_count_a);
                let prio_b = compute_ordering_priority(rel_b, crit_b, fb.is_pinned, dep_count_b);
                prio_b
                    .partial_cmp(&prio_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            ordered
        };

        // ── Build Explainability Snapshot ──
        let mut fragment_scores_snap: Vec<ExplainScore> = Vec::with_capacity(frags.len());
        for frag in frags.iter() {
            let fm = feedback_mults
                .get(&frag.fragment_id)
                .copied()
                .unwrap_or(1.0);
            let db = dep_boosts.get(&frag.fragment_id).copied().unwrap_or(0.0);
            let crit = file_criticality(&frag.source);
            let composite = compute_relevance(
                frag,
                self.w_recency,
                self.w_frequency,
                self.w_semantic,
                self.w_entropy,
                fm,
            );
            let is_selected = selected_id_set.contains(&frag.fragment_id);
            let is_explored = explored_ids.contains(&frag.fragment_id);
            let reason = if frag.is_pinned {
                "pinned/critical".into()
            } else if is_explored {
                "ε-exploration".into()
            } else if is_selected && db > 0.3 {
                format!("dep-boosted (boost={:.2})", db)
            } else if is_selected {
                "knapsack-optimal".into()
            } else if composite < self.min_relevance {
                "below min relevance".into()
            } else {
                "budget exceeded".into()
            };
            fragment_scores_snap.push(ExplainScore {
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
            fragment_scores: fragment_scores_snap,
            sufficiency,
            explored_ids: explored_ids.clone(),
        });

        // ── Context efficiency ──
        let context_efficiency = if final_tokens > 0 {
            let weighted_entropy: f64 = final_indices
                .iter()
                .map(|&i| frags[i].entropy_score * frags[i].token_count as f64)
                .sum::<f64>()
                + skeleton_indices
                    .iter()
                    .map(|&i| {
                        let actual_tc = match ios_resolutions.get(&i) {
                            Some(&Resolution::Reference) => {
                                (frags[i].source.len() as u32 / 4).clamp(3, 10)
                            }
                            _ => frags[i]
                                .skeleton_token_count
                                .unwrap_or(frags[i].token_count),
                        };
                        frags[i].entropy_score * actual_tc as f64
                    })
                    .sum::<f64>();
            weighted_entropy / final_tokens as f64
        } else {
            0.0
        };

        // ── Coverage estimate ──
        let coverage_est = resonance::estimate_coverage(
            ordered_indices.len() + skeleton_indices.len(),
            self.last_semantic_candidates,
            self.last_structural_candidates,
            self.last_candidate_overlap,
        );

        // Total relevance
        let total_rel: f64 = final_indices
            .iter()
            .chain(skeleton_indices.iter())
            .map(|&i| {
                let fm = feedback_mults
                    .get(&frags[i].fragment_id)
                    .copied()
                    .unwrap_or(1.0);
                compute_relevance(
                    &frags[i],
                    self.w_recency,
                    self.w_frequency,
                    self.w_semantic,
                    self.w_entropy,
                    fm,
                )
            })
            .sum();

        // ── Build selected list ──
        let mut selected_json: Vec<serde_json::Value> = Vec::new();
        for &idx in &ordered_indices {
            let f = &frags[idx];
            let fm = feedback_mults.get(&f.fragment_id).copied().unwrap_or(1.0);
            let rel = compute_relevance(
                f,
                self.w_recency,
                self.w_frequency,
                self.w_semantic,
                self.w_entropy,
                fm,
            );
            let preview = if f.content.len() > 100 {
                let mut end = 100;
                while end < f.content.len() && !f.content.is_char_boundary(end) {
                    end += 1;
                }
                format!("{}...", &f.content[..end])
            } else {
                f.content.clone()
            };
            selected_json.push(serde_json::json!({
                "id": f.fragment_id, "source": f.source, "token_count": f.token_count,
                "variant": "full", "relevance": (rel * 10000.0).round() / 10000.0,
                "entropy_score": (f.entropy_score * 10000.0).round() / 10000.0, "preview": preview,
            }));
        }
        for &idx in &skeleton_indices {
            let f = &frags[idx];
            let resolution = ios_resolutions
                .get(&idx)
                .copied()
                .unwrap_or(Resolution::Skeleton);
            let fm = feedback_mults.get(&f.fragment_id).copied().unwrap_or(1.0);
            let rel = compute_relevance(
                f,
                self.w_recency,
                self.w_frequency,
                self.w_semantic,
                self.w_entropy,
                fm,
            );
            if resolution == Resolution::Reference {
                let ref_tokens = (f.source.len() as u32 / 4).clamp(3, 10);
                selected_json.push(serde_json::json!({
                    "id": f.fragment_id, "source": f.source, "token_count": ref_tokens,
                    "variant": "reference", "relevance": (rel * 10000.0).round() / 10000.0,
                    "entropy_score": (f.entropy_score * 10000.0).round() / 10000.0,
                    "preview": format!("[ref] {}", f.source),
                }));
            } else if let (Some(ref skel), Some(stc)) =
                (&f.skeleton_content, f.skeleton_token_count)
            {
                let preview = if skel.len() > 100 {
                    let mut end = 100;
                    while end < skel.len() && !skel.is_char_boundary(end) {
                        end += 1;
                    }
                    format!("{}...", &skel[..end])
                } else {
                    skel.clone()
                };
                selected_json.push(serde_json::json!({
                    "id": f.fragment_id, "source": f.source, "token_count": stc,
                    "variant": "skeleton", "relevance": (rel * 10000.0).round() / 10000.0,
                    "entropy_score": (f.entropy_score * 10000.0).round() / 10000.0, "preview": preview,
                }));
            }
        }

        // ── Build result ──
        let cache_eligible = !query.is_empty() && explored_ids.is_empty();
        self.last_cache_feedback_eligible = cache_eligible;

        let mut result = serde_json::json!({
            "method": selection_method, "total_tokens": final_tokens,
            "context_efficiency": (context_efficiency * 10000.0).round() / 10000.0,
            "total_relevance": (total_rel * 10000.0).round() / 10000.0,
            "selected_count": ordered_indices.len() + skeleton_indices.len(),
            "skeleton_count": skeleton_indices.len(), "skeleton_tokens": skeleton_tokens_used,
            "tokens_saved": saved, "effective_budget": effective_budget,
            "budget_utilization": if effective_budget > 0 { (final_tokens as f64 / effective_budget as f64 * 10000.0).round() / 10000.0 } else { 0.0 },
            "sufficiency": (sufficiency * 10000.0).round() / 10000.0,
            "selected": selected_json, "cache_hit": false, "cache_eligible": cache_eligible,
            "optimization_policy": if explored_ids.is_empty() { "exploit" } else { "explore" },
            "coverage": (coverage_est.coverage * 10000.0).round() / 10000.0,
            "coverage_confidence": (coverage_est.confidence * 10000.0).round() / 10000.0,
            "coverage_gap": coverage_est.estimated_gap.round(),
            "coverage_risk": coverage_est.risk_level,
        });

        if let Some(obj) = result.as_object_mut() {
            if sufficiency < 0.7 {
                obj.insert(
                    "sufficiency_warning".into(),
                    serde_json::json!(format!(
                        "Only {:.0}% of referenced symbols have definitions in context",
                        sufficiency * 100.0
                    )),
                );
            }
            if !explored_ids.is_empty() {
                obj.insert("explored".into(), serde_json::json!(explored_ids));
            }
            if contradictions_evicted > 0 {
                obj.insert(
                    "contradictions_evicted".into(),
                    serde_json::json!(contradictions_evicted),
                );
            }
            if let Some(div_score) = ios_diversity_score {
                obj.insert("ios_diversity_score".into(), serde_json::json!(div_score));
                obj.insert("ios_enabled".into(), serde_json::json!(true));
            }
            if self.enable_resonance {
                obj.insert(
                    "resonance_pairs".into(),
                    serde_json::json!(self.resonance_matrix.len()),
                );
                obj.insert(
                    "w_resonance".into(),
                    serde_json::json!((self.w_resonance * 10000.0).round() / 10000.0),
                );
            }
            if self.enable_causal && !self.causal_graph.is_empty() {
                let cs = self.causal_graph.stats();
                obj.insert(
                    "causal_tracked".into(),
                    serde_json::json!(cs.tracked_fragments),
                );
                obj.insert(
                    "causal_temporal_links".into(),
                    serde_json::json!(cs.temporal_links),
                );
            }
        }

        // ── EGSC Cache store ──
        if cache_eligible {
            let current_frag_ids: HashSet<String> = self.fragments.keys().cloned().collect();
            let entropies: Vec<(f64, u32)> = final_indices
                .iter()
                .chain(skeleton_indices.iter())
                .map(|&i| (frags[i].entropy_score, frags[i].token_count))
                .collect();
            self.egsc_cache.store_with_budget(
                &query,
                current_frag_ids,
                &entropies,
                result.to_string(),
                final_tokens,
                self.current_turn,
                effective_budget,
            );
        }

        // ── Causal state tracking ──
        if self.enable_causal {
            self.prev_selected_ids = ordered_indices
                .iter()
                .chain(skeleton_indices.iter())
                .map(|&i| frags[i].fragment_id.clone())
                .collect();
            self.prev_explored_ids = explored_ids;
        }

        json_to_js(&result)
    }

    /// Record successful outcome for specific fragment IDs.
    #[wasm_bindgen]
    pub fn record_success(&mut self, fragment_ids_json: &str) {
        let fragment_ids: Vec<String> = serde_json::from_str(fragment_ids_json).unwrap_or_default();
        if fragment_ids.is_empty() {
            return;
        }
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
        let raw_reward = if self.enable_channel_coding {
            let suff = self
                .last_optimization
                .as_ref()
                .map(|s| s.sufficiency)
                .unwrap_or(0.7);
            channel::modulated_reward(true, suff)
        } else {
            1.0
        };
        let advantage = raw_reward - self.reward_baseline_ema;
        self.reward_baseline_ema = 0.9 * self.reward_baseline_ema + 0.1 * raw_reward;
        if self.enable_resonance {
            self.resonance_matrix
                .record_outcome(&fragment_ids, advantage, self.current_turn);
        }
        if self.enable_causal {
            let query_hash = simhash(&self.last_query);
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

    /// Record failed outcome for specific fragment IDs.
    #[wasm_bindgen]
    pub fn record_failure(&mut self, fragment_ids_json: &str) {
        let fragment_ids: Vec<String> = serde_json::from_str(fragment_ids_json).unwrap_or_default();
        if fragment_ids.is_empty() {
            return;
        }
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
        let raw_reward = if self.enable_channel_coding {
            let suff = self
                .last_optimization
                .as_ref()
                .map(|s| s.sufficiency)
                .unwrap_or(0.7);
            channel::modulated_reward(false, suff)
        } else {
            -1.0
        };
        let advantage = raw_reward - self.reward_baseline_ema;
        self.reward_baseline_ema = 0.9 * self.reward_baseline_ema + 0.1 * raw_reward;
        if self.enable_resonance {
            self.resonance_matrix
                .record_outcome(&fragment_ids, advantage, self.current_turn);
        }
        if self.enable_causal {
            let query_hash = simhash(&self.last_query);
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

    /// Record a continuous reward signal.
    #[wasm_bindgen]
    pub fn record_reward(&mut self, fragment_ids_json: &str, reward: f64) {
        let fragment_ids: Vec<String> = serde_json::from_str(fragment_ids_json).unwrap_or_default();
        let r = if reward.is_finite() { reward } else { 0.0 };
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
        let advantage = r - self.reward_baseline_ema;
        self.reward_baseline_ema = 0.9 * self.reward_baseline_ema + 0.1 * r;
        if self.enable_resonance {
            self.resonance_matrix
                .record_outcome(&fragment_ids, advantage, self.current_turn);
        }
        if self.enable_causal {
            let query_hash = simhash(&self.last_query);
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

    /// Advance the turn counter and apply Ebbinghaus decay.
    #[wasm_bindgen]
    pub fn advance_turn(&mut self) {
        self.current_turn += 1;
        let decay_rate = (2.0_f64).ln() / self.decay_half_life.max(1) as f64;
        for frag in self.fragments.values_mut() {
            let dt = self.current_turn.saturating_sub(frag.turn_last_accessed) as f64;
            frag.recency_score = (-decay_rate * dt).exp();
        }
        let to_evict: Vec<String> = self
            .fragments
            .iter()
            .filter(|(_, f)| f.recency_score < self.min_relevance && !f.is_pinned)
            .map(|(k, _)| k.clone())
            .collect();
        for id in &to_evict {
            self.dedup_index.remove(id);
        }
        self.fragments
            .retain(|_, f| f.recency_score >= self.min_relevance || f.is_pinned);
        self.rebuild_lsh_index();
        if self.enable_query_personas {
            self.query_manifold.lifecycle_tick();
        }
        self.egsc_cache.gc(0.15);
        if self.enable_resonance {
            self.resonance_matrix.decay_tick();
        }
        if self.enable_causal {
            self.causal_graph.decay_tick(self.current_turn);
        }
        // Maxwell's Demon consolidation every 5 turns
        if self.current_turn % 5 == 0 && self.fragments.len() > 10 {
            let frag_data: Vec<(String, u64, f64, bool, u32)> = self
                .fragments
                .values()
                .map(|f| {
                    let fm = self.feedback.learned_value(&f.fragment_id);
                    (
                        f.fragment_id.clone(),
                        f.simhash,
                        fm,
                        f.is_pinned,
                        f.token_count,
                    )
                })
                .collect();
            let groups = resonance::find_consolidation_groups(
                &frag_data,
                self.consolidation_hamming_threshold,
            );
            for group in &groups {
                let total_access: u32 = group
                    .consolidated_ids
                    .iter()
                    .filter_map(|id| self.fragments.get(id))
                    .map(|f| f.access_count)
                    .sum();
                if let Some(winner) = self.fragments.get_mut(&group.winner_id) {
                    winner.access_count += total_access;
                }
                for loser_id in &group.consolidated_ids {
                    self.fragments.remove(loser_id);
                    self.dedup_index.remove(loser_id);
                }
                self.total_consolidations += group.consolidated_ids.len() as u64;
                self.consolidation_tokens_saved += group.tokens_saved as u64;
            }
            if !groups.is_empty() {
                self.rebuild_lsh_index();
            }
        }
    }

    /// Semantic recall of relevant fragments.
    #[wasm_bindgen]
    pub fn recall(&self, query: String, top_k: usize) -> JsValue {
        let query_fp = simhash(&query);
        let candidates = self.lsh_index.query(query_fp);
        let mut scored: Vec<(&ContextFragment, f64)> = if !candidates.is_empty() {
            candidates
                .iter()
                .filter_map(|&slot| {
                    let frag_id = self.fragment_slot_ids.get(slot)?;
                    self.fragments.get(frag_id)
                })
                .map(|f| {
                    let dist = hamming_distance(query_fp, f.simhash);
                    let fm = self.feedback.learned_value(&f.fragment_id);
                    let rel = self.context_scorer.score(
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
            self.fragments
                .values()
                .map(|f| {
                    let dist = hamming_distance(query_fp, f.simhash);
                    let fm = self.feedback.learned_value(&f.fragment_id);
                    let rel = self.context_scorer.score(
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
        let results: Vec<serde_json::Value> = scored
            .iter()
            .map(|(f, rel)| {
                serde_json::json!({
                    "fragment_id": f.fragment_id, "source": f.source,
                    "relevance": (*rel * 10000.0).round() / 10000.0,
                    "entropy": (f.entropy_score * 10000.0).round() / 10000.0,
                    "content": f.content,
                })
            })
            .collect();
        json_to_js(&serde_json::Value::Array(results))
    }

    /// Remove a fragment by ID.
    #[wasm_bindgen]
    pub fn remove(&mut self, fragment_id: &str) -> bool {
        self.dedup_index.remove(fragment_id);
        self.fragments.remove(fragment_id).is_some()
    }

    /// Get full engine statistics as JSON.
    #[wasm_bindgen]
    pub fn stats(&mut self) -> JsValue {
        let total_tokens: u32 = self.fragments.values().map(|f| f.token_count).sum();
        let avg_entropy = if self.fragments.is_empty() {
            0.0
        } else {
            self.fragments
                .values()
                .map(|f| f.entropy_score)
                .sum::<f64>()
                / self.fragments.len() as f64
        };
        let pinned = self.fragments.values().filter(|f| f.is_pinned).count();
        let cs = self.egsc_cache.stats();
        let total_exploitations = self
            .total_optimizations
            .saturating_sub(self.total_explorations);

        let result = serde_json::json!({
            "session": {
                "current_turn": self.current_turn,
                "total_fragments": self.fragments.len(),
                "total_tokens_tracked": total_tokens,
                "avg_entropy": (avg_entropy * 10000.0).round() / 10000.0,
                "pinned": pinned,
            },
            "savings": {
                "total_tokens_saved": self.total_tokens_saved,
                "total_duplicates_caught": self.total_duplicates_caught,
                "total_optimizations": self.total_optimizations,
                "total_fragments_ingested": self.total_fragments_ingested,
                "estimated_cost_saved_usd": (self.total_tokens_saved as f64 * 0.000003 * 10000.0).round() / 10000.0,
            },
            "dedup": { "indexed_fragments": self.dedup_index.size(), "duplicates_detected": self.dedup_index.duplicates_detected },
            "policy": {
                "configured_exploration_rate": self.exploration_rate,
                "total_explorations": self.total_explorations,
                "total_exploitations": total_exploitations,
                "explore_ratio": if self.total_optimizations > 0 { self.total_explorations as f64 / self.total_optimizations as f64 } else { 0.0 },
            },
            "prism": {
                "w_recency": (self.w_recency * 10000.0).round() / 10000.0,
                "w_frequency": (self.w_frequency * 10000.0).round() / 10000.0,
                "w_semantic": (self.w_semantic * 10000.0).round() / 10000.0,
                "w_entropy": (self.w_entropy * 10000.0).round() / 10000.0,
                "gradient_temperature": (self.gradient_temperature * 10000.0).round() / 10000.0,
                "condition_number": (self.prism_optimizer.condition_number() * 100.0).round() / 100.0,
            },
            "cache": {
                "entries": cs.total_entries, "lookups": cs.total_lookups,
                "exact_hits": cs.exact_hits, "semantic_hits": cs.semantic_hits,
                "misses": cs.misses, "hit_rate": (cs.hit_rate * 10000.0).round() / 10000.0,
                "tokens_saved": cs.total_tokens_saved,
            },
            "consolidation": { "total_consolidations": self.total_consolidations, "tokens_saved": self.consolidation_tokens_saved },
        });
        json_to_js(&result)
    }

    /// Get the current turn number.
    #[wasm_bindgen]
    pub fn get_turn(&self) -> u32 {
        self.current_turn
    }

    /// Get fragment count.
    #[wasm_bindgen]
    pub fn fragment_count(&self) -> usize {
        self.fragments.len()
    }

    /// Set scoring weights (normalized to sum=1.0).
    #[wasm_bindgen]
    pub fn set_weights(
        &mut self,
        w_recency: f64,
        w_frequency: f64,
        w_semantic: f64,
        w_entropy: f64,
    ) {
        self.w_recency = w_recency.clamp(0.05, 0.80);
        self.w_frequency = w_frequency.clamp(0.05, 0.80);
        self.w_semantic = w_semantic.clamp(0.05, 0.80);
        self.w_entropy = w_entropy.clamp(0.05, 0.80);
        let sum = self.w_recency + self.w_frequency + self.w_semantic + self.w_entropy;
        if sum > 0.0 {
            self.w_recency /= sum;
            self.w_frequency /= sum;
            self.w_semantic /= sum;
            self.w_entropy /= sum;
        }
        self.context_scorer.w_recency = self.w_recency;
        self.context_scorer.w_frequency = self.w_frequency;
        self.context_scorer.w_similarity = self.w_semantic;
        self.context_scorer.w_entropy = self.w_entropy;
    }

    /// Set exploration rate.
    #[wasm_bindgen]
    pub fn set_exploration_rate(&mut self, rate: f64) {
        self.exploration_rate = rate.clamp(0.0, 1.0);
    }

    /// Enable/disable query personas.
    #[wasm_bindgen]
    pub fn set_query_personas_enabled(&mut self, enabled: bool) {
        self.enable_query_personas = enabled;
    }

    /// Enable/disable channel coding.
    #[wasm_bindgen]
    pub fn set_channel_coding_enabled(&mut self, enabled: bool) {
        self.enable_channel_coding = enabled;
    }

    /// Set cost model from model name.
    #[wasm_bindgen]
    pub fn set_model(&mut self, model_name: &str) {
        let cost_model = cache::CostModel::for_model(model_name);
        self.egsc_cache
            .set_cost_per_token(cost_model.cost_per_token);
    }

    /// Set cost-per-token directly.
    #[wasm_bindgen]
    pub fn set_cache_cost_per_token(&mut self, cost: f64) {
        self.egsc_cache.set_cost_per_token(cost);
    }

    /// Classify a task query.
    #[wasm_bindgen]
    pub fn classify_task(&self, query: &str) -> JsValue {
        let task_type = TaskType::classify(query);
        let result = serde_json::json!({ "task_type": format!("{:?}", task_type), "budget_multiplier": task_type.budget_multiplier() });
        json_to_js(&result)
    }

    /// Clear EGSC cache.
    #[wasm_bindgen]
    pub fn cache_clear(&mut self) {
        self.egsc_cache.clear();
    }
    /// Cache entry count.
    #[wasm_bindgen]
    pub fn cache_len(&self) -> usize {
        self.egsc_cache.len()
    }
    /// Cache empty check.
    #[wasm_bindgen]
    pub fn cache_is_empty(&self) -> bool {
        self.egsc_cache.is_empty()
    }
    /// Cache hit rate.
    #[wasm_bindgen]
    pub fn cache_hit_rate(&mut self) -> f64 {
        self.egsc_cache.stats().hit_rate
    }

    /// Dependency graph stats.
    #[wasm_bindgen]
    pub fn dep_graph_stats(&self) -> JsValue {
        let result = serde_json::json!({ "nodes": self.dep_graph.node_count(), "edges": self.dep_graph.edge_count() });
        json_to_js(&result)
    }

    /// Query manifold stats.
    #[wasm_bindgen]
    pub fn query_manifold_stats(&self) -> JsValue {
        let ms = self.query_manifold.stats();
        let result = serde_json::json!({
            "enabled": self.enable_query_personas, "population": ms.population,
            "total_births": ms.total_births, "total_deaths": ms.total_deaths,
            "total_fusions": ms.total_fusions, "tick": ms.tick
        });
        json_to_js(&result)
    }

    /// Explain why each fragment was selected/excluded in last optimization.
    #[wasm_bindgen]
    pub fn explain_selection(&self) -> JsValue {
        let snapshot = match &self.last_optimization {
            Some(s) => s,
            None => {
                let r = serde_json::json!({"error": "No optimization has been run yet"});
                return json_to_js(&r);
            }
        };
        let included: Vec<serde_json::Value> = snapshot.fragment_scores.iter().filter(|fs| fs.selected)
            .map(|fs| serde_json::json!({
                "id": fs.fragment_id, "source": fs.source, "decision": "included",
                "scores": { "recency": fs.recency, "frequency": fs.frequency, "semantic": fs.semantic,
                    "entropy": fs.entropy, "feedback_mult": fs.feedback_mult, "dep_boost": fs.dep_boost,
                    "criticality": fs.criticality, "composite": fs.composite },
                "reason": fs.reason,
            })).collect();
        let excluded: Vec<serde_json::Value> = snapshot.fragment_scores.iter().filter(|fs| !fs.selected)
            .map(|fs| serde_json::json!({
                "id": fs.fragment_id, "source": fs.source, "decision": "excluded",
                "scores": { "recency": fs.recency, "frequency": fs.frequency, "semantic": fs.semantic,
                    "entropy": fs.entropy, "feedback_mult": fs.feedback_mult, "dep_boost": fs.dep_boost,
                    "criticality": fs.criticality, "composite": fs.composite },
                "reason": fs.reason,
            })).collect();
        let result = serde_json::json!({
            "sufficiency": (snapshot.sufficiency * 10000.0).round() / 10000.0,
            "included": included, "excluded": excluded, "explored": snapshot.explored_ids,
        });
        json_to_js(&result)
    }

    /// Scan a fragment for security vulnerabilities.
    #[wasm_bindgen]
    pub fn scan_fragment(&self, fragment_id: &str) -> JsValue {
        let frag = match self.fragments.get(fragment_id) {
            Some(f) => f,
            None => {
                let r =
                    serde_json::json!({"error": format!("Fragment '{}' not found", fragment_id)});
                return json_to_js(&r);
            }
        };
        let report = sast::scan_content(&frag.content, &frag.source);
        {
            let r = serde_json::to_value(&report).unwrap_or(serde_json::Value::Null);
            json_to_js(&r)
        }
    }

    /// Security report for all fragments.
    #[wasm_bindgen]
    pub fn security_report(&self) -> JsValue {
        let mut all_findings: Vec<serde_json::Value> = Vec::new();
        let mut critical_total = 0usize;
        let mut max_risk: f64 = 0.0;
        let mut most_vulnerable = String::new();
        for (fid, frag) in &self.fragments {
            let report = sast::scan_content(&frag.content, &frag.source);
            if report.findings.is_empty() {
                continue;
            }
            critical_total += report.critical_count;
            if report.risk_score > max_risk {
                max_risk = report.risk_score;
                most_vulnerable = fid.clone();
            }
            all_findings.push(serde_json::json!({
                "fragment_id": fid, "source": frag.source, "risk_score": report.risk_score,
                "critical_count": report.critical_count, "finding_count": report.findings.len(), "top_fix": report.top_fix,
            }));
        }
        let result = serde_json::json!({
            "fragments_scanned": self.fragments.len(), "fragments_with_findings": all_findings.len(),
            "critical_total": critical_total, "max_risk_score": (max_risk * 100.0).round() / 100.0,
            "most_vulnerable_fragment": most_vulnerable, "vulnerable_fragments": all_findings,
        });
        json_to_js(&result)
    }

    /// Analyze codebase health (A-F grade).
    #[wasm_bindgen]
    pub fn analyze_health(&self) -> JsValue {
        let frags: Vec<&ContextFragment> = self.fragments.values().collect();
        let report = health::analyze_health(&frags, &self.dep_graph);
        let json_str = serde_json::to_string(&report).unwrap_or_default();
        js_sys::JSON::parse(&json_str).unwrap_or(JsValue::NULL)
    }

    /// Detect entropy anomalies.
    #[wasm_bindgen]
    pub fn entropy_anomalies(&self) -> JsValue {
        let frags: Vec<&ContextFragment> = self.fragments.values().collect();
        let report = anomaly::scan_anomalies(&frags);
        let json_str = serde_json::to_string(&report).unwrap_or_default();
        js_sys::JSON::parse(&json_str).unwrap_or(JsValue::NULL)
    }

    /// Score context utilization from LLM response.
    #[wasm_bindgen]
    pub fn score_utilization(&self, response: &str) -> JsValue {
        let frags: Vec<&ContextFragment> = self.fragments.values().collect();
        let report = utilization::score_utilization(&frags, response);
        let json_str = serde_json::to_string(&report).unwrap_or_default();
        js_sys::JSON::parse(&json_str).unwrap_or(JsValue::NULL)
    }

    /// Semantic dedup report.
    #[wasm_bindgen]
    pub fn semantic_dedup_report(&self) -> JsValue {
        let frags: Vec<ContextFragment> = self.fragments.values().cloned().collect();
        if frags.is_empty() {
            return json_to_js(&serde_json::json!({"kept": 0, "removed": 0, "tokens_saved": 0}));
        }
        let sorted: Vec<usize> = (0..frags.len()).collect();
        let result = semantic_dedup::semantic_deduplicate_with_stats(&frags, &sorted, None);
        let report = serde_json::json!({
            "total_fragments": frags.len(), "kept": result.kept_indices.len(),
            "removed": result.removed_count, "tokens_saved": result.tokens_saved,
        });
        json_to_js(&report)
    }

    /// Hierarchical context compression.
    #[wasm_bindgen]
    pub fn hierarchical_compress(&self, token_budget: u32, query: String) -> JsValue {
        let mut frags: Vec<ContextFragment> = self.fragments.values().cloned().collect();
        frags.sort_by(|a, b| a.fragment_id.cmp(&b.fragment_id));
        if frags.is_empty() {
            return json_to_js(&serde_json::json!({"status": "empty"}));
        }
        let query_hash = simhash(&query);
        let mut scored: Vec<(usize, f64)> = frags
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let dist = hamming_distance(query_hash, f.simhash);
                (i, (1.0 - dist as f64 / 64.0).max(0.0))
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k: Vec<String> = scored
            .iter()
            .take(10)
            .filter(|(_, sim)| *sim > 0.1)
            .map(|(i, _)| frags[*i].fragment_id.clone())
            .collect();
        let mean_entropy = frags.iter().map(|f| f.entropy_score).sum::<f64>() / frags.len() as f64;
        let hcc = hierarchical::hierarchical_compress(
            &frags,
            &self.dep_graph,
            &top_k,
            token_budget,
            mean_entropy,
        );
        let result = serde_json::json!({
            "status": "compressed",
            "level1_map": hcc.level1_map, "level1_tokens": hcc.budget_used.0,
            "level2_cluster": hcc.level2_cluster, "level2_tokens": hcc.budget_used.1,
            "level3_count": hcc.level3_indices.len(), "level3_tokens": hcc.budget_used.2,
            "budget_utilization": if token_budget > 0 { ((hcc.budget_used.0 + hcc.budget_used.1 + hcc.budget_used.2) as f64 / token_budget as f64 * 10000.0).round() / 10000.0 } else { 0.0 },
        });
        json_to_js(&result)
    }

    /// Export full engine state as JSON string.
    #[wasm_bindgen]
    pub fn export_state(&self) -> JsValue {
        let state = serde_json::json!({
            "w_recency": self.w_recency, "w_frequency": self.w_frequency,
            "w_semantic": self.w_semantic, "w_entropy": self.w_entropy,
            "current_turn": self.current_turn, "total_tokens_saved": self.total_tokens_saved,
            "total_optimizations": self.total_optimizations, "total_fragments_ingested": self.total_fragments_ingested,
            "total_duplicates_caught": self.total_duplicates_caught, "fragment_count": self.fragments.len(),
            "gradient_temperature": self.gradient_temperature, "w_resonance": self.w_resonance,
        });
        json_to_js(&state)
    }

    /// Import engine state from JSON string.
    #[wasm_bindgen]
    pub fn import_state(&mut self, json_str: &str) -> JsValue {
        match serde_json::from_str::<serde_json::Value>(json_str) {
            Ok(state) => {
                if let Some(v) = state.get("w_recency").and_then(|v| v.as_f64()) {
                    self.w_recency = v;
                }
                if let Some(v) = state.get("w_frequency").and_then(|v| v.as_f64()) {
                    self.w_frequency = v;
                }
                if let Some(v) = state.get("w_semantic").and_then(|v| v.as_f64()) {
                    self.w_semantic = v;
                }
                if let Some(v) = state.get("w_entropy").and_then(|v| v.as_f64()) {
                    self.w_entropy = v;
                }
                if let Some(v) = state.get("gradient_temperature").and_then(|v| v.as_f64()) {
                    self.gradient_temperature = v;
                }
                json_to_js(&serde_json::json!({"status": "imported"}))
            }
            Err(e) => json_to_js(&serde_json::json!({"error": e.to_string()})),
        }
    }

    /// Export all fragments as JSON.
    #[wasm_bindgen]
    pub fn export_fragments(&self) -> JsValue {
        let frags: Vec<serde_json::Value> = self.fragments.values().map(|f| {
            serde_json::json!({
                "fragment_id": f.fragment_id, "content": f.content, "token_count": f.token_count,
                "source": f.source, "is_pinned": f.is_pinned, "recency_score": f.recency_score,
                "frequency_score": f.frequency_score, "semantic_score": f.semantic_score,
                "entropy_score": f.entropy_score, "turn_created": f.turn_created,
                "turn_last_accessed": f.turn_last_accessed, "access_count": f.access_count, "simhash": f.simhash,
            })
        }).collect();
        json_to_js(&serde_json::Value::Array(frags))
    }

    /// Clear all fragments and reset engine.
    #[wasm_bindgen]
    pub fn clear(&mut self) {
        self.fragments.clear();
        self.fragment_slot_ids.clear();
        self.dedup_index = DedupIndex::new(self.hamming_threshold);
        self.dep_graph = DepGraph::new();
        self.egsc_cache.clear();
        self.lsh_index.clear();
        self.last_optimization = None;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Non-wasm-bindgen implementation methods (private helpers)
// ═══════════════════════════════════════════════════════════════════

impl WasmEntrolyEngine {
    /// Rebuild LSH index from current fragment map.
    fn rebuild_lsh_index(&mut self) {
        self.lsh_index.clear();
        self.fragment_slot_ids.clear();
        let mut ids: Vec<String> = self.fragments.keys().cloned().collect();
        ids.sort_unstable();
        for (slot, id) in ids.iter().enumerate() {
            if let Some(frag) = self.fragments.get(id) {
                self.lsh_index.insert(frag.simhash, slot);
            }
            self.fragment_slot_ids.push(id.clone());
        }
    }

    /// Numerically stable sigmoid.
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

    /// Compute context sufficiency.
    fn compute_sufficiency(&self, frags: &[ContextFragment], selected_indices: &[usize]) -> f64 {
        let selected_ids: HashSet<&str> = selected_indices
            .iter()
            .map(|&i| frags[i].fragment_id.as_str())
            .collect();
        let defined_symbols: HashSet<String> = self
            .dep_graph
            .symbol_definitions()
            .iter()
            .filter(|(_, fid)| selected_ids.contains(fid.as_str()))
            .map(|(symbol, _)| symbol.clone())
            .collect();
        let mut referenced_symbols: HashSet<String> = HashSet::new();
        for &idx in selected_indices {
            for ident in extract_identifiers(&frags[idx].content) {
                if self.dep_graph.has_symbol(&ident) {
                    referenced_symbols.insert(ident);
                }
            }
        }
        if referenced_symbols.is_empty() {
            return 1.0;
        }
        referenced_symbols
            .iter()
            .filter(|s| defined_symbols.contains(*s))
            .count() as f64
            / referenced_symbols.len() as f64
    }

    /// Apply PRISM RL update with REINFORCE-with-baseline policy gradient.
    fn apply_prism_rl_update(&mut self, fragment_ids: &[String], reward: f64) {
        if self.fragments.is_empty() {
            return;
        }
        let tau = self.gradient_temperature.max(0.01);
        let selected: HashSet<&str> = fragment_ids.iter().map(|s| s.as_str()).collect();
        let selected_entropy_sum: f64 = fragment_ids
            .iter()
            .filter_map(|id| self.fragments.get(id))
            .map(|f| f.entropy_score.max(0.01))
            .sum::<f64>()
            .max(0.01);
        let lambda = self.last_lambda_star;
        let selected_ids_vec: Vec<&str> = selected.iter().copied().collect();

        let mut g = [0.0_f64; 4];
        let mut g5 = [0.0_f64; 5];

        for frag in self.fragments.values_mut() {
            let score = self.w_recency * frag.recency_score
                + self.w_frequency * frag.frequency_score
                + self.w_semantic * frag.semantic_score
                + self.w_entropy * frag.entropy_score;
            let tc = frag.token_count as f64;
            let p = Self::sigmoid((score - lambda * tc) / tau);
            let dp = p * (1.0 - p) / tau;
            let action = if selected.contains(frag.fragment_id.as_str()) {
                1.0
            } else {
                0.0
            };
            let phi = if action > 0.5 {
                (frag.entropy_score.max(0.01) / selected_entropy_sum).clamp(0.1, 3.0)
            } else {
                1.0
            };
            let trace_update = (action - p) * dp;
            frag.eligibility_trace = 0.7 * frag.eligibility_trace + trace_update;
            let advantage = phi * frag.eligibility_trace * reward;
            g[prism::dim::RECENCY] += advantage * frag.recency_score;
            g[prism::dim::FREQUENCY] += advantage * frag.frequency_score;
            g[prism::dim::SEMANTIC] += advantage * frag.semantic_score;
            g[prism::dim::ENTROPY] += advantage * frag.entropy_score;
            if self.enable_resonance {
                let res_feature = self
                    .resonance_matrix
                    .resonance_bonus(frag.fragment_id.as_str(), &selected_ids_vec);
                g5[prism::dim::RECENCY] += advantage * frag.recency_score;
                g5[prism::dim::FREQUENCY] += advantage * frag.frequency_score;
                g5[prism::dim::SEMANTIC] += advantage * frag.semantic_score;
                g5[prism::dim::ENTROPY] += advantage * frag.entropy_score;
                g5[prism::dim::RESONANCE] += advantage * res_feature;
            }
        }

        // ADGT temperature adaptation
        let g_norm = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        if self.gradient_norm_ema > 1e-8 && g_norm > self.gradient_norm_ema * 3.0 {
            self.gradient_temperature = 2.0;
        } else if self.last_dual_gap > 0.0 && self.fragments.len() > 1 {
            let n = self.fragments.len() as f64;
            let gap_per_frag = self.last_dual_gap / (n * 2_f64.ln());
            let kappa = self.prism_optimizer.condition_number();
            let kappa_norm = (kappa / 2.0).clamp(0.5, 4.0);
            let natural_tau = (gap_per_frag * kappa_norm).clamp(0.1, 2.0);
            self.gradient_temperature =
                (0.90 * self.gradient_temperature + 0.10 * natural_tau).max(0.1);
        } else {
            self.gradient_temperature = (self.gradient_temperature * 0.998).max(0.1);
        }
        self.gradient_norm_ema = 0.95 * self.gradient_norm_ema + 0.05 * g_norm;

        let update = self.prism_optimizer.compute_update(&g);
        self.w_recency += update[prism::dim::RECENCY];
        self.w_frequency += update[prism::dim::FREQUENCY];
        self.w_semantic += update[prism::dim::SEMANTIC];
        self.w_entropy += update[prism::dim::ENTROPY];
        if self.enable_resonance {
            let update5 = self.prism_optimizer_5d.compute_update(&g5);
            self.w_resonance = (self.w_resonance + update5[prism::dim::RESONANCE]).clamp(0.0, 0.5);
        }
        self.w_recency = self.w_recency.clamp(0.05, 0.8);
        self.w_frequency = self.w_frequency.clamp(0.05, 0.8);
        self.w_semantic = self.w_semantic.clamp(0.05, 0.8);
        self.w_entropy = self.w_entropy.clamp(0.05, 0.8);
        let sum = self.w_recency + self.w_frequency + self.w_semantic + self.w_entropy;
        self.w_recency /= sum;
        self.w_frequency /= sum;
        self.w_semantic /= sum;
        self.w_entropy /= sum;
        self.context_scorer.w_similarity = self.w_semantic;
        self.context_scorer.w_recency = self.w_recency;
        self.context_scorer.w_entropy = self.w_entropy;
        self.context_scorer.w_frequency = self.w_frequency;
        if self.enable_query_personas {
            if let Some(ref aid) = self.last_archetype_id {
                self.query_manifold.record_result(aid, &g, reward > 0.0);
            }
        }
    }
}
