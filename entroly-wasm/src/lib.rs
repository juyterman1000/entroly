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

mod fragment;
mod knapsack;
mod knapsack_sds;
mod entropy;
mod dedup;
mod depgraph;
mod guardrails;
mod prism;
mod query;
mod nkbe;
mod cognitive_bus;
mod cache;
mod skeleton;
mod lsh;
mod sast;
mod health;
mod hierarchical;
mod anomaly;
mod channel;
mod conversation_pruner;
mod query_persona;
mod semantic_dedup;
mod resonance;
mod causal;
mod utilization;

use wasm_bindgen::prelude::*;
use serde::Serialize;
use std::collections::{HashMap, HashSet};

use fragment::{ContextFragment, compute_relevance, apply_ebbinghaus_decay};
use knapsack::{knapsack_optimize, ScoringWeights};
use entropy::information_score;
use dedup::{simhash, hamming_distance, DedupIndex};
use depgraph::{DepGraph, extract_identifiers};
use guardrails::{file_criticality, has_safety_signal, TaskType, FeedbackTracker, Criticality};
use cache::{EgscCache, EgscConfig, CacheLookup};

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

// ═══════════════════════════════════════════════════════════════════
// WasmEntrolyEngine
// ═══════════════════════════════════════════════════════════════════

#[wasm_bindgen]
pub struct WasmEntrolyEngine {
    fragments: HashMap<String, ContextFragment>,
    current_turn: u32,
    fragment_counter: u64,

    // Scoring weights
    w_recency: f64,
    w_frequency: f64,
    w_semantic: f64,
    w_entropy: f64,

    // Decay
    decay_half_life: u32,

    // Dedup
    dedup_index: DedupIndex,
    hamming_threshold: u32,

    // Dependency graph
    dep_graph: DepGraph,

    // RL feedback
    feedback: FeedbackTracker,

    // Cache
    egsc_cache: EgscCache,

    // State for feedback recording
    last_query: String,
    last_effective_budget: u32,
    last_cache_hit: bool,
}

#[wasm_bindgen]
impl WasmEntrolyEngine {
    /// Create a new Entroly engine with default parameters.
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmEntrolyEngine {
        WasmEntrolyEngine {
            fragments: HashMap::new(),
            current_turn: 0,
            fragment_counter: 0,
            w_recency: 0.30,
            w_frequency: 0.25,
            w_semantic: 0.25,
            w_entropy: 0.20,
            decay_half_life: 15,
            dedup_index: DedupIndex::new(3),
            hamming_threshold: 3,
            dep_graph: DepGraph::new(),
            feedback: FeedbackTracker::new(),
            egsc_cache: EgscCache::new(EgscConfig::default()),
            last_query: String::new(),
            last_effective_budget: 0,
            last_cache_hit: false,
        }
    }

    /// Ingest a context fragment into the engine.
    ///
    /// Returns a JSON object with fragment metadata.
    #[wasm_bindgen]
    pub fn ingest(
        &mut self,
        source: String,
        content: String,
        token_count: u32,
        is_pinned: bool,
    ) -> JsValue {
        self.current_turn += 1;
        self.fragment_counter += 1;

        let fragment_id = format!("frag_{}_{}", self.fragment_counter, self.current_turn);

        let tc = if token_count == 0 {
            (content.len() / 4).max(1) as u32
        } else {
            token_count
        };

        // Entropy scoring
        let entropy = information_score(&content, &[]);

        // Dedup check
        let dup = self.dedup_index.insert(&fragment_id, &content);
        let is_duplicate = dup.is_some();

        if is_duplicate {
            let result = IngestResult {
                fragment_id,
                token_count: tc,
                entropy_score: entropy,
                is_duplicate: true,
                duplicate_of: dup,
            };
            return serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL);
        }

        // Build fragment
        let mut frag = ContextFragment::new(fragment_id.clone(), content.clone(), tc, source);
        frag.entropy_score = entropy;
        frag.turn_created = self.current_turn;
        frag.turn_last_accessed = self.current_turn;
        frag.is_pinned = is_pinned;
        frag.simhash = simhash(&content);

        // Build dependency links
        let identifiers = extract_identifiers(&content);
        for ident in &identifiers {
            self.dep_graph.register_symbol(ident, &fragment_id);
        }
        self.dep_graph.auto_link(&fragment_id, &content);

        self.fragments.insert(fragment_id.clone(), frag);

        let result = IngestResult {
            fragment_id,
            token_count: tc,
            entropy_score: entropy,
            is_duplicate: false,
            duplicate_of: None,
        };
        serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL)
    }

    /// Optimize context selection for a given token budget and query.
    ///
    /// Returns a JSON object with selected fragments.
    #[wasm_bindgen]
    pub fn optimize(&mut self, token_budget: u32, query: String) -> JsValue {
        self.current_turn += 1;

        // Task classification for budget multiplier
        let task_type = TaskType::classify(&query);
        let effective_budget = (token_budget as f64 * task_type.budget_multiplier()) as u32;

        self.last_query = query.clone();
        self.last_effective_budget = effective_budget;

        // Apply Ebbinghaus decay
        let mut frag_vec: Vec<ContextFragment> = self.fragments.values().cloned().collect();
        apply_ebbinghaus_decay(&mut frag_vec, self.current_turn, self.decay_half_life);
        for f in &frag_vec {
            if let Some(stored) = self.fragments.get_mut(&f.fragment_id) {
                stored.recency_score = f.recency_score;
            }
        }

        // Cache lookup
        let current_frag_ids: HashSet<String> = self.fragments.keys().cloned().collect();
        let cache_result = self.egsc_cache.lookup_with_budget(&query, &current_frag_ids, effective_budget);
        if let CacheLookup::ExactHit { response, .. } | CacheLookup::SemanticHit { response, .. } = &cache_result {
            self.last_cache_hit = true;
            if let Ok(cached) = serde_json::from_str::<serde_json::Value>(response) {
                return serde_wasm_bindgen::to_value(&cached).unwrap_or(JsValue::NULL);
            }
        }
        self.last_cache_hit = false;

        // Build feedback multiplier map for knapsack
        let feedback_mults: HashMap<String, f64> = self.fragments.keys()
            .map(|id| (id.clone(), self.feedback.learned_value(id)))
            .collect();

        // Prepare fragment slice for knapsack
        let frag_list: Vec<ContextFragment> = self.fragments.values().cloned().collect();
        let weights = ScoringWeights {
            recency: self.w_recency,
            frequency: self.w_frequency,
            semantic: self.w_semantic,
            entropy: self.w_entropy,
        };

        // Run knapsack optimization (temperature = 0.1 for soft bisection)
        let knapsack_result = knapsack_optimize(
            &frag_list,
            effective_budget,
            &weights,
            &feedback_mults,
            0.1, // temperature
        );

        // Build selected indices set
        let selected_set: HashSet<usize> = knapsack_result.selected_indices.iter().copied().collect();

        // Build result
        let mut selected = Vec::new();
        let mut total_tokens = 0u32;

        for (i, frag) in frag_list.iter().enumerate() {
            let is_selected = selected_set.contains(&i);
            let fm = feedback_mults.get(&frag.fragment_id).copied().unwrap_or(1.0);
            let relevance = compute_relevance(
                frag,
                self.w_recency,
                self.w_frequency,
                self.w_semantic,
                self.w_entropy,
                fm,
            );

            if is_selected {
                total_tokens += frag.token_count;
            }

            selected.push(FragmentScore {
                fragment_id: frag.fragment_id.clone(),
                content: frag.content.clone(),
                token_count: frag.token_count,
                source: frag.source.clone(),
                relevance,
                entropy_score: frag.entropy_score,
                selected: is_selected,
            });
        }

        // Store in cache
        if !query.is_empty() {
            let selected_ids_set: HashSet<String> = selected.iter()
                .filter(|f| f.selected)
                .map(|f| f.fragment_id.clone())
                .collect();
            let fragment_entropies: Vec<(f64, u32)> = selected.iter()
                .filter(|f| f.selected)
                .map(|f| (f.entropy_score, f.token_count))
                .collect();
            let response_json = serde_json::json!({
                "selected_ids": selected_ids_set.iter().collect::<Vec<_>>(),
                "total_tokens": total_tokens,
            }).to_string();
            self.egsc_cache.store_with_budget(
                &query,
                current_frag_ids,
                &fragment_entropies,
                response_json,
                total_tokens,
                self.current_turn,
                effective_budget,
            );
        }

        let result = OptimizeResult {
            selected,
            total_tokens,
            token_budget: effective_budget,
            fragments_considered: frag_list.len(),
            cache_hit: false,
        };
        serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL)
    }

    /// Record successful outcome — updates RL feedback weights.
    #[wasm_bindgen]
    pub fn record_success(&mut self) {
        let selected_ids: Vec<String> = self.fragments.keys().cloned().collect();
        if !selected_ids.is_empty() {
            self.feedback.record_success(&selected_ids);
        }
        if self.last_cache_hit {
            let fids: HashSet<String> = self.fragments.keys().cloned().collect();
            self.egsc_cache.record_feedback_with_budget(
                &self.last_query, &fids, self.last_effective_budget, true,
            );
        }
    }

    /// Record failed outcome — updates RL feedback weights.
    #[wasm_bindgen]
    pub fn record_failure(&mut self) {
        let selected_ids: Vec<String> = self.fragments.keys().cloned().collect();
        if !selected_ids.is_empty() {
            self.feedback.record_failure(&selected_ids);
        }
        if self.last_cache_hit {
            let fids: HashSet<String> = self.fragments.keys().cloned().collect();
            self.egsc_cache.record_feedback_with_budget(
                &self.last_query, &fids, self.last_effective_budget, false,
            );
        }
    }

    /// Remove a fragment by ID.
    #[wasm_bindgen]
    pub fn remove(&mut self, fragment_id: &str) -> bool {
        self.dedup_index.remove(fragment_id);
        self.fragments.remove(fragment_id).is_some()
    }

    /// Get engine statistics as JSON.
    #[wasm_bindgen]
    pub fn stats(&mut self) -> JsValue {
        let total_tokens: u32 = self.fragments.values().map(|f| f.token_count).sum();
        let cache_stats = self.egsc_cache.stats();

        let result = EngineStats {
            total_fragments: self.fragments.len(),
            total_tokens,
            current_turn: self.current_turn,
            cache: CacheStatsView {
                total_entries: cache_stats.total_entries,
                hit_rate: cache_stats.hit_rate,
                total_tokens_saved: cache_stats.total_tokens_saved,
                exact_hits: cache_stats.exact_hits,
                semantic_hits: cache_stats.semantic_hits,
            },
        };
        serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL)
    }

    /// Get fragment count.
    #[wasm_bindgen]
    pub fn fragment_count(&self) -> usize {
        self.fragments.len()
    }

    /// Clear all fragments.
    #[wasm_bindgen]
    pub fn clear(&mut self) {
        self.fragments.clear();
        self.dedup_index = DedupIndex::new(self.hamming_threshold);
        self.dep_graph = DepGraph::new();
        self.egsc_cache.clear();
    }
}
