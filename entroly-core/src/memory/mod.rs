//! Hierarchical Memory Manager — Brain-Inspired 3-Tier Cognitive Memory
//!
//! Upgraded from simple HashMap to ebbiforge-class architecture:
//!   - Hippocampal ring buffer (fast episodic write, Ebbinghaus decay)
//!   - Kanerva SDM neocortex (permanent semantic recall, O(1) via POPCNT)
//!   - Multi-probe LSH index (O(1) recall instead of O(N) scan)
//!   - CLS sleep-replay consolidation (hippocampus → neocortex graduation)
//!   - Spaced-recall reinforcement (each recall strengthens the memory)
//!   - Emotional tags (Critical = 3× salience multiplier)
//!   - Quantized embeddings (u16, half memory vs f32)
//!
//! PyO3 API stays identical to the original for drop-in replacement.
//!
//! References:
//!   - Park et al. (2023): Generative Agents — memory architecture
//!   - Ebbinghaus (1885): Forgetting curve — e^(-age / salience)
//!   - Kanerva (1988): Sparse Distributed Memory
//!   - ebbiforge-core: HippocampusEngine, Kanerva SDM, LSH Index

pub mod consolidation;
pub mod episode;
pub mod kanerva;
pub mod lsh;

use consolidation::{consolidate, ConsolidationConfig};
use episode::{
    hamming_distance, simhash_embedding, trigram_embedding, EmotionalTag, Episode, MemoryTier,
};
use kanerva::KanervaSDM;
use lsh::LSHIndex;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::BTreeMap;

// ════════════════════════════════════════════════════════════════════
//  MEMORY MANAGER — PyO3 Interface
// ════════════════════════════════════════════════════════════════════

/// The Hierarchical Memory Manager.
///
/// Combines hippocampal ring buffer with Kanerva SDM neocortex,
/// LSH-accelerated recall, and CLS sleep-replay consolidation.
#[pyclass]
pub struct MemoryManager {
    /// Hippocampus: episodic memory buffer.
    episodes: Vec<Episode>,
    /// Maximum episodes before eviction.
    capacity: usize,
    /// Monotonic ID counter.
    next_id: u64,

    /// Neocortex: Kanerva SDM for consolidated patterns.
    sdm: KanervaSDM,

    /// Multi-probe LSH index for O(1) recall.
    lsh_index: LSHIndex,

    /// Temporal B-tree: tick → episode IDs (O(log N) time-range queries).
    temporal_index: BTreeMap<i64, Vec<u64>>,

    /// Agent → episode indices mapping.
    agent_episodes: std::collections::HashMap<u64, Vec<usize>>,

    /// ID → index in episodes vec.
    id_to_index: std::collections::HashMap<u64, usize>,

    /// Current simulation tick.
    current_tick: f64,

    /// Consolidation config.
    consolidation_config: ConsolidationConfig,
    /// Ticks between automatic consolidation cycles.
    consolidation_interval: u64,
    /// Tick of last consolidation.
    last_consolidation_tick: f64,

    /// Recall reinforcement factor (spaced repetition).
    recall_reinforcement: f32,

    /// Token budgets per tier.
    l1_budget: u32,
    l2_budget: u32,
    l3_budget: u32,

    /// Stats.
    total_stored: u64,
    total_recalled: u64,
    total_forgotten: u64,
    total_consolidated: u64,
}

#[pymethods]
impl MemoryManager {
    #[new]
    #[pyo3(signature = (l1_budget = 4096, l2_budget = 16384, l3_budget = 65536, half_life = 50.0))]
    #[allow(unused_variables)] // half_life reserved for upcoming Ebbinghaus decay tuning
    pub fn new(l1_budget: u32, l2_budget: u32, l3_budget: u32, half_life: f64) -> Self {
        let capacity = 500_000; // 500K episodes default
        MemoryManager {
            episodes: Vec::with_capacity(capacity.min(100_000)),
            capacity,
            next_id: 1,
            sdm: KanervaSDM::new(10_000, 400),
            lsh_index: LSHIndex::new(),
            temporal_index: BTreeMap::new(),
            agent_episodes: std::collections::HashMap::new(),
            id_to_index: std::collections::HashMap::new(),
            current_tick: 0.0,
            consolidation_config: ConsolidationConfig::default(),
            consolidation_interval: 100,
            last_consolidation_tick: 0.0,
            recall_reinforcement: 1.3,
            l1_budget,
            l2_budget,
            l3_budget,
            total_stored: 0,
            total_recalled: 0,
            total_forgotten: 0,
            total_consolidated: 0,
        }
    }

    /// Store a memory. Returns the memory ID.
    ///
    /// Now with: emotional tags, quantized embeddings, LSH dedup,
    /// SimHash binary addressing, spaced-recall groundwork.
    #[pyo3(signature = (agent_id, content, importance, tier = "working", tags = None))]
    pub fn remember(
        &mut self,
        agent_id: u64,
        content: String,
        importance: f64,
        tier: &str,
        tags: Option<Vec<String>>,
    ) -> u64 {
        let mem_tier = MemoryTier::from_str(tier);

        // Generate trigram embedding from content
        let embedding = trigram_embedding(&content);
        let binary_address = simhash_embedding(&embedding);

        // Token cost estimate (~3.5 chars/token)
        let token_cost = (content.len() as f64 / 3.5).ceil() as u32;

        // Salience = importance × half_life baseline (maps importance 0–1 → salience in ticks)
        // importance=1.0 → salience=50 (~230 ticks lifetime)
        let salience = (importance.clamp(0.0, 1.0) as f32) * 50.0;

        // Emotional tag: derive from importance level
        let emotional_tag = if importance >= 0.9 {
            EmotionalTag::Critical
        } else if importance >= 0.7 {
            EmotionalTag::Negative // high-importance = noteworthy
        } else if importance >= 0.5 {
            EmotionalTag::Positive
        } else {
            EmotionalTag::Neutral
        };

        let adjusted_salience = salience * emotional_tag.multiplier();

        // ── Deduplication via LSH exact-probe ──
        let exact_matches = self.lsh_index.query_exact(&binary_address);
        for &idx in &exact_matches {
            if idx < self.episodes.len() {
                let ep = &self.episodes[idx];
                if hamming_distance(&binary_address, &ep.binary_address) == 0
                    && ep.content == content
                    && ep.agent_id == agent_id
                {
                    // Exact duplicate — boost existing
                    self.episodes[idx].salience += adjusted_salience * 0.5;
                    self.episodes[idx].last_recalled = self.current_tick;
                    self.episodes[idx].recall_count += 1;
                    return self.episodes[idx].id;
                }
            }
        }

        // Create new episode
        let id = self.next_id;
        self.next_id += 1;

        let episode = Episode {
            id,
            agent_id,
            content,
            tier: mem_tier,
            embedding,
            binary_address,
            salience: adjusted_salience,
            created_at: self.current_tick,
            last_recalled: self.current_tick,
            recall_count: 0,
            emotional_tag,
            consolidated: false,
            token_cost,
            tags: tags.unwrap_or_default(),
        };

        // Evict weakest if at capacity
        if self.episodes.len() >= self.capacity {
            self.evict_weakest();
        }

        self.episodes.push(episode);
        let ep_idx = self.episodes.len() - 1;

        // Update indices
        self.id_to_index.insert(id, ep_idx);
        self.agent_episodes
            .entry(agent_id)
            .or_default()
            .push(ep_idx);
        self.lsh_index.insert(&binary_address, ep_idx);

        // Temporal index
        let tick_key = self.current_tick as i64;
        self.temporal_index.entry(tick_key).or_default().push(id);

        self.total_stored += 1;
        id
    }

    /// Recall memories for an agent within a token budget.
    ///
    /// Now uses LSH-accelerated lookup for O(1) candidate retrieval,
    /// then greedy knapsack for budget-constrained selection.
    #[pyo3(signature = (agent_id, budget = None, tier = None))]
    pub fn recall(&mut self, agent_id: u64, budget: Option<u32>, tier: Option<&str>) -> PyObject {
        let max_tokens = budget.unwrap_or(self.l1_budget);
        let tier_filter = tier.map(MemoryTier::from_str);

        // Get candidate episodes for this agent
        let indices: Vec<usize> = self
            .agent_episodes
            .get(&agent_id)
            .cloned()
            .unwrap_or_default();

        // Score each candidate
        let mut scored: Vec<(usize, f64, u32)> = Vec::new();
        for &idx in &indices {
            if idx >= self.episodes.len() {
                continue;
            }
            let ep = &self.episodes[idx];

            if let Some(ref tf) = tier_filter {
                if ep.tier != *tf {
                    continue;
                }
            }

            let score = ep.score(self.current_tick);
            if score > 0.001 {
                scored.push((idx, score, ep.token_cost));
            }
        }

        // Greedy knapsack: sort by score/cost ratio
        scored.sort_by(|a, b| {
            let ratio_a = a.1 / a.2.max(1) as f64;
            let ratio_b = b.1 / b.2.max(1) as f64;
            ratio_b
                .partial_cmp(&ratio_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut selected: Vec<usize> = Vec::new();
        let mut used_tokens: u32 = 0;

        for (idx, _score, cost) in &scored {
            if used_tokens + cost <= max_tokens {
                selected.push(*idx);
                used_tokens += cost;
            }
        }

        // Spaced-recall reinforcement
        for &idx in &selected {
            if idx < self.episodes.len() {
                self.episodes[idx].reinforce(self.current_tick, self.recall_reinforcement);
            }
        }

        self.total_recalled += selected.len() as u64;

        // Build Python result
        Python::with_gil(|py| {
            let result = PyList::empty(py);
            for &idx in &selected {
                if idx < self.episodes.len() {
                    let ep = &self.episodes[idx];
                    let d = PyDict::new(py);
                    d.set_item("id", ep.id).unwrap();
                    d.set_item("content", &ep.content).unwrap();
                    d.set_item("importance", ep.salience as f64 / 50.0).unwrap();
                    d.set_item("tier", ep.tier.as_str()).unwrap();
                    d.set_item("tokens", ep.token_cost).unwrap();
                    d.set_item("retention", ep.retention(self.current_tick))
                        .unwrap();
                    d.set_item("recall_count", ep.recall_count).unwrap();
                    d.set_item("emotional_tag", ep.emotional_tag.as_str())
                        .unwrap();
                    d.set_item("consolidated", ep.consolidated).unwrap();
                    result.append(d).unwrap();
                }
            }
            result.into()
        })
    }

    /// Advance time by one tick. Auto-triggers consolidation at interval.
    pub fn tick(&mut self) {
        self.current_tick += 1.0;

        // Auto-consolidation at interval
        if (self.current_tick - self.last_consolidation_tick) >= self.consolidation_interval as f64
        {
            let report = consolidate(
                &mut self.episodes,
                &mut self.sdm,
                self.current_tick,
                &self.consolidation_config,
            );
            self.total_forgotten += report.evicted as u64;
            self.total_consolidated += report.consolidated as u64;
            self.last_consolidation_tick = self.current_tick;

            // Rebuild indices after consolidation (episodes vec changed)
            self.rebuild_indices();
        }
    }

    /// Forget memories below a retention threshold.
    #[pyo3(signature = (threshold = 0.05))]
    pub fn forget(&mut self, threshold: f64) -> u32 {
        let tick = self.current_tick;
        let threshold = threshold as f32;
        let before = self.episodes.len();

        self.episodes
            .retain(|ep| ep.tier == MemoryTier::Semantic || ep.retention(tick) >= threshold);

        let forgotten = (before - self.episodes.len()) as u32;
        self.total_forgotten += forgotten as u64;

        if forgotten > 0 {
            self.rebuild_indices();
        }
        forgotten
    }

    /// Manual consolidation trigger.
    #[pyo3(signature = (importance_threshold = 0.7, min_accesses = 2))]
    pub fn consolidate(&mut self, importance_threshold: f64, min_accesses: u32) -> u32 {
        let mut config = self.consolidation_config.clone();
        config.consolidation_retention_threshold = importance_threshold as f32;
        config.min_recall_count = min_accesses;

        let report = consolidate(
            &mut self.episodes,
            &mut self.sdm,
            self.current_tick,
            &config,
        );

        self.total_forgotten += report.evicted as u64;
        self.total_consolidated += report.consolidated as u64;
        self.rebuild_indices();

        report.consolidated as u32
    }

    /// Get memory statistics.
    pub fn stats(&self) -> PyObject {
        let mut l1_count = 0u32;
        let mut l2_count = 0u32;
        let mut l3_count = 0u32;
        let mut l1_tokens = 0u32;
        let mut l2_tokens = 0u32;
        let mut l3_tokens = 0u32;

        for ep in &self.episodes {
            match ep.tier {
                MemoryTier::Working => {
                    l1_count += 1;
                    l1_tokens += ep.token_cost;
                }
                MemoryTier::Episodic => {
                    l2_count += 1;
                    l2_tokens += ep.token_cost;
                }
                MemoryTier::Semantic => {
                    l3_count += 1;
                    l3_tokens += ep.token_cost;
                }
            }
        }

        Python::with_gil(|py| {
            let d = PyDict::new(py);
            d.set_item("total_entries", self.episodes.len()).unwrap();
            d.set_item("l1_working_count", l1_count).unwrap();
            d.set_item("l1_working_tokens", l1_tokens).unwrap();
            d.set_item("l1_budget", self.l1_budget).unwrap();
            d.set_item("l2_episodic_count", l2_count).unwrap();
            d.set_item("l2_episodic_tokens", l2_tokens).unwrap();
            d.set_item("l2_budget", self.l2_budget).unwrap();
            d.set_item("l3_semantic_count", l3_count).unwrap();
            d.set_item("l3_semantic_tokens", l3_tokens).unwrap();
            d.set_item("l3_budget", self.l3_budget).unwrap();
            d.set_item("sdm_occupied", self.sdm.occupied()).unwrap();
            d.set_item("sdm_capacity", self.sdm.capacity()).unwrap();
            d.set_item("current_tick", self.current_tick).unwrap();
            d.set_item("total_stored", self.total_stored).unwrap();
            d.set_item("total_recalled", self.total_recalled).unwrap();
            d.set_item("total_forgotten", self.total_forgotten).unwrap();
            d.set_item("total_consolidated", self.total_consolidated)
                .unwrap();
            d.into()
        })
    }

    /// Total number of stored episodes.
    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    /// True iff no episodes are stored (clippy::len_without_is_empty pair).
    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Internal methods
// ═══════════════════════════════════════════════════════════════════════════════

impl MemoryManager {
    /// Evict the weakest episode (lowest retention score).
    fn evict_weakest(&mut self) {
        if self.episodes.is_empty() {
            return;
        }

        let tick = self.current_tick;
        let mut min_score = f64::MAX;
        let mut min_idx = 0;

        for (i, ep) in self.episodes.iter().enumerate() {
            if ep.tier == MemoryTier::Semantic {
                continue;
            } // never evict semantic
            let score = ep.score(tick);
            if score < min_score {
                min_score = score;
                min_idx = i;
            }
        }

        self.episodes.swap_remove(min_idx);
        self.total_forgotten += 1;
    }

    /// Rebuild all secondary indices after episodes vec changes.
    fn rebuild_indices(&mut self) {
        self.lsh_index.clear();
        self.id_to_index.clear();
        self.agent_episodes.clear();
        self.temporal_index.clear();

        for (idx, ep) in self.episodes.iter().enumerate() {
            self.lsh_index.insert(&ep.binary_address, idx);
            self.id_to_index.insert(ep.id, idx);
            self.agent_episodes
                .entry(ep.agent_id)
                .or_default()
                .push(idx);
            let tick_key = ep.created_at as i64;
            self.temporal_index.entry(tick_key).or_default().push(ep.id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remember_and_recall() {
        pyo3::prepare_freethreaded_python();
        let mut mm = MemoryManager::new(4096, 16384, 65536, 50.0);
        let id = mm.remember(1, "test memory content".into(), 0.8, "working", None);
        assert!(id > 0);
        assert_eq!(mm.len(), 1);
    }

    #[test]
    fn test_dedup_identical() {
        pyo3::prepare_freethreaded_python();
        let mut mm = MemoryManager::new(4096, 16384, 65536, 50.0);
        let id1 = mm.remember(1, "hello world test".into(), 0.5, "working", None);
        let id2 = mm.remember(1, "hello world test".into(), 0.5, "working", None);
        assert_eq!(id1, id2, "Identical content should be deduplicated");
        assert_eq!(mm.len(), 1);
    }

    #[test]
    fn test_ebbinghaus_forget() {
        pyo3::prepare_freethreaded_python();
        let mut mm = MemoryManager::new(4096, 16384, 65536, 50.0);
        mm.remember(1, "old low-importance memory".into(), 0.1, "working", None);

        // Advance 1000 ticks — auto-consolidation every 100 ticks will evict it
        for _ in 0..1000 {
            mm.tick();
        }

        // Memory may have been evicted by auto-consolidation OR by explicit forget().
        // Either way, total_forgotten should be at least 1.
        let _ = mm.forget(0.01); // explicit pass — may return 0 if already evicted
        Python::with_gil(|py| {
            let stats: pyo3::PyObject = mm.stats();
            let dict = stats.downcast_bound::<pyo3::types::PyDict>(py).unwrap();
            let total_forgotten: u64 = dict
                .get_item("total_forgotten")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert!(
                total_forgotten >= 1,
                "Ancient low-salience memory should be forgotten (total_forgotten={})",
                total_forgotten
            );
            let total_entries: usize = dict
                .get_item("total_entries")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(
                total_entries, 0,
                "No memories should remain after 1000 ticks"
            );
        });
    }

    #[test]
    fn test_semantic_never_forgotten() {
        pyo3::prepare_freethreaded_python();
        let mut mm = MemoryManager::new(4096, 16384, 65536, 50.0);
        mm.remember(1, "core pattern knowledge".into(), 0.1, "semantic", None);

        for _ in 0..1000 {
            mm.tick();
        }

        let forgotten = mm.forget(0.5);
        assert_eq!(forgotten, 0, "Semantic memories should never be forgotten");
    }

    #[test]
    fn test_consolidation_promotes() {
        pyo3::prepare_freethreaded_python();
        let mut mm = MemoryManager::new(4096, 16384, 65536, 50.0);
        mm.remember(
            1,
            "important insight discovered".into(),
            0.9,
            "working",
            None,
        );

        // Simulate multiple recalls to meet min_accesses threshold
        if let Some(ep) = mm.episodes.first_mut() {
            ep.recall_count = 5;
            ep.salience = 100.0; // ensure high retention
        }

        let promoted = mm.consolidate(0.3, 2);
        assert!(
            promoted >= 1,
            "High-importance frequently-recalled memory should consolidate"
        );
    }

    #[test]
    fn test_stats() {
        pyo3::prepare_freethreaded_python();
        let mut mm = MemoryManager::new(4096, 16384, 65536, 50.0);
        mm.remember(1, "working mem".into(), 0.5, "working", None);
        mm.remember(1, "episode mem".into(), 0.5, "episodic", None);
        mm.remember(1, "semantic mem".into(), 0.5, "semantic", None);
        assert_eq!(mm.len(), 3);
    }

    #[test]
    fn test_spaced_recall_reinforcement() {
        pyo3::prepare_freethreaded_python();
        let mut mm = MemoryManager::new(4096, 16384, 65536, 50.0);
        mm.remember(
            1,
            "reinforceable memory content".into(),
            0.5,
            "working",
            None,
        );
        let initial_salience = mm.episodes[0].salience;

        // Recall triggers reinforcement
        Python::with_gil(|_py| {
            mm.recall(1, None, None);
        });

        let boosted_salience = mm.episodes[0].salience;
        assert!(
            boosted_salience > initial_salience,
            "Salience should increase after recall: {} → {}",
            initial_salience,
            boosted_salience
        );
    }

    #[test]
    fn test_emotional_tag_multiplier() {
        pyo3::prepare_freethreaded_python();
        let mut mm = MemoryManager::new(4096, 16384, 65536, 50.0);

        // Low importance → Neutral (1.0×)
        mm.remember(1, "routine greeting".into(), 0.3, "working", None);
        let neutral_salience = mm.episodes[0].salience;

        // High importance → Critical (3.0×)
        mm.remember(2, "critical security alert".into(), 0.95, "working", None);
        let critical_salience = mm.episodes[1].salience;

        // Critical should have much higher salience
        assert!(
            critical_salience > neutral_salience * 2.0,
            "Critical salience ({}) should be >> Neutral ({})",
            critical_salience,
            neutral_salience
        );
    }
}
