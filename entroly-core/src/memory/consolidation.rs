//! Sleep-Replay Consolidation — CLS (Complementary Learning Systems)
//!
//! Replays hippocampal episodes, identifies which earned permanent status,
//! and migrates them to the neocortex (Kanerva SDM).
//!
//! Ported from ebbiforge-core/src/memory/consolidation.rs.

use super::episode::{Episode, MemoryTier};
use super::kanerva::KanervaSDM;

/// Thresholds governing the consolidation process.
#[derive(Clone, Debug)]
pub struct ConsolidationConfig {
    /// Minimum retention below which episode is evicted.
    pub death_threshold: f32,
    /// Minimum retention AND recall count to consolidate to neocortex.
    pub consolidation_retention_threshold: f32,
    /// Minimum recall count for consolidation eligibility.
    pub min_recall_count: u32,
    /// Salience reduction factor after consolidation.
    /// Hippocampal copy is weakened since neocortex owns it.
    pub post_consolidation_salience_factor: f32,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        ConsolidationConfig {
            death_threshold: 0.01,
            consolidation_retention_threshold: 0.5,
            min_recall_count: 2,
            post_consolidation_salience_factor: 0.5,
        }
    }
}

/// Report from a single consolidation cycle.
#[derive(Clone, Debug, Default)]
pub struct ConsolidationReport {
    pub evicted: usize,
    pub consolidated: usize,
    pub surviving: usize,
    pub total_processed: usize,
}

/// Run one CLS consolidation cycle (sleep replay).
///
/// 1. Scan all hippocampal episodes
/// 2. Evict those whose retention < death_threshold
/// 3. Consolidate high-retention frequently-recalled episodes to Kanerva SDM
/// 4. Semantic-tier episodes are NEVER evicted (but may be consolidated)
pub fn consolidate(
    episodes: &mut Vec<Episode>,
    sdm: &mut KanervaSDM,
    current_tick: f64,
    config: &ConsolidationConfig,
) -> ConsolidationReport {
    let total = episodes.len();
    let mut report = ConsolidationReport {
        total_processed: total,
        ..Default::default()
    };

    let mut surviving = Vec::with_capacity(total);

    for mut ep in episodes.drain(..) {
        let retention = ep.retention(current_tick);

        // Semantic memories are NEVER evicted (matches original agentOS behavior)
        if retention < config.death_threshold && ep.tier != MemoryTier::Semantic {
            report.evicted += 1;
            continue;
        }

        // Consolidation: high retention + frequently recalled + not yet consolidated
        if retention >= config.consolidation_retention_threshold
            && ep.recall_count >= config.min_recall_count
            && !ep.consolidated
        {
            // Write to Kanerva SDM neocortex
            let content: Vec<f32> = ep.embedding.iter()
                .map(|&v| (v as f32 / 32768.0) - 1.0)
                .collect();
            sdm.write(&ep.binary_address, &content);

            ep.consolidated = true;
            ep.salience *= config.post_consolidation_salience_factor;

            // Promote tier: Working → Episodic, Episodic → Semantic
            match ep.tier {
                MemoryTier::Working  => ep.tier = MemoryTier::Episodic,
                MemoryTier::Episodic => ep.tier = MemoryTier::Semantic,
                MemoryTier::Semantic => {},
            }

            report.consolidated += 1;
        }

        surviving.push(ep);
        report.surviving += 1;
    }

    *episodes = surviving;
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::episode::{EmotionalTag, simhash_embedding};

    fn make_episode(id: u64, salience: f32, created_at: f64,
                    recall_count: u32, tier: MemoryTier) -> Episode {
        let embedding = vec![32768u16; 64];
        let binary_address = simhash_embedding(&embedding);
        Episode {
            id, agent_id: 0, content: format!("memory_{}", id),
            tier, embedding, binary_address, salience,
            created_at, last_recalled: created_at, recall_count,
            emotional_tag: EmotionalTag::Neutral, consolidated: false,
            token_cost: 10, tags: vec![],
        }
    }

    #[test]
    fn test_eviction() {
        let mut sdm = KanervaSDM::new(100, 512);
        let config = ConsolidationConfig::default();
        let mut episodes = vec![make_episode(1, 0.1, 0.0, 0, MemoryTier::Working)];
        let report = consolidate(&mut episodes, &mut sdm, 100.0, &config);
        assert_eq!(report.evicted, 1, "Low-salience old memory should be evicted");
        assert!(episodes.is_empty());
    }

    #[test]
    fn test_semantic_never_evicted() {
        let mut sdm = KanervaSDM::new(100, 512);
        let config = ConsolidationConfig::default();
        let mut episodes = vec![make_episode(1, 0.1, 0.0, 0, MemoryTier::Semantic)];
        let report = consolidate(&mut episodes, &mut sdm, 100.0, &config);
        assert_eq!(report.evicted, 0, "Semantic memories should never be evicted");
        assert_eq!(report.surviving, 1);
    }

    #[test]
    fn test_consolidation_promotes() {
        let mut sdm = KanervaSDM::new(100, 512);
        let config = ConsolidationConfig::default();
        let mut episodes = vec![make_episode(1, 100.0, 0.0, 3, MemoryTier::Working)];
        let report = consolidate(&mut episodes, &mut sdm, 1.0, &config);
        assert_eq!(report.consolidated, 1);
        assert!(episodes[0].consolidated);
        assert_eq!(episodes[0].tier, MemoryTier::Episodic); // promoted
    }

    #[test]
    fn test_post_consolidation_salience_reduction() {
        let mut sdm = KanervaSDM::new(100, 512);
        let config = ConsolidationConfig::default();
        let mut episodes = vec![make_episode(1, 100.0, 0.0, 3, MemoryTier::Working)];
        consolidate(&mut episodes, &mut sdm, 1.0, &config);
        assert!((episodes[0].salience - 50.0).abs() < 0.1, "Salience should be halved");
    }
}
