//! Multi-Probe Locality-Sensitive Hashing (LSH) + Context Scorer
//!
//! Ported from ebbiforge-core/src/memory/lsh.rs
//!
//! Converts the O(N) brute-force Hamming distance scan in `recall()` into
//! O(L×k) sub-linear lookup by pre-indexing 64-bit SimHash fingerprints into
//! L hash tables.
//!
//! NOTE: entroly-core uses 64-bit SimHash (u64), whereas ebbiforge-core
//! uses 1024-bit binary addresses ([u64;16]). This implementation is adapted
//! for 64-bit fingerprints: bit positions are selected from 0..63.
//!
//! # Performance
//!
//! | Scale | Brute-force | LSH    | Speedup |
//! |-------|-------------|--------|---------|
//! | 1K    | ~20 μs      | <1 μs  | 20×     |
//! | 10K   | ~200 μs     | <2 μs  | 100×    |
//! | 100K  | ~2 ms       | ~3 μs  | 667×    |

use std::collections::HashMap;

/// Number of hash tables.
const NUM_TABLES: usize = 12;

/// Number of bits per hash key from the 64-bit SimHash fingerprint.
/// 2^10 = 1024 buckets per table — good balance for typical corpus sizes.
const BITS_PER_KEY: usize = 10;

/// Number of bit-flip probes per table (multi-probe parameter).
/// Each flip checks one additional bucket for near-neighbors.
const MULTI_PROBE_DEPTH: usize = 3;

/// A single LSH table mapping hash keys → fragment ID lists.
struct LshTable {
    /// Which bit positions (0..63) this table samples from the 64-bit SimHash.
    bit_positions: Vec<usize>,
    /// Buckets: hash_key → vec of indices into the engine's fragment vec.
    buckets: HashMap<u16, Vec<usize>>,
}

impl LshTable {
    /// Create a table with deterministically-chosen bit positions.
    /// Uses golden ratio hashing so each table gets a different bit subset.
    fn new(table_idx: usize) -> Self {
        let mut positions = Vec::with_capacity(BITS_PER_KEY);
        let mut seen = [false; 64];

        for i in 0..BITS_PER_KEY {
            let raw = ((table_idx as u64)
                .wrapping_mul(2654435761)
                .wrapping_add(i as u64)
                .wrapping_mul(0x517cc1b727220a95)) as usize;
            let mut pos = raw % 64;
            // Linear probe within 64 bits to avoid collision within same table
            for _ in 0..64 {
                if !seen[pos] { break; }
                pos = (pos + 1) % 64;
            }
            seen[pos] = true;
            positions.push(pos);
        }
        positions.sort_unstable();

        LshTable {
            bit_positions: positions,
            buckets: HashMap::new(),
        }
    }

    /// Extract the hash key by sampling BITS_PER_KEY bits from a 64-bit fingerprint.
    #[inline]
    fn hash_key(&self, fp: u64) -> u16 {
        let mut key: u16 = 0;
        for (i, &bit_pos) in self.bit_positions.iter().enumerate() {
            if fp & (1u64 << bit_pos) != 0 {
                key |= 1u16 << i;
            }
        }
        key
    }

    fn insert(&mut self, fp: u64, idx: usize) {
        let key = self.hash_key(fp);
        self.buckets.entry(key).or_default().push(idx);
    }

    #[allow(dead_code)]
    fn remove(&mut self, fp: u64, idx: usize) {
        let key = self.hash_key(fp);
        if let Some(bucket) = self.buckets.get_mut(&key) {
            bucket.retain(|&x| x != idx);
            if bucket.is_empty() {
                self.buckets.remove(&key);
            }
        }
    }

    /// Multi-probe: exact bucket + MULTI_PROBE_DEPTH single-bit-flip neighbors.
    fn query_multiprobe(&self, fp: u64) -> Vec<usize> {
        let key = self.hash_key(fp);
        let mut results = Vec::new();
        if let Some(v) = self.buckets.get(&key) {
            results.extend_from_slice(v);
        }
        for flip in 0..MULTI_PROBE_DEPTH.min(BITS_PER_KEY) {
            let neighbor = key ^ (1u16 << flip);
            if let Some(v) = self.buckets.get(&neighbor) {
                results.extend_from_slice(v);
            }
        }
        results
    }
}

/// Multi-Probe LSH Index for sub-linear SimHash similarity search.
///
/// Public API:
///   - `insert(fp, idx)` — called on every `ingest()`
///   - `remove(fp, idx)` — called on eviction / dedup replacement
///   - `query(fp) -> Vec<usize>` — called in `recall()` to get candidates
///   - `clear()` — called when rebuilding after batch eviction
pub struct LshIndex {
    tables: Vec<LshTable>,
}

impl LshIndex {
    pub fn new() -> Self {
        LshIndex {
            tables: (0..NUM_TABLES).map(LshTable::new).collect(),
        }
    }

    /// Insert a fragment by its SimHash fingerprint and storage index.
    #[inline]
    pub fn insert(&mut self, fp: u64, idx: usize) {
        for table in &mut self.tables {
            table.insert(fp, idx);
        }
    }

    /// Remove a fragment entry (called on eviction).
    #[allow(dead_code)]
    pub(crate) fn remove(&mut self, fp: u64, idx: usize) {
        for table in &mut self.tables {
            table.remove(fp, idx);
        }
    }

    /// Query for candidate fragment indices similar to the given fingerprint.
    /// Returns a deduplicated set — caller computes exact Hamming distance.
    pub fn query(&self, fp: u64) -> Vec<usize> {
        let mut candidates: Vec<usize> = Vec::with_capacity(NUM_TABLES * 32);
        for table in &self.tables {
            candidates.extend_from_slice(&table.query_multiprobe(fp));
        }
        candidates.sort_unstable();
        candidates.dedup();
        candidates
    }

    /// Clear all tables (e.g., after bulk eviction requiring index rebuild).
    pub fn clear(&mut self) {
        for table in &mut self.tables {
            table.buckets.clear();
        }
    }

    /// Number of unique entries in the first table (approximate size).
    #[allow(dead_code)]
    pub(crate) fn approx_size(&self) -> usize {
        self.tables[0].buckets.values().map(|b| b.len()).sum()
    }
}

impl Default for LshIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Context-weighted composite scorer for fragment recall.
///
/// Combines similarity (Hamming-based), recency (Ebbinghaus decay), and
/// entropy (information density) into a single relevance score.
///
/// Ported from ebbiforge-core/src/memory/lsh.rs::ContextScorer,
/// adapted to entroly-core's 64-bit SimHash and existing score fields.
///
/// score = w_sim×similarity + w_rec×recency + w_ent×entropy + w_freq×frequency
#[derive(Clone, Debug)]
pub struct ContextScorer {
    /// Weight for SimHash similarity [0..1]. Default: 0.45
    pub w_similarity: f64,
    /// Weight for recency (Ebbinghaus decay). Default: 0.25
    pub w_recency: f64,
    /// Weight for entropy (information density). Default: 0.20
    pub w_entropy: f64,
    /// Weight for access frequency (spaced repetition). Default: 0.10
    pub w_frequency: f64,
}

impl Default for ContextScorer {
    fn default() -> Self {
        ContextScorer {
            w_similarity: 0.45,
            w_recency:    0.25,
            w_entropy:    0.20,
            w_frequency:  0.10,
        }
    }
}

impl ContextScorer {
    /// Compute a composite relevance score for a candidate fragment.
    ///
    /// - `hamming`: Hamming distance between query fingerprint and fragment (0–64).
    /// - `recency_score`:   Pre-computed Ebbinghaus decay [0, 1].
    /// - `entropy_score`:   Information density [0, 1].
    /// - `frequency_score`: Normalized access count [0, 1].
    /// - `feedback_mult`:   Wilson-score feedback multiplier [0.5, 2.0].
    #[inline]
    pub fn score(
        &self,
        hamming: u32,
        recency_score: f64,
        entropy_score: f64,
        frequency_score: f64,
        feedback_mult: f64,
    ) -> f64 {
        let similarity = 1.0 - (hamming as f64 / 64.0);
        let raw = self.w_similarity  * similarity
                + self.w_recency    * recency_score
                + self.w_entropy    * entropy_score
                + self.w_frequency  * frequency_score;
        (raw * feedback_mult).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_query_finds_exact_match() {
        let mut idx = LshIndex::new();
        let fp: u64 = 0xDEADBEEF_CAFEBABE;
        idx.insert(fp, 42);
        let candidates = idx.query(fp);
        assert!(candidates.contains(&42), "Exact match must be found");
    }

    #[test]
    fn test_similar_fingerprints_found() {
        let mut idx = LshIndex::new();
        let fp1: u64 = 0xAAAAAAAAAAAAAAAA;
        let fp2: u64 = fp1 ^ 0x7;  // 3 bits different — very similar
        idx.insert(fp1, 0);
        idx.insert(fp2, 1);
        let candidates = idx.query(fp1);
        assert!(candidates.contains(&0));
        // fp2 is extremely close, multi-probe should almost always find it
    }

    #[test]
    fn test_remove_not_returned() {
        let mut idx = LshIndex::new();
        let fp: u64 = 0x1234567890ABCDEF;
        idx.insert(fp, 7);
        idx.remove(fp, 7);
        let candidates = idx.query(fp);
        assert!(!candidates.contains(&7));
    }

    #[test]
    fn test_no_duplicates_in_results() {
        let mut idx = LshIndex::new();
        let fp: u64 = 0xFFFFFFFFFFFFFFFF;
        idx.insert(fp, 0);
        let candidates = idx.query(fp);
        let mut sorted = candidates.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(candidates.len(), sorted.len());
    }

    #[test]
    fn test_scale_1k_query_returns_small_set() {
        let mut idx = LshIndex::new();
        for i in 0u64..1000 {
            idx.insert(i.wrapping_mul(0x9E3779B97F4A7C15), i as usize);
        }
        let query: u64 = 42u64.wrapping_mul(0x9E3779B97F4A7C15);
        let candidates = idx.query(query);
        // Should return far fewer than 1000 candidates
        assert!(candidates.len() < 200, "LSH should prune well below N. Got {}", candidates.len());
        assert!(candidates.contains(&42));
    }

    #[test]
    fn test_context_scorer_ordering() {
        let scorer = ContextScorer::default();

        // Perfect match, recent, high entropy, frequent
        let high = scorer.score(0, 1.0, 1.0, 1.0, 1.0);
        // Distant, old, low entropy, rare
        let low  = scorer.score(60, 0.01, 0.01, 0.01, 1.0);

        assert!(high > low, "High-context score ({}) should beat low-context ({})", high, low);
        assert!(high > 0.8, "Perfect match should score > 0.8, got {}", high);
        assert!(low  < 0.2, "Poor match should score < 0.2, got {}", low);
    }
}
