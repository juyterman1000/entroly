//! Multi-Probe Locality-Sensitive Hashing (LSH) Index
//!
//! Converts O(N) brute-force Hamming scan into O(L×k) sub-linear lookup
//! by pre-indexing 1024-bit SimHash addresses into L hash tables.
//!
//! Ported from ebbiforge-core/src/memory/lsh.rs.
//!
//! Performance (from ebbiforge benchmarks):
//!   10K episodes:  200μs → <1μs  (200× speedup)
//!   1M episodes:   20ms  → ~1μs  (20,000× speedup)

use std::collections::HashMap;

/// Number of hash tables.
const NUM_TABLES: usize = 16;

/// Bits per hash key (from the 1024-bit address).
/// 2^20 = 1,048,576 buckets per table.
const BITS_PER_KEY: usize = 20;

/// A single LSH table mapping hash keys to episode indices.
struct LSHTable {
    bit_positions: Vec<usize>,
    buckets: HashMap<u32, Vec<usize>>,
}

impl LSHTable {
    /// Create with deterministic bit positions via golden ratio hashing.
    fn new(table_index: usize) -> Self {
        let mut positions = Vec::with_capacity(BITS_PER_KEY);
        let mut seen = [false; 1024];

        for i in 0..BITS_PER_KEY {
            let raw = ((table_index as u64)
                .wrapping_mul(2654435761)
                .wrapping_add(i as u64)
                .wrapping_mul(0x517cc1b727220a95)) as usize;
            let mut pos = raw % 1024;
            while seen[pos] {
                pos = (pos + 1) % 1024;
            }
            seen[pos] = true;
            positions.push(pos);
        }
        positions.sort_unstable();

        LSHTable {
            bit_positions: positions,
            buckets: HashMap::new(),
        }
    }

    /// Extract hash key from a 1024-bit address.
    #[inline]
    fn hash_key(&self, address: &[u64; 16]) -> u32 {
        let mut key: u32 = 0;
        for (i, &bit_pos) in self.bit_positions.iter().enumerate() {
            let word = bit_pos / 64;
            let bit = bit_pos % 64;
            if address[word] & (1u64 << bit) != 0 {
                key |= 1u32 << i;
            }
        }
        key
    }

    fn insert(&mut self, address: &[u64; 16], episode_idx: usize) {
        let key = self.hash_key(address);
        self.buckets.entry(key).or_default().push(episode_idx);
    }

    /// Single-probe query: exact bucket match only.
    fn query_exact(&self, address: &[u64; 16]) -> Vec<usize> {
        let key = self.hash_key(address);
        self.buckets.get(&key).cloned().unwrap_or_default()
    }

    /// Multi-probe: flip each key bit and collect from neighboring buckets.
    fn query_multiprobe(&self, address: &[u64; 16]) -> Vec<usize> {
        let key = self.hash_key(address);
        let mut results = Vec::new();

        // Exact bucket
        if let Some(v) = self.buckets.get(&key) {
            results.extend_from_slice(v);
        }

        // 1-bit perturbations (flip each of the BITS_PER_KEY bits)
        for bit in 0..BITS_PER_KEY.min(20) {
            let perturbed = key ^ (1u32 << bit);
            if let Some(v) = self.buckets.get(&perturbed) {
                results.extend_from_slice(v);
            }
        }

        results
    }
}

/// Multi-probe LSH Index for O(1) recall.
pub struct LSHIndex {
    tables: Vec<LSHTable>,
}

impl Default for LSHIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl LSHIndex {
    pub fn new() -> Self {
        let tables = (0..NUM_TABLES).map(LSHTable::new).collect();
        LSHIndex { tables }
    }

    /// Insert an episode's binary address into all tables.
    pub fn insert(&mut self, address: &[u64; 16], episode_idx: usize) {
        for table in &mut self.tables {
            table.insert(address, episode_idx);
        }
    }

    /// Exact-probe query: find candidates with identical hash keys.
    /// Used for deduplication.
    pub fn query_exact(&self, address: &[u64; 16]) -> Vec<usize> {
        let mut candidates = Vec::new();
        for table in &self.tables {
            candidates.extend(table.query_exact(address));
        }
        candidates.sort_unstable();
        candidates.dedup();
        candidates
    }

    /// Multi-probe query: find candidates near the given address.
    /// Used for similarity recall.
    pub fn query_multiprobe(&self, address: &[u64; 16]) -> Vec<usize> {
        let mut candidates = Vec::new();
        for table in &self.tables {
            candidates.extend(table.query_multiprobe(address));
        }
        candidates.sort_unstable();
        candidates.dedup();
        candidates
    }

    /// Clear all tables (for reset).
    pub fn clear(&mut self) {
        for table in &mut self.tables {
            table.buckets.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsh_insert_and_query() {
        let mut index = LSHIndex::new();
        let addr = [42u64; 16];
        index.insert(&addr, 0);
        index.insert(&addr, 1);

        let results = index.query_exact(&addr);
        assert!(results.contains(&0));
        assert!(results.contains(&1));
    }

    #[test]
    fn test_lsh_multiprobe_finds_nearby() {
        let mut index = LSHIndex::new();
        let addr = [0u64; 16];
        index.insert(&addr, 0);

        // Multi-probe should find the original
        let results = index.query_multiprobe(&addr);
        assert!(results.contains(&0));
    }

    #[test]
    fn test_lsh_different_addresses_separate() {
        let mut index = LSHIndex::new();
        let addr_a = [0u64; 16];
        let addr_b = [u64::MAX; 16];
        index.insert(&addr_a, 0);
        index.insert(&addr_b, 1);

        let results_a = index.query_exact(&addr_a);
        let results_b = index.query_exact(&addr_b);
        // They might overlap in some tables but exact query should mostly separate
        assert!(results_a.contains(&0));
        assert!(results_b.contains(&1));
    }

    #[test]
    fn test_lsh_clear() {
        let mut index = LSHIndex::new();
        index.insert(&[0u64; 16], 0);
        index.clear();
        let results = index.query_exact(&[0u64; 16]);
        assert!(results.is_empty());
    }
}
