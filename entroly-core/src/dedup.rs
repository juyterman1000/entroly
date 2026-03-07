/// SimHash Deduplication — Rust implementation.
///
/// O(1) near-duplicate detection using:
///   1. 64-bit SimHash fingerprinting (word trigram features)
///   2. LSH banding (4 bands × 16 bits) for candidate retrieval
///   3. Hamming distance verification
///
/// Architecture mirrors ebbiforge-core/src/memory/lsh.rs but
/// optimized for context fragments instead of memory episodes.
/// Uses the same multi-table bucketing approach.
///
/// References:
///   - Charikar (2002) — SimHash
///   - Proximity (arXiv 2026) — LSH-bucketed semantic caching
///   - ebbiforge-core LSHIndex — 16-table multi-probe design

use std::collections::{HashMap, HashSet};
use md5::{Md5, Digest};
use serde::{Deserialize, Serialize};

/// Hash a token to a 64-bit integer using MD5.
#[inline]
fn hash_token(token: &str) -> u64 {
    let mut hasher = Md5::new();
    hasher.update(token.as_bytes());
    let digest = hasher.finalize();
    u64::from_le_bytes(digest[..8].try_into().unwrap())
}

/// Compute the 64-bit SimHash fingerprint of a text.
///
/// Algorithm:
///   1. Extract word trigrams as features
///   2. Hash each feature → 64-bit
///   3. For each bit: feature hash bit=1 → +1, bit=0 → -1
///   4. Final bit = 1 if sum > 0, else 0
pub fn simhash(text: &str) -> u64 {
    if text.is_empty() {
        return 0;
    }

    let words: Vec<&str> = text.split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '_'))
        .filter(|w| !w.is_empty())
        .collect();

    if words.is_empty() {
        return 0;
    }

    let mut bit_sums = [0i32; 64];

    // Use trigrams if enough words, else unigrams
    if words.len() >= 3 {
        for window in words.windows(3) {
            let feature = format!("{} {} {}", window[0], window[1], window[2]);
            let h = hash_token(&feature.to_lowercase());
            for i in 0..64 {
                if h & (1u64 << i) != 0 {
                    bit_sums[i] += 1;
                } else {
                    bit_sums[i] -= 1;
                }
            }
        }
    } else {
        for word in &words {
            let h = hash_token(&word.to_lowercase());
            for i in 0..64 {
                if h & (1u64 << i) != 0 {
                    bit_sums[i] += 1;
                } else {
                    bit_sums[i] -= 1;
                }
            }
        }
    }

    let mut fingerprint: u64 = 0;
    for i in 0..64 {
        if bit_sums[i] > 0 {
            fingerprint |= 1u64 << i;
        }
    }

    fingerprint
}

/// Hamming distance between two 64-bit fingerprints.
#[inline]
pub fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// LSH-bucketed deduplication index.
///
/// Split 64-bit fingerprints into 4 bands of 16 bits.
/// Two fingerprints sharing any band → candidate pair.
/// Verify with full Hamming distance.
///
/// Same approach as ebbiforge-core LSHIndex but with 4 tables
/// instead of 16 (sufficient for 64-bit fingerprints).
#[derive(Serialize, Deserialize)]
pub struct DedupIndex {
    hamming_threshold: u32,
    num_bands: usize,
    bits_per_band: usize,

    /// band_index → {band_hash → [fragment_ids]}
    buckets: Vec<HashMap<u64, Vec<String>>>,

    /// fragment_id → fingerprint
    fingerprints: HashMap<String, u64>,

    pub duplicates_detected: u64,
}

impl DedupIndex {
    pub fn new(hamming_threshold: u32) -> Self {
        let num_bands = 4;
        DedupIndex {
            hamming_threshold,
            num_bands,
            bits_per_band: 64 / num_bands,
            buckets: (0..num_bands).map(|_| HashMap::new()).collect(),
            fingerprints: HashMap::new(),
            duplicates_detected: 0,
        }
    }

    /// Extract band hashes from a fingerprint.
    fn extract_bands(&self, fp: u64) -> Vec<u64> {
        let mut bands = Vec::with_capacity(self.num_bands);
        for b in 0..self.num_bands {
            let shift = b * self.bits_per_band;
            let mask = (1u64 << self.bits_per_band) - 1;
            bands.push((fp >> shift) & mask);
        }
        bands
    }

    /// Insert a fragment. Returns Some(duplicate_id) if near-dup found.
    pub fn insert(&mut self, fragment_id: &str, text: &str) -> Option<String> {
        let fp = simhash(text);

        // Check for candidates via band matching
        let bands = self.extract_bands(fp);
        let mut candidates: HashSet<String> = HashSet::new();

        for (b, &band_hash) in bands.iter().enumerate() {
            if let Some(ids) = self.buckets[b].get(&band_hash) {
                for id in ids {
                    if id != fragment_id {
                        candidates.insert(id.clone());
                    }
                }
            }
        }

        // Verify with Hamming distance
        for cid in &candidates {
            if let Some(&existing_fp) = self.fingerprints.get(cid) {
                if hamming_distance(fp, existing_fp) <= self.hamming_threshold {
                    self.duplicates_detected += 1;
                    return Some(cid.clone());
                }
            }
        }

        // No duplicate — insert
        self.fingerprints.insert(fragment_id.to_string(), fp);
        for (b, &band_hash) in bands.iter().enumerate() {
            self.buckets[b]
                .entry(band_hash)
                .or_insert_with(Vec::new)
                .push(fragment_id.to_string());
        }

        None
    }

    /// Remove a fragment from the index.
    pub fn remove(&mut self, fragment_id: &str) {
        if let Some(fp) = self.fingerprints.remove(fragment_id) {
            let bands = self.extract_bands(fp);
            for (b, &band_hash) in bands.iter().enumerate() {
                if let Some(ids) = self.buckets[b].get_mut(&band_hash) {
                    ids.retain(|id| id != fragment_id);
                    if ids.is_empty() {
                        self.buckets[b].remove(&band_hash);
                    }
                }
            }
        }
    }

    pub fn size(&self) -> usize {
        self.fingerprints.len()
    }

    /// Get the stored fingerprint for a fragment.
    pub fn get_fingerprint(&self, fragment_id: &str) -> Option<u64> {
        self.fingerprints.get(fragment_id).copied()
    }

    /// Export all fingerprints for checkpointing.
    pub fn export_fingerprints(&self) -> Vec<(String, u64)> {
        self.fingerprints.iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_texts() {
        let text = "def process_payment(amount, currency):";
        assert_eq!(simhash(text), simhash(text));
    }

    #[test]
    fn test_dedup_catches_exact() {
        let mut idx = DedupIndex::new(3);
        let text = "def calculate_tax(income, rate): return income * rate";

        assert!(idx.insert("f1", text).is_none());
        assert_eq!(idx.insert("f2", text), Some("f1".to_string()));
    }

    #[test]
    fn test_dedup_allows_different() {
        let mut idx = DedupIndex::new(3);
        idx.insert("a", "machine learning neural network gradient descent");
        assert!(idx.insert("b", "kubernetes docker container orchestration").is_none());
    }
}
