//! # Shared Memory — cross-agent content-addressed store with deduplication
//!
//! Provides Rust-accelerated primitives for the cross-agent shared memory
//! system. Python orchestrates the store lifecycle; Rust handles the
//! compute-heavy parts:
//!
//! - **Content hashing**: truncated MD5 for fast content addressing
//! - **SimHash dedup**: batch near-duplicate detection across entries
//! - **BM25 search**: ranked retrieval over the memory corpus
//!
//! Designed for concurrent access by multiple agent processes. The store is
//! append-only with periodic compaction handled by Python.

use crate::bm25::{tokenize_code, BM25Index};
use crate::dedup::{hamming_distance, simhash};
use sha2::{Digest, Sha256};

/// Content hash for deduplication — SHA-256 truncated to 16 hex chars.
pub fn content_hash(text: &str) -> String {
    let hash = Sha256::digest(text.as_bytes());
    format!("{:x}", hash)[..16].to_string()
}

/// Batch deduplication: given a list of (id, content) pairs, returns which
/// entries are near-duplicates of earlier entries in the batch.
///
/// Returns Vec<(entry_id, is_duplicate, duplicate_of)> where `duplicate_of`
/// is the entry_id of the first matching entry, or empty if original.
pub fn batch_dedup(
    entries: &[(String, String)],
    threshold: u32,
) -> Vec<(String, bool, String)> {
    let mut results = Vec::with_capacity(entries.len());
    let mut seen: Vec<(String, u64)> = Vec::new();

    for (id, content) in entries {
        let hash = simhash(content);
        let mut is_dup = false;
        let mut dup_of = String::new();

        for (seen_id, seen_hash) in &seen {
            if hamming_distance(hash, *seen_hash) <= threshold {
                is_dup = true;
                dup_of = seen_id.clone();
                break;
            }
        }

        if !is_dup {
            seen.push((id.clone(), hash));
        }

        results.push((id.clone(), is_dup, dup_of));
    }

    results
}

/// Search entries by BM25 relevance to a query.
///
/// Returns (entry_index, score) pairs sorted by descending score.
pub fn search_entries(
    entries: &[String],
    query: &str,
    top_k: usize,
) -> Vec<(usize, f64)> {
    if entries.is_empty() || query.is_empty() {
        return Vec::new();
    }

    // Build document triples for BM25: (id, content, source_path)
    let documents: Vec<(String, String, String)> = entries
        .iter()
        .enumerate()
        .map(|(i, content)| (i.to_string(), content.clone(), String::new()))
        .collect();

    let index = BM25Index::build(&documents);
    let query_terms = tokenize_code(query);

    let mut scored: Vec<(usize, f64)> = entries
        .iter()
        .enumerate()
        .map(|(i, content)| {
            let score = index.score(&query_terms, content, "", &[]);
            (i, score.combined)
        })
        .filter(|(_, score)| *score > 0.0)
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);
    scored
}

/// Compute the SimHash fingerprint for a piece of content.
pub fn fingerprint(text: &str) -> u64 {
    simhash(text)
}

/// Check if two fingerprints are near-duplicates within the given threshold.
pub fn is_near_duplicate(a: u64, b: u64, threshold: u32) -> bool {
    hamming_distance(a, b) <= threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_hash_deterministic() {
        let h1 = content_hash("hello world");
        let h2 = content_hash("hello world");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 16);
    }

    #[test]
    fn test_batch_dedup_finds_duplicates() {
        let entries = vec![
            ("a".to_string(), "The quick brown fox jumps over the lazy dog".to_string()),
            ("b".to_string(), "The quick brown fox jumps over the lazy cat".to_string()),
            ("c".to_string(), "Completely different content about rust programming".to_string()),
        ];

        let results = batch_dedup(&entries, 6);
        assert_eq!(results.len(), 3);
        assert!(!results[0].1);
        assert!(!results[2].1);
    }

    #[test]
    fn test_fingerprint_similar() {
        let a = fingerprint("The quick brown fox jumps over the lazy dog");
        let b = fingerprint("The quick brown fox jumps over the lazy cat");
        assert!(hamming_distance(a, b) < 10);
    }

    #[test]
    fn test_fingerprint_different() {
        let a = fingerprint("The quick brown fox jumps over the lazy dog");
        let b = fingerprint("Rust is a systems programming language focused on safety");
        assert!(hamming_distance(a, b) > 15);
    }

    #[test]
    fn test_search_entries() {
        let entries = vec![
            "Rust is a systems programming language".to_string(),
            "Python is great for data science".to_string(),
            "Rust and Python can work together via PyO3".to_string(),
        ];

        let results = search_entries(&entries, "Rust programming", 2);
        assert!(!results.is_empty());
    }
}
