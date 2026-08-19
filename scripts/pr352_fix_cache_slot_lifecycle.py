#!/usr/bin/env python3
"""Guarded one-shot repair for PR #352 EGSC cache slot lifecycle.

This script intentionally edits only the audited private in-memory index lifecycle.
It aborts if any expected source anchor changed. CacheSnapshot is not modified:
indices remain reconstructed from entries on import.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "entroly-engine/src/cache.rs"
LSH = ROOT / "entroly-engine/src/lsh.rs"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly 1 anchor, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    cache = CACHE.read_text(encoding="utf-8")
    lsh = LSH.read_text(encoding="utf-8")

    # Idempotent exit after the product commit lands and the PR synchronizes.
    if "slot_by_hash: HashMap<u64, usize>" in cache:
        print("cache slot lifecycle repair already applied")
        return 0

    cache = replace_once(
        cache,
        "    entries: HashMap<u64, CacheEntry>,\n    slot_to_hash: Vec<u64>,\n",
        "    entries: HashMap<u64, CacheEntry>,\n"
        "    // Private in-memory semantic-index bookkeeping. These are rebuilt\n"
        "    // from live entries on import and are deliberately not persisted.\n"
        "    slot_to_hash: Vec<u64>,\n"
        "    slot_by_hash: HashMap<u64, usize>,\n"
        "    free_slots: Vec<usize>,\n",
        "cache fields",
    )

    cache = replace_once(
        cache,
        "            entries: HashMap::with_capacity(config.max_entries),\n"
        "            slot_to_hash: Vec::with_capacity(config.max_entries),\n",
        "            entries: HashMap::with_capacity(config.max_entries),\n"
        "            slot_to_hash: Vec::with_capacity(config.max_entries),\n"
        "            slot_by_hash: HashMap::with_capacity(config.max_entries),\n"
        "            free_slots: Vec::new(),\n",
        "cache initialization",
    )

    # Add lifecycle helpers immediately before the budget-aware store path.
    cache = replace_once(
        cache,
        "    /// Budget-aware store used by the engine so exact hits respect the budget that produced them.\n"
        "    #[allow(clippy::too_many_arguments)]\n",
        "    /// Remove one live entry and all of its private lookup-index state.\n"
        "    ///\n"
        "    /// Semantic slots are recycled instead of appended forever. This keeps\n"
        "    /// lookup memory/latency bounded by the cache high-water mark while\n"
        "    /// leaving CacheSnapshot unchanged (indices are rebuilt on import).\n"
        "    fn remove_entry(&mut self, hash: u64) -> bool {\n"
        "        let Some(entry) = self.entries.remove(&hash) else {\n"
        "            return false;\n"
        "        };\n"
        "        self.exact_index.remove(&hash);\n"
        "        if let Some(slot) = self.slot_by_hash.remove(&hash) {\n"
        "            self.semantic_index.remove(entry.query_simhash, slot);\n"
        "            self.free_slots.push(slot);\n"
        "        }\n"
        "        true\n"
        "    }\n\n"
        "    fn allocate_semantic_slot(&mut self, hash: u64, query_fp: u64) {\n"
        "        let slot = if let Some(slot) = self.free_slots.pop() {\n"
        "            debug_assert!(slot < self.slot_to_hash.len());\n"
        "            self.slot_to_hash[slot] = hash;\n"
        "            slot\n"
        "        } else {\n"
        "            let slot = self.slot_to_hash.len();\n"
        "            self.slot_to_hash.push(hash);\n"
        "            slot\n"
        "        };\n"
        "        self.slot_by_hash.insert(hash, slot);\n"
        "        self.semantic_index.insert(query_fp, slot);\n"
        "    }\n\n"
        "    /// Budget-aware store used by the engine so exact hits respect the budget that produced them.\n"
        "    #[allow(clippy::too_many_arguments)]\n",
        "lifecycle helpers",
    )

    cache = replace_once(
        cache,
        "                // Admit: evict the victim\n"
        "                self.entries.remove(&vh);\n"
        "                self.exact_index.remove(&vh);\n"
        "                self.total_evictions += 1;\n",
        "                // Admit: evict the victim and recycle its semantic slot.\n"
        "                self.remove_entry(vh);\n"
        "                self.total_evictions += 1;\n",
        "submodular eviction",
    )

    cache = replace_once(
        cache,
        "        self.exact_index.insert(eh, eh);\n"
        "        let slot = self.slot_to_hash.len();\n"
        "        self.slot_to_hash.push(eh);\n"
        "        self.semantic_index.insert(query_fp, slot);\n"
        "        self.entries.insert(eh, entry);\n",
        "        self.exact_index.insert(eh, eh);\n"
        "        self.allocate_semantic_slot(eh, query_fp);\n"
        "        self.entries.insert(eh, entry);\n",
        "insertion slot allocation",
    )

    cache = replace_once(
        cache,
        "    fn evict_one(&mut self) {\n"
        "        if let Some(hash) = self.find_victim() {\n"
        "            self.entries.remove(&hash);\n"
        "            self.exact_index.remove(&hash);\n"
        "            self.total_evictions += 1;\n"
        "        }\n"
        "    }\n",
        "    fn evict_one(&mut self) {\n"
        "        if let Some(hash) = self.find_victim() {\n"
        "            if self.remove_entry(hash) {\n"
        "                self.total_evictions += 1;\n"
        "            }\n"
        "        }\n"
        "    }\n",
        "fallback eviction",
    )

    cache = replace_once(
        cache,
        "        for hash in &to_remove {\n"
        "            self.entries.remove(hash);\n"
        "            self.exact_index.remove(hash);\n"
        "        }\n",
        "        for hash in &to_remove {\n"
        "            self.remove_entry(*hash);\n"
        "        }\n",
        "gc removal",
    )

    # There are exactly two lifecycle clear sites: clear() and import_cache().
    old_clear = (
        "        self.semantic_index.clear();\n"
        "        self.slot_to_hash.clear();\n"
    )
    if cache.count(old_clear) != 2:
        raise SystemExit(f"index clear sites: expected 2 anchors, found {cache.count(old_clear)}")
    cache = cache.replace(
        old_clear,
        "        self.semantic_index.clear();\n"
        "        self.slot_to_hash.clear();\n"
        "        self.slot_by_hash.clear();\n"
        "        self.free_slots.clear();\n",
    )

    cache = replace_once(
        cache,
        "            let slot = self.slot_to_hash.len();\n"
        "            self.slot_to_hash.push(hash);\n"
        "            self.semantic_index.insert(entry.query_simhash, slot);\n"
        "            // Store entry\n",
        "            let slot = self.slot_to_hash.len();\n"
        "            self.slot_to_hash.push(hash);\n"
        "            self.slot_by_hash.insert(hash, slot);\n"
        "            self.semantic_index.insert(entry.query_simhash, slot);\n"
        "            // Store entry\n",
        "import index rebuild",
    )

    test_anchor = "    // ── Thompson Gate ──\n"
    tests = r'''    fn churn_config(max_entries: usize) -> EgscConfig {
        EgscConfig {
            max_entries,
            enable_entropy_gate: false,
            enable_submodular_eviction: false,
            ..Default::default()
        }
    }

    fn store_churn_entry(cache: &mut EgscCache, index: usize) {
        assert!(cache.store_with_budget(
            &format!("query-{index}"),
            fids(&[&format!("fragment-{index}")]),
            &[],
            format!("response-{index}"),
            32,
            index as u32,
            512,
        ));
    }

    fn assert_semantic_slot_invariants(cache: &EgscCache) {
        assert_eq!(cache.slot_by_hash.len(), cache.entries.len());
        assert_eq!(
            cache.slot_by_hash.len() + cache.free_slots.len(),
            cache.slot_to_hash.len(),
            "every allocated slot must be either live or reusable"
        );
        for (&hash, &slot) in &cache.slot_by_hash {
            assert!(slot < cache.slot_to_hash.len());
            assert_eq!(cache.slot_to_hash[slot], hash);
            assert!(cache.entries.contains_key(&hash));
        }
    }

    #[test]
    fn semantic_slots_stay_bounded_under_long_eviction_churn() {
        let mut cache = EgscCache::new(churn_config(8));
        for index in 0..2_000 {
            store_churn_entry(&mut cache, index);
            assert!(cache.len() <= 8);
            assert!(
                cache.slot_to_hash.len() <= 8,
                "semantic slots grew beyond live cache capacity at iteration {index}: {}",
                cache.slot_to_hash.len()
            );
            assert_semantic_slot_invariants(&cache);
        }

        // No LSH candidate may point at an evicted hash after long churn.
        for index in 0..2_000 {
            for slot in cache.semantic_index.query(simhash(&format!("query-{index}"))) {
                let hash = cache.slot_to_hash[slot];
                assert!(cache.entries.contains_key(&hash));
                assert_eq!(cache.slot_by_hash.get(&hash), Some(&slot));
            }
        }
    }

    #[test]
    fn gc_recycles_slots_and_snapshot_import_rebuilds_private_indices() {
        let mut cache = EgscCache::new(churn_config(4));
        for index in 0..4 {
            store_churn_entry(&mut cache, index);
        }
        let high_water = cache.slot_to_hash.len();
        let victim = *cache.entries.keys().next().expect("expected live cache entry");
        cache.entries.get_mut(&victim).unwrap().quality_score = 0.0;
        assert_eq!(cache.gc(0.1), 1);
        assert_eq!(cache.free_slots.len(), 1);
        assert_semantic_slot_invariants(&cache);

        store_churn_entry(&mut cache, 100);
        assert_eq!(cache.slot_to_hash.len(), high_water, "GC slot must be reused");
        assert!(cache.free_slots.is_empty());
        assert_semantic_slot_invariants(&cache);

        let snapshot = cache.export_cache().expect("snapshot export");
        let mut restored = EgscCache::new(churn_config(1));
        let restored_count = restored.import_cache(&snapshot).expect("snapshot import");
        assert_eq!(restored_count, cache.len());
        assert_eq!(restored.slot_to_hash.len(), restored.len());
        assert!(restored.free_slots.is_empty());
        assert_semantic_slot_invariants(&restored);
    }

'''
    cache = replace_once(cache, test_anchor, tests + test_anchor, "cache lifecycle tests")

    lsh = replace_once(
        lsh,
        "    #[allow(dead_code)]\n    pub fn remove(&mut self, fp: u64, idx: usize) {\n",
        "    pub fn remove(&mut self, fp: u64, idx: usize) {\n",
        "LSH remove activation",
    )

    CACHE.write_text(cache, encoding="utf-8")
    LSH.write_text(lsh, encoding="utf-8")
    print("applied guarded cache slot lifecycle repair")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
