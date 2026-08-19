#!/usr/bin/env python3
"""Make Work Graph commitments incremental without changing their bytes.

The public/persisted commitment remains SHA-256 over the exact compact serde JSON
object `{schema_version, repo_id, events}`. The private runtime keeps the SHA-256
state immediately before the closing `]}` so an in-order append hashes only the
new canonical event bytes. Rebuild/import reconstruct the state from the source
of truth event log.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "entroly-engine/src/work_graph.rs"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    text = PATH.read_text(encoding="utf-8")
    if "commitment_hasher: Sha256" in text:
        print("incremental Work Graph commitment already applied")
        return 0
    if "event_ids: BTreeSet<String>" not in text:
        raise SystemExit("event membership hardening must run before commitment hardening")

    text = replace_once(
        text,
        '''    event_ids: BTreeSet<String>,
    nodes: BTreeMap<String, WorkNode>,
''',
        '''    event_ids: BTreeSet<String>,
    /// SHA-256 state for the canonical commitment bytes through the open events array.
    /// Derived runtime state only; persisted WorkGraphDocument remains unchanged.
    commitment_hasher: Sha256,
    nodes: BTreeMap<String, WorkNode>,
''',
        "WorkGraph commitment hasher field",
    )
    text = replace_once(
        text,
        '''            event_ids: BTreeSet::new(),
            nodes: BTreeMap::new(),
''',
        '''            event_ids: BTreeSet::new(),
            commitment_hasher: Sha256::new(),
            nodes: BTreeMap::new(),
''',
        "WorkGraph commitment hasher init",
    )
    text = replace_once(
        text,
        '''        let result = if append_in_order {
            self.apply_materialized(&event)
                .and_then(|_| self.refresh_commitment())
        } else {
            self.rebuild()
        };
''',
        '''        let result = if append_in_order {
            self.apply_materialized(&event)
                .and_then(|_| self.append_commitment_event(&event))
        } else {
            self.rebuild()
        };
''',
        "WorkGraph append commitment path",
    )

    old_refresh = '''    fn refresh_commitment(&mut self) -> Result<(), WorkGraphError> {
        #[derive(Serialize)]
        struct Commitment<'a> {
            schema_version: u32,
            repo_id: &'a str,
            events: &'a [WorkEvent],
        }
        self.graph_commitment = sha256_json(&Commitment {
            schema_version: WORK_GRAPH_SCHEMA_VERSION,
            repo_id: &self.repo_id,
            events: &self.events,
        })?;
        Ok(())
    }
'''
    new_refresh = '''    fn commitment_prefix_hasher(repo_id: &str) -> Result<Sha256, WorkGraphError> {
        let mut hasher = Sha256::new();
        hasher.update(b"{\\\"schema_version\\\":");
        hasher.update(WORK_GRAPH_SCHEMA_VERSION.to_string().as_bytes());
        hasher.update(b",\\\"repo_id\\\":");
        hasher.update(serde_json::to_vec(repo_id)?);
        hasher.update(b",\\\"events\\\":[");
        Ok(hasher)
    }

    fn finalize_commitment(hasher: &Sha256) -> String {
        let mut final_hasher = hasher.clone();
        final_hasher.update(b"]}");
        format!("{:x}", final_hasher.finalize())
    }

    fn refresh_commitment(&mut self) -> Result<(), WorkGraphError> {
        let mut hasher = Self::commitment_prefix_hasher(&self.repo_id)?;
        for (index, event) in self.events.iter().enumerate() {
            if index > 0 {
                hasher.update(b",");
            }
            hasher.update(serde_json::to_vec(event)?);
        }
        self.graph_commitment = Self::finalize_commitment(&hasher);
        self.commitment_hasher = hasher;
        Ok(())
    }

    fn append_commitment_event(&mut self, event: &WorkEvent) -> Result<(), WorkGraphError> {
        let event_bytes = serde_json::to_vec(event)?;
        if self.events.len() > 1 {
            self.commitment_hasher.update(b",");
        }
        self.commitment_hasher.update(event_bytes);
        self.graph_commitment = Self::finalize_commitment(&self.commitment_hasher);
        Ok(())
    }
'''
    text = replace_once(text, old_refresh, new_refresh, "WorkGraph commitment implementation")

    test_anchor = '''    #[test]
    fn derived_event_id_index_tracks_append_dedupe_and_import() {
'''
    test = '''    fn canonical_full_graph_commitment(graph: &WorkGraph) -> String {
        #[derive(Serialize)]
        struct Commitment<'a> {
            schema_version: u32,
            repo_id: &'a str,
            events: &'a [WorkEvent],
        }
        sha256_json(&Commitment {
            schema_version: WORK_GRAPH_SCHEMA_VERSION,
            repo_id: &graph.repo_id,
            events: &graph.events,
        })
        .unwrap()
    }

    #[test]
    fn incremental_commitment_matches_canonical_serde_bytes() {
        let mut graph = WorkGraph::new("repo-commitment-parity").unwrap();
        assert_eq!(
            graph.graph_commitment(),
            canonical_full_graph_commitment(&graph)
        );
        for i in 0..512u64 {
            let digest = format!("git-blob:{:040x}", i + 1);
            graph
                .observe_repository(passive_dirty_observation(&digest, 10_000 + i as i64))
                .unwrap();
            assert_eq!(
                graph.graph_commitment(),
                canonical_full_graph_commitment(&graph),
                "incremental commitment diverged after append {i}"
            );
        }

        let compact = graph.export_json(false).unwrap();
        let restored = WorkGraph::from_json(&compact).unwrap();
        assert_eq!(restored.graph_commitment(), graph.graph_commitment());
        assert_eq!(
            restored.graph_commitment(),
            canonical_full_graph_commitment(&restored)
        );
    }

    #[test]
    fn derived_event_id_index_tracks_append_dedupe_and_import() {
'''
    text = replace_once(text, test_anchor, test, "WorkGraph commitment parity regression")

    PATH.write_text(text, encoding="utf-8")
    print("applied exact incremental Work Graph commitment state")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
