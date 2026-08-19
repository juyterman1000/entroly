#!/usr/bin/env python3
"""Apply PR #352 production safety and long-session hardening.

This guarded transform owns three narrowly scoped fixes:
1. Entroly never mutates the host process's Python GC policy.
2. SAST secret findings never echo bytes from a secret-bearing source line.
3. Work Graph event deduplication uses a derived membership index rather than
   scanning the append-only event log on every append.

The Work Graph persisted document/schema and graph commitment remain unchanged.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "entroly/engine.py"
SAST = ROOT / "entroly-engine/src/sast.rs"
WORK_GRAPH = ROOT / "entroly-engine/src/work_graph.rs"
PROD_E2E = ROOT / "tests/test_production_e2e.py"
GC_TEST = ROOT / "tests/test_engine_host_gc_policy.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


def harden_engine() -> None:
    text = ENGINE.read_text(encoding="utf-8")
    if "Host GC policy is process-wide" in text:
        print("engine GC ownership already hardened")
        return

    text = replace_once(text, "import gc\n", "", "engine gc import")
    text = replace_once(
        text,
        """        # GC freeze at startup: Python's cyclic GC causes ~500ms stalls on large
        # heaps. Freeze all existing long-lived objects and disable automatic
        # collection. We manually collect every N tool calls in advance_turn()
        # to reclaim short-lived garbage without unpredictable pauses.
        self._gc_collect_interval = 50  # collect every 50 turns
        gc.collect()
        gc.freeze()
        gc.disable()
""",
        """        # Host GC policy is process-wide. Entroly deliberately leaves it
        # untouched: a library must not freeze, disable, enable, or collect
        # objects owned by the embedding application or its other threads.
""",
        "engine constructor GC policy",
    )
    text = replace_once(
        text,
        """        # Periodic GC amortization: frozen at init, collect every N turns
        if self._turn_counter > 0 and self._turn_counter % self._gc_collect_interval == 0:
            gc.collect()

""",
        "",
        "engine periodic GC",
    )

    start_marker = "    def ingest_fragment(\n"
    end_marker = "    def remove_sources("
    start = text.find(start_marker)
    if start < 0:
        raise SystemExit("engine ingest_fragment start anchor missing")
    end = text.find(end_marker, start)
    if end < 0:
        raise SystemExit("engine remove_sources end anchor missing")
    old_block = text[start:end]
    if "gc.disable()" not in old_block or "gc.enable()" not in old_block:
        raise SystemExit("engine ingest GC anchors missing")

    new_block = '''    def ingest_fragment(
        self,
        content: str,
        source: str = "",
        token_count: int = 0,
        is_pinned: bool = False,
    ) -> dict[str, Any]:
        """Ingest a new context fragment without changing host GC policy."""
        # Lazy warm-start MUST run before the first mutation: load_index
        # replaces the fragment set, so it has to happen before any ingest or
        # it would wipe freshly-ingested fragments.
        self._ensure_index_loaded()

        # Invalidate the fast-path's fragment cache: the Rust engine has
        # gained a new fragment, so any cached export_fragments() snapshot
        # is now stale.
        self._fragment_cache_dirty = True
        if self._use_rust:
            # Enforce max_fragments cap on Rust engine (Rust doesn't enforce it)
            if self._rust.fragment_count() >= self.config.max_fragments:
                return {
                    "status": "rejected",
                    "reason": "max_fragments cap reached",
                    "max_fragments": self.config.max_fragments,
                }
            result = self._rust.ingest(content, source, token_count, is_pinned)
            # result is a dict from PyO3
            if source:
                self._prefetch.record_access(source, self._rust.get_turn())
            if self._checkpoint_mgr.should_auto_checkpoint():
                self._auto_checkpoint()
            return dict(result)
        return self._ingest_python(content, source, token_count, is_pinned)

'''
    text = text[:start] + new_block + text[end:]
    if "gc." in text or "import gc" in text:
        raise SystemExit("engine.py still contains process-global GC manipulation")
    ENGINE.write_text(text, encoding="utf-8")
    print("hardened Python host GC ownership")


def harden_sast() -> None:
    text = SAST.read_text(encoding="utf-8")
    if "[REDACTED — secret-bearing line]" in text:
        print("SAST zero-secret-byte redaction already hardened")
        return

    old = '''            // Privacy: redact line content for secret-category findings
            // so that actual credentials are never exposed in SAST reports.
            let safe_content = if rule.category == "Hardcoded Secrets" {
                let trimmed = line.trim();
                if let Some(eq_pos) = trimmed.find('=') {
                    // Show key name but redact value: "api_key = [REDACTED]"
                    format!("{}= [REDACTED]", &trimmed[..eq_pos])
                } else if trimmed.len() > 30 {
                    // Find nearest valid UTF-8 boundary at or before byte 20
                    let safe = (0..=20.min(trimmed.len()))
                        .rev()
                        .find(|&i| trimmed.is_char_boundary(i))
                        .unwrap_or(0);
                    format!("{}...[REDACTED]", &trimmed[..safe])
                } else {
                    "[REDACTED — secret detected]".to_string()
                }
            } else {
                line.trim().to_string()
            };
'''
    new = '''            // Privacy invariant: once a line matches a secret-category rule,
            // never echo *any* source bytes from that line. Even a truncated
            // prefix can disclose a credential when the token begins the line,
            // appears on the left-hand side, or is embedded in non-assignment
            // syntax. Rule/category/path/line metadata already carries the
            // debugging context without repeating secret material.
            let safe_content = if rule.category == "Hardcoded Secrets" {
                "[REDACTED — secret-bearing line]".to_string()
            } else {
                line.trim().to_string()
            };
'''
    text = replace_once(text, old, new, "SAST secret redaction")

    old_test_tail = '''        // Should still show the key name for debugging
        assert!(
            finding.line_content.contains("password"),
            "Key name should be preserved for context: {}",
            finding.line_content
        );
    }

    #[test]
    fn test_openai_key_redacted_not_leaked() {
'''
    new_test_tail = '''        assert_eq!(finding.line_content, "[REDACTED — secret-bearing line]");
    }

    #[test]
    fn secret_findings_never_echo_source_bytes_at_any_position() {
        let secret = "sk-proj-supersecret123456789";
        let cases = [
            format!("{secret} trailing diagnostic text"),
            format!("prefix text {secret} trailing text"),
            format!("prefix text ending with {secret}"),
            format!("{secret} = placeholder"),
        ];
        for code in cases {
            let report = scan(&code, "leak.txt");
            let finding = report
                .findings
                .iter()
                .find(|finding| finding.rule_id == "SEC-003")
                .expect("sk- token must still be detected");
            assert_eq!(
                finding.line_content,
                "[REDACTED — secret-bearing line]",
                "secret findings must not retain source bytes"
            );
            let serialized = serde_json::to_string(&report).unwrap();
            assert!(
                !serialized.contains(secret),
                "serialized SAST report leaked secret bytes: {serialized}"
            );
        }
    }

    #[test]
    fn test_openai_key_redacted_not_leaked() {
'''
    text = replace_once(text, old_test_tail, new_test_tail, "SAST redaction regression")
    SAST.write_text(text, encoding="utf-8")
    print("hardened SAST zero-secret-byte reports")


def harden_work_graph() -> None:
    text = WORK_GRAPH.read_text(encoding="utf-8")
    if "event_ids: BTreeSet<String>" in text:
        print("Work Graph event membership index already hardened")
        return

    text = replace_once(
        text,
        '''    repo_id: String,
    events: Vec<WorkEvent>,
    nodes: BTreeMap<String, WorkNode>,
''',
        '''    repo_id: String,
    events: Vec<WorkEvent>,
    /// Derived membership index for the append-only event log. Never serialized.
    /// Keeping this separate removes an O(N) scan from every long-session append.
    event_ids: BTreeSet<String>,
    nodes: BTreeMap<String, WorkNode>,
''',
        "WorkGraph derived event index field",
    )
    text = replace_once(
        text,
        '''            repo_id,
            events: Vec::new(),
            nodes: BTreeMap::new(),
''',
        '''            repo_id,
            events: Vec::new(),
            event_ids: BTreeSet::new(),
            nodes: BTreeMap::new(),
''',
        "WorkGraph event index init",
    )
    text = replace_once(
        text,
        '''        if self.events.iter().any(|existing| existing.event_id == id) {
            return Ok(id);
        }
''',
        '''        if self.event_ids.contains(&id) {
            return Ok(id);
        }
''',
        "WorkGraph append duplicate lookup",
    )
    text = replace_once(
        text,
        '''        self.events.push(event.clone());

        let result = if append_in_order {
''',
        '''        self.events.push(event.clone());
        self.event_ids.insert(id.clone());

        let result = if append_in_order {
''',
        "WorkGraph append event index insert",
    )
    text = replace_once(
        text,
        '''        let mut candidate = self.events.clone();
        let mut existing: BTreeSet<String> = candidate.iter().map(|e| e.event_id.clone()).collect();
''',
        '''        let mut candidate = self.events.clone();
        let mut existing = self.event_ids.clone();
''',
        "WorkGraph merge membership seed",
    )
    text = replace_once(
        text,
        '''        self.nodes.clear();
        self.edges.clear();
        self.evidence.clear();
        self.adjacency.clear();
        let events = self.events.clone();
        for event in &events {
            self.apply_materialized(event)?;
        }
''',
        '''        self.event_ids.clear();
        self.nodes.clear();
        self.edges.clear();
        self.evidence.clear();
        self.adjacency.clear();
        let events = self.events.clone();
        for event in &events {
            if !self.event_ids.insert(event.event_id.clone()) {
                return Err(WorkGraphError::InvalidInput(format!(
                    "duplicate event id in work graph: {}",
                    event.event_id
                )));
            }
            self.apply_materialized(event)?;
        }
''',
        "WorkGraph rebuild event index",
    )

    test_anchor = '''    #[test]
    fn passive_snapshot_byte_change_appends_new_event() {
'''
    test = '''    #[test]
    fn derived_event_id_index_tracks_append_dedupe_and_import() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        for i in 0..2_048u64 {
            let digest = format!("git-blob:{:040x}", i + 1);
            graph
                .observe_repository(passive_dirty_observation(&digest, 1_000 + i as i64))
                .unwrap();
        }
        assert_eq!(graph.event_ids.len(), graph.events.len());
        assert_eq!(graph.event_ids.len(), 2_048);

        let duplicate = graph.events[1_024].clone();
        let duplicate_id = duplicate.event_id.clone();
        let before = graph.graph_commitment().to_string();
        assert_eq!(graph.apply_event(duplicate).unwrap(), duplicate_id);
        assert_eq!(graph.event_count(), 2_048);
        assert_eq!(graph.event_ids.len(), 2_048);
        assert_eq!(graph.graph_commitment(), before);

        let restored = WorkGraph::from_json(&graph.export_json(false).unwrap()).unwrap();
        assert_eq!(restored.event_ids.len(), restored.events.len());
        assert_eq!(restored.event_ids, graph.event_ids);
        assert_eq!(restored.graph_commitment(), graph.graph_commitment());
    }

    #[test]
    fn passive_snapshot_byte_change_appends_new_event() {
'''
    text = replace_once(text, test_anchor, test, "WorkGraph event index regression")
    WORK_GRAPH.write_text(text, encoding="utf-8")
    print("hardened Work Graph long-session event membership")


def update_production_e2e() -> None:
    text = PROD_E2E.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "  P-04  GC FREEZE STATE              gc is disabled after engine __init__; re-enabled after optimize\n",
        "  P-04  HOST GC OWNERSHIP            engine construction/use preserves caller GC policy\n",
        "production E2E GC description",
    )
    old = '''    # ─── P-04: GC Freeze State ───────────────────────────────────────────────────
    section("P-04  GC FREEZE STATE")
    # GC is disabled by EntrolyEngine.__init__ (server.py) at startup
    # Cannot re-test __init__ here without reinitializing, but we can verify
    # the pattern manually:
    gc.enable()
    gc.isenabled()
    gc.disable()
    try:
        _ = [x * x for x in range(1000)]  # no GC during tight loop
    finally:
        gc.enable()
        gc.collect()
    ok("manual gc.disable/enable cycle works correctly", gc.isenabled())
    ok("gc.freeze does not raise on non-empty heap", True)  # structural

'''
    new = '''    # ─── P-04: Host GC ownership ─────────────────────────────────────────────────
    section("P-04  HOST GC OWNERSHIP")
    from entroly.config import EntrolyConfig as PythonConfig
    from entroly.engine import EntrolyEngine as PythonEngine

    original_gc = gc.isenabled()
    try:
        for expected in (True, False):
            if expected:
                gc.enable()
            else:
                gc.disable()
            py_engine = PythonEngine(
                PythonConfig(
                    checkpoint_dir=Path(test_state.name) / f"gc-{expected}",
                    use_persistent_index=False,
                )
            )
            ok(
                f"engine construction preserves GC enabled={expected}",
                gc.isenabled() is expected,
            )
            py_engine.ingest_fragment("def gc_policy_probe(): return 1", "gc_policy.py", 8)
            ok(
                f"ingest preserves GC enabled={expected}",
                gc.isenabled() is expected,
            )
            py_engine.advance_turn()
            ok(
                f"advance_turn preserves GC enabled={expected}",
                gc.isenabled() is expected,
            )
    finally:
        if original_gc:
            gc.enable()
        else:
            gc.disable()

'''
    text = replace_once(text, old, new, "production E2E GC contract")
    PROD_E2E.write_text(text, encoding="utf-8")


def write_gc_test() -> None:
    content = '''"""Library-safety contract: Entroly never owns the host process GC policy."""

from __future__ import annotations

import gc
from pathlib import Path

import pytest

from entroly.config import EntrolyConfig
from entroly.engine import EntrolyEngine


def _restore_gc(enabled: bool) -> None:
    if enabled:
        gc.enable()
    else:
        gc.disable()


@pytest.mark.parametrize("enabled", [True, False])
def test_engine_construction_and_hot_paths_preserve_host_gc_policy(
    tmp_path: Path, enabled: bool
) -> None:
    original = gc.isenabled()
    try:
        _restore_gc(enabled)
        engine = EntrolyEngine(
            EntrolyConfig(
                checkpoint_dir=tmp_path / ("enabled" if enabled else "disabled"),
                use_persistent_index=False,
            )
        )
        assert gc.isenabled() is enabled

        result = engine.ingest_fragment(
            "def host_gc_policy_probe(): return 1",
            "gc_policy.py",
            8,
            False,
        )
        assert result.get("status") in {"ingested", "duplicate"}
        assert gc.isenabled() is enabled

        engine.advance_turn()
        assert gc.isenabled() is enabled
    finally:
        _restore_gc(original)


def test_engine_source_contains_no_process_global_gc_controls() -> None:
    source = (Path(__file__).resolve().parents[1] / "entroly" / "engine.py").read_text(
        encoding="utf-8"
    )
    for forbidden in ("gc.disable(", "gc.enable(", "gc.freeze(", "gc.collect("):
        assert forbidden not in source
'''
    if GC_TEST.exists():
        existing = GC_TEST.read_text(encoding="utf-8")
        if existing != content:
            raise SystemExit(f"refusing to overwrite unexpected {GC_TEST}")
        return
    GC_TEST.write_text(content, encoding="utf-8")
    print("added host GC policy regressions")


def main() -> int:
    harden_engine()
    harden_sast()
    harden_work_graph()
    update_production_e2e()
    write_gc_test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
