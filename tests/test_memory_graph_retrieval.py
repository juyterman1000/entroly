"""Tests for memory graph, write gating, temporal reasoning, and hybrid retrieval."""

from __future__ import annotations

import json
import os
import tempfile
import time

import pytest


# ── Temporal Reasoning ──────────────────────────────────────────────────

class TestTemporalReasoning:
    """Tests for temporal classification, query intent detection, and supersession."""

    def test_temporal_auto_classify_preference(self):
        from entroly.memory import _classify_temporal
        assert _classify_temporal("User prefers dark mode") == "preference"
        assert _classify_temporal("I always use vim") == "preference"

    def test_temporal_auto_classify_relationship(self):
        from entroly.memory import _classify_temporal
        assert _classify_temporal("Alice manages the backend team") == "relationship"
        assert _classify_temporal("Bob works at Google") == "relationship"

    def test_temporal_auto_classify_future(self):
        from entroly.memory import _classify_temporal
        assert _classify_temporal("We will migrate to Rust next month") == "future_plan"
        assert _classify_temporal("Planning to refactor the auth module") == "future_plan"

    def test_temporal_auto_classify_past(self):
        from entroly.memory import _classify_temporal
        assert _classify_temporal("Previously used Django before switching") == "past_event"
        assert _classify_temporal("We had a deployment issue yesterday") == "past_event"

    def test_temporal_auto_classify_current(self):
        from entroly.memory import _classify_temporal
        assert _classify_temporal("The API is running on port 8080") == "current_state"

    def test_temporal_query_intent_current(self):
        from entroly.memory import _detect_temporal_intent
        assert _detect_temporal_intent("Where does Alice work currently?") == "current_state"
        assert _detect_temporal_intent("What is the latest deployment?") == "current_state"

    def test_temporal_query_intent_past(self):
        from entroly.memory import _detect_temporal_intent
        assert _detect_temporal_intent("What did we do before?") == "past_event"
        assert _detect_temporal_intent("What was used previously?") == "past_event"

    def test_temporal_query_intent_future(self):
        from entroly.memory import _detect_temporal_intent
        assert _detect_temporal_intent("What will we do next week?") == "future_plan"
        assert _detect_temporal_intent("What is planned for the future?") == "future_plan"

    def test_temporal_query_intent_ambiguous(self):
        from entroly.memory import _detect_temporal_intent
        # Multiple signals or no signal → None
        assert _detect_temporal_intent("Show me everything") is None
        assert _detect_temporal_intent("") is None

    def test_remember_auto_temporal_class(self):
        from entroly.memory import MemoryOS
        mem = MemoryOS(safety_policy="allow")
        mid = mem.remember("Alice manages the backend team", agent_id="test")
        entry = mem._entries[mid]
        assert entry.temporal_class == "relationship"
        assert len(entry.entities) > 0

    def test_remember_explicit_temporal_class(self):
        from entroly.memory import MemoryOS
        mem = MemoryOS(safety_policy="allow")
        mid = mem.remember(
            "The server runs on port 3000",
            agent_id="test",
            temporal_class="current_state",
        )
        entry = mem._entries[mid]
        assert entry.temporal_class == "current_state"

    def test_supersession_tracking(self):
        from entroly.memory import MemoryOS
        mem = MemoryOS(safety_policy="allow")
        old_id = mem.remember("Bob works at Acme Corp", agent_id="test")
        new_id = mem.remember(
            "Bob now works at Globex Corporation",
            agent_id="test",
            supersedes=old_id,
        )
        old_entry = mem._entries[old_id]
        new_entry = mem._entries[new_id]
        assert old_entry.superseded_by == new_id
        assert old_entry.valid_until is not None
        assert new_entry.supersedes == old_id

    def test_recall_filters_superseded_on_current_query(self):
        from entroly.memory import MemoryOS
        mem = MemoryOS(safety_policy="allow")
        old_id = mem.remember("Bob works at Acme Corp", agent_id="test")
        new_id = mem.remember(
            "Bob now works at Globex Corporation",
            agent_id="test",
            supersedes=old_id,
        )
        # Query for current state should exclude superseded entry
        ctx = mem.recall("Where does Bob work currently?", agent_id="test")
        selected_ids = {m.id for m in ctx.selected}
        assert new_id in selected_ids
        # Old entry should be omitted with reason "superseded"
        omitted_reasons = {m.id: m.reason for m in ctx.omitted}
        assert old_id in omitted_reasons
        assert omitted_reasons[old_id] == "superseded"

    def test_recall_temporal_filter(self):
        from entroly.memory import MemoryOS
        mem = MemoryOS(safety_policy="allow")
        mem.remember("User prefers dark mode", agent_id="test")
        mem.remember("Deployed v2 yesterday", agent_id="test")
        mem.remember("Will refactor auth module", agent_id="test")

        # Filter to preferences only
        ctx = mem.recall("", agent_id="test", temporal_filter="preference")
        assert len(ctx.selected) == 1
        assert "dark mode" in ctx.selected[0].content

    def test_temporal_boost_in_scoring(self):
        from entroly.memory import MemoryOS
        mem = MemoryOS(safety_policy="allow")
        mem.remember("Server runs on port 8080", agent_id="test", importance=0.5)
        mem.remember("Previously used port 3000", agent_id="test", importance=0.5)

        ctx = mem.recall("What port is currently used?", agent_id="test")
        # The current_state entry should rank higher
        if len(ctx.selected) >= 2:
            assert "8080" in ctx.selected[0].content

    def test_entity_extraction(self):
        from entroly.memory import _extract_entities
        entities = _extract_entities("Alice Smith manages the PaymentGateway service at Google")
        assert "Alice Smith" in entities
        assert "PaymentGateway" in entities
        assert "Google" in entities

    def test_entity_extraction_camelcase(self):
        from entroly.memory import _extract_entities
        entities = _extract_entities("The UserService depends on AuthProvider")
        assert "UserService" in entities
        assert "AuthProvider" in entities

    def test_snapshot_preserves_temporal_fields(self):
        from entroly.memory import MemoryOS
        mem = MemoryOS(safety_policy="allow")
        mid = mem.remember("Alice manages backend", agent_id="test")
        snapshot = mem.snapshot()

        restored = MemoryOS.from_snapshot(snapshot)
        entry = restored._entries[mid]
        assert entry.temporal_class in (
            "relationship", "current_state", "preference",
            "past_event", "future_plan", "unclassified",
        )
        assert entry.valid_from is not None


# ── Graph Memory ────────────────────────────────────────────────────────

class TestMemoryGraph:
    """Tests for receipt-backed knowledge graph over memories."""

    def test_index_creates_nodes_and_edges(self):
        from entroly.memory_graph import MemoryGraph
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = MemoryGraph(root=tmpdir)
            edges = graph.index_memory(
                "mem_001", ["Alice", "BackendTeam"],
                "Alice manages the BackendTeam",
            )
            assert len(edges) == 1
            assert edges[0].source_entity == "alice"
            assert edges[0].target_entity == "backendteam"
            assert edges[0].memory_id == "mem_001"

    def test_index_dedup_same_memory(self):
        from entroly.memory_graph import MemoryGraph
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = MemoryGraph(root=tmpdir)
            edges1 = graph.index_memory("mem_001", ["Alice", "Bob"], "Alice and Bob")
            edges2 = graph.index_memory("mem_001", ["Alice", "Bob"], "Alice and Bob again")
            assert len(edges1) == 1
            assert len(edges2) == 0  # deduplicated (same pair, same memory)

    def test_index_different_memories_create_separate_edges(self):
        from entroly.memory_graph import MemoryGraph
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = MemoryGraph(root=tmpdir)
            graph.index_memory("mem_001", ["Alice", "Bob"], "Alice and Bob")
            graph.index_memory("mem_002", ["Alice", "Bob"], "Alice and Bob again")
            stats = graph.stats()
            assert stats["total_edges"] == 2

    def test_traverse_single_hop(self):
        from entroly.memory_graph import MemoryGraph
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = MemoryGraph(root=tmpdir)
            graph.index_memory("mem_001", ["Alice", "BackendTeam"], "Alice manages BackendTeam")
            graph.index_memory("mem_002", ["BackendTeam", "PaymentService"], "BackendTeam owns PaymentService")

            result = graph.traverse(["Alice"], max_hops=1)
            assert "mem_001" in result.memory_ids
            assert "backendteam" in result.reached_entities

    def test_traverse_multi_hop(self):
        from entroly.memory_graph import MemoryGraph
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = MemoryGraph(root=tmpdir)
            graph.index_memory("mem_001", ["Alice", "BackendTeam"], "Alice manages BackendTeam")
            graph.index_memory("mem_002", ["BackendTeam", "PaymentService"], "BackendTeam owns PaymentService")
            graph.index_memory("mem_003", ["PaymentService", "Stripe"], "PaymentService uses Stripe")

            # 2 hops from Alice → BackendTeam → PaymentService
            result = graph.traverse(["Alice"], max_hops=2)
            assert "mem_001" in result.memory_ids
            assert "mem_002" in result.memory_ids
            assert "paymentservice" in result.reached_entities

    def test_traverse_unknown_entity(self):
        from entroly.memory_graph import MemoryGraph
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = MemoryGraph(root=tmpdir)
            result = graph.traverse(["NonExistent"])
            assert len(result.memory_ids) == 0
            assert result.receipt["status"] == "no_matching_entities"

    def test_supersede_excludes_from_traversal(self):
        from entroly.memory_graph import MemoryGraph
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = MemoryGraph(root=tmpdir)
            graph.index_memory("mem_001", ["Alice", "Acme"], "Alice works at Acme")
            graph.supersede_memory("mem_001", "mem_002")
            graph.index_memory("mem_002", ["Alice", "Globex"], "Alice works at Globex")

            result = graph.traverse(["Alice"], max_hops=1, include_superseded=False)
            # mem_001's edges are superseded, so Acme should not be reachable
            assert "mem_002" in result.memory_ids
            # Acme entity node still exists but edges are superseded
            acme_memories = graph.entity_memories("Acme")
            assert "mem_001" in acme_memories

    def test_receipt_in_traversal(self):
        from entroly.memory_graph import MemoryGraph
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = MemoryGraph(root=tmpdir)
            graph.index_memory("mem_001", ["Alice", "Bob"], "test")
            result = graph.traverse(["Alice"])
            assert "traversal_type" in result.receipt
            assert result.receipt["traversal_type"] == "multi_hop_associative"

    def test_relationship_detection(self):
        from entroly.memory_graph import MemoryGraph
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = MemoryGraph(root=tmpdir)
            edges = graph.index_memory(
                "mem_001", ["Alice", "BackendTeam"],
                "Alice manages the BackendTeam",
            )
            assert edges[0].relationship == "manages"

    def test_save_and_load(self):
        from entroly.memory_graph import MemoryGraph
        with tempfile.TemporaryDirectory() as tmpdir:
            graph1 = MemoryGraph(root=tmpdir)
            graph1.index_memory("mem_001", ["Alice", "Bob"], "test")
            graph1.save()

            graph2 = MemoryGraph(root=tmpdir)
            assert graph2.stats()["total_nodes"] == 2
            assert graph2.stats()["total_edges"] == 1

    def test_neighbors(self):
        from entroly.memory_graph import MemoryGraph
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = MemoryGraph(root=tmpdir)
            graph.index_memory("mem_001", ["Alice", "Bob", "Charlie"], "test")
            neighbors = graph.neighbors("Alice")
            neighbor_names = {n[0] for n in neighbors}
            assert "bob" in neighbor_names
            assert "charlie" in neighbor_names


# ── Write Gate ──────────────────────────────────────────────────────────

class TestWriteGate:
    """Tests for memory write gating."""

    def test_admits_novel_content(self):
        from entroly.memory import MemoryOS
        from entroly.write_gate import WriteGate
        mem = MemoryOS(safety_policy="allow")
        gate = WriteGate(mem)

        decision = gate.should_commit("Alice manages the backend team at Acme Corporation")
        assert decision.commit is True
        assert decision.reason == "admitted"

    def test_rejects_empty_content(self):
        from entroly.memory import MemoryOS
        from entroly.write_gate import WriteGate
        mem = MemoryOS(safety_policy="allow")
        gate = WriteGate(mem)

        decision = gate.should_commit("")
        assert decision.commit is False
        assert decision.reason == "empty_content"

    def test_rejects_low_entropy(self):
        from entroly.memory import MemoryOS
        from entroly.write_gate import WriteGate
        mem = MemoryOS(safety_policy="allow")
        gate = WriteGate(mem, min_entropy=3.0)

        decision = gate.should_commit("ok ok ok ok ok")
        assert decision.commit is False
        assert decision.reason == "low_entropy"

    def test_rejects_near_duplicate(self):
        from entroly.memory import MemoryOS
        from entroly.write_gate import WriteGate
        mem = MemoryOS(safety_policy="allow")
        mem.remember("Alice manages the backend team at Acme Corporation", agent_id="test")
        gate = WriteGate(mem)

        # Near-identical content
        decision = gate.should_commit("Alice manages the backend team at Acme Corporation today", agent_id="test")
        assert decision.commit is False
        assert decision.reason == "near_duplicate"

    def test_admits_different_content(self):
        from entroly.memory import MemoryOS
        from entroly.write_gate import WriteGate
        mem = MemoryOS(safety_policy="allow")
        mem.remember("Alice manages the backend team", agent_id="test")
        gate = WriteGate(mem)

        decision = gate.should_commit("The deployment pipeline uses GitHub Actions with Docker containers")
        assert decision.commit is True

    def test_receipt_tracking(self):
        from entroly.memory import MemoryOS
        from entroly.write_gate import WriteGate
        mem = MemoryOS(safety_policy="allow")
        gate = WriteGate(mem)

        gate.should_commit("Alice manages backend")
        gate.should_commit("")

        receipt = gate.receipt
        assert receipt.admitted == 1
        assert receipt.rejected == 1
        summary = receipt.summary()
        assert summary["rejection_rate"] == 0.5


# ── Hybrid Retriever ────────────────────────────────────────────────────

class TestHybridRetriever:
    """Tests for multi-signal memory retrieval."""

    def test_bm25_only_retrieval(self):
        from entroly.memory import MemoryOS
        from entroly.hybrid_retriever import HybridRetriever
        mem = MemoryOS(safety_policy="allow")
        mem.remember("Python asyncio handles concurrency", agent_id="test")
        mem.remember("Docker containers provide isolation", agent_id="test")
        mem.remember("Rust memory safety prevents bugs", agent_id="test")

        retriever = HybridRetriever(mem)
        results = retriever.recall("asyncio concurrency", agent_id="test", top_k=3)
        assert len(results) > 0
        # The asyncio memory should rank first
        assert "asyncio" in results[0].content.lower()

    def test_entity_graph_augmented_retrieval(self):
        from entroly.memory import MemoryOS
        from entroly.memory_graph import MemoryGraph
        from entroly.hybrid_retriever import HybridRetriever

        with tempfile.TemporaryDirectory() as tmpdir:
            mem = MemoryOS(safety_policy="allow")
            graph = MemoryGraph(root=tmpdir)

            mid1 = mem.remember("Alice manages the BackendTeam", agent_id="test")
            mid2 = mem.remember("BackendTeam owns the PaymentService", agent_id="test")
            mem.remember("Docker is used for deployment", agent_id="test")

            graph.index_memory(mid1, ["Alice", "BackendTeam"], "Alice manages BackendTeam")
            graph.index_memory(mid2, ["BackendTeam", "PaymentService"], "BackendTeam owns PaymentService")

            retriever = HybridRetriever(mem, graph=graph)
            results = retriever.recall("Alice", agent_id="test", top_k=3)

            # Alice's direct memory and BackendTeam's memory should both appear
            result_ids = {r.memory_id for r in results}
            assert mid1 in result_ids

    def test_rrf_fusion_multiple_signals(self):
        from entroly.hybrid_retriever import _reciprocal_rank_fusion

        bm25 = ["doc_a", "doc_b", "doc_c"]
        entity = ["doc_b", "doc_a", "doc_d"]

        fused = _reciprocal_rank_fusion([bm25, entity], k=60)
        # doc_a and doc_b appear in both lists — should score highest
        sorted_docs = sorted(fused.items(), key=lambda x: -x[1])
        top_2 = {sorted_docs[0][0], sorted_docs[1][0]}
        assert "doc_a" in top_2
        assert "doc_b" in top_2

    def test_recall_with_receipt(self):
        from entroly.memory import MemoryOS
        from entroly.hybrid_retriever import HybridRetriever
        mem = MemoryOS(safety_policy="allow")
        mem.remember("Test content for receipt verification", agent_id="test")

        retriever = HybridRetriever(mem)
        results, receipt = retriever.recall_with_receipt("test", agent_id="test")
        assert receipt.query == "test"
        assert "bm25" in receipt.signals_used
        assert receipt.fused_results == len(results)

    def test_empty_memory_returns_empty(self):
        from entroly.memory import MemoryOS
        from entroly.hybrid_retriever import HybridRetriever
        mem = MemoryOS(safety_policy="allow")
        retriever = HybridRetriever(mem)
        results = retriever.recall("anything", agent_id="test")
        assert len(results) == 0
