"""Tests for MemoryFabric integration with MemoryGraph, WriteGate, and HybridRetriever."""

from entroly.memory_fabric import MemoryFabric


def test_fabric_capabilities_include_new_layers():
    fabric = MemoryFabric()
    caps = {layer.name: layer.status for layer in fabric.capabilities()}
    assert caps["memory_os"] == "active"
    assert caps["memory_graph"] == "active"
    assert caps["write_gate"] == "active"
    assert caps["hybrid_retriever"] == "active"


def test_fabric_stats_include_new_metrics():
    fabric = MemoryFabric()
    stats = fabric.stats()
    assert "memory_graph" in stats
    assert "node_count" in stats["memory_graph"]
    assert "edge_count" in stats["memory_graph"]
    assert "write_gate" in stats
    assert "admitted" in stats["write_gate"]
    assert "hybrid_retriever" in stats
    assert stats["hybrid_retriever"]["rrf_k"] == 60


def test_fabric_remember_auto_indexes_graph():
    fabric = MemoryFabric()
    # Memory with clear named entities
    m1 = fabric.remember("Alice oversees the AlphaProject infrastructure")
    m2 = fabric.remember("Bob works with Alice on the AlphaProject")

    assert len(fabric.graph._nodes) >= 2
    # AlphaProject or Alice should be indexed
    assert "alice" in fabric.graph._nodes or "alphaproject" in fabric.graph._nodes

    # Traversal from Alice
    traversal = fabric.graph.traverse(["Alice"], max_hops=2)
    assert m1 in traversal.memory_ids or m2 in traversal.memory_ids


def test_fabric_remember_supersedes_graph():
    fabric = MemoryFabric()
    m1 = fabric.remember("Alice currently lives in Seattle Washington")
    m2 = fabric.remember("Alice moved to NewYork and lives there now", supersedes=m1)

    assert fabric.memory_os._entries[m1].superseded_by == m2
    assert fabric.memory_os._entries[m2].supersedes == m1

    # Verify that edges linked to m1 are marked superseded
    m1_edges = [fabric.graph._edges[idx] for idx in fabric.graph._memory_edges.get(m1, [])]
    for e in m1_edges:
        assert e.superseded is True


def test_fabric_recall_with_graph_and_hybrid():
    fabric = MemoryFabric()
    fabric.remember("Charlie is the lead architect for Kubernetes clusters")
    fabric.remember("Dana leads the PostgreSQL database migration")
    fabric.remember("Charlie and Dana collaborate on production resilience")

    recall_res = fabric.recall(
        "Charlie Kubernetes",
        traverse_graph=True,
        graph_seeds=["Charlie"],
        use_hybrid=True,
        hybrid_top_k=5,
    )

    assert recall_res.context is not None
    assert len(recall_res.graph_memories) > 0
    assert len(recall_res.hybrid_scored) > 0

    receipt = recall_res.receipt()
    assert "graph_memories" in receipt
    assert "hybrid_scored" in receipt


def test_fabric_write_gate_enforcement():
    fabric = MemoryFabric(enable_write_gate=True)
    # 1. High-information memory gets admitted
    m1 = fabric.remember("PostgreSQL cluster failover runbook for production infrastructure")
    assert m1.startswith("mem_")

    # 2. Near-duplicate memory returns existing or gets gated
    m2 = fabric.remember("PostgreSQL cluster failover runbook for production infrastructure")
    assert m2 == m1

    assert fabric.write_gate.receipt.admitted >= 1
