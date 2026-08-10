"""Deterministic, source-verified repository architecture evidence."""
from __future__ import annotations

import copy
import hashlib
import heapq
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Iterable, Mapping

from .models import RepositoryIndex

VERIFIED_ARCHITECTURE_SCHEMA_VERSION = "entroly.verified-architecture.v1"


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")).hexdigest()


def _verified_sources(
    root: Path,
    index: RepositoryIndex,
) -> tuple[set[str], dict[str, str], dict[str, int]]:
    verified: set[str] = set()
    sources: dict[str, str] = {}
    omissions: Counter[str] = Counter()
    for path, record in sorted(index.files.items()):
        try:
            candidate = (root / path).resolve(strict=True)
            candidate.relative_to(root)
            raw = candidate.read_bytes()
        except (OSError, RuntimeError, ValueError):
            omissions["unsafe-or-unreadable-source"] += 1
            continue
        digest = hashlib.sha256(raw).hexdigest()
        if digest != record.sha256:
            omissions["stale-source"] += 1
            continue
        verified.add(path)
        sources[path] = digest
    return verified, sources, dict(sorted(omissions.items()))


def _verified_dependencies(
    index: RepositoryIndex,
    verified: set[str],
) -> dict[str, tuple[str, ...]]:
    return {
        path: tuple(sorted(
            target
            for target in index.file_dependencies.get(path, ())
            if target in verified and target != path
        ))
        for path in sorted(verified)
    }


def _strong_components(
    graph: Mapping[str, Iterable[str]],
) -> list[tuple[str, ...]]:
    """Iterative Kosaraju decomposition; safe for very deep dependency chains."""
    reverse: dict[str, list[str]] = {node: [] for node in graph}
    for source, targets in graph.items():
        for target in targets:
            reverse[target].append(source)
    for targets in reverse.values():
        targets.sort()

    visited: set[str] = set()
    finish_order: list[str] = []
    for seed in sorted(graph):
        if seed in visited:
            continue
        visited.add(seed)
        stack: list[tuple[str, bool]] = [(seed, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                finish_order.append(node)
                continue
            stack.append((node, True))
            for target in reversed(sorted(graph.get(node, ()))):
                if target not in visited:
                    visited.add(target)
                    stack.append((target, False))

    components: list[tuple[str, ...]] = []
    visited.clear()
    for seed in reversed(finish_order):
        if seed in visited:
            continue
        visited.add(seed)
        members: list[str] = []
        stack = [(seed, False)]
        while stack:
            node, _expanded = stack.pop()
            members.append(node)
            for target in reversed(reverse[node]):
                if target not in visited:
                    visited.add(target)
                    stack.append((target, False))
        components.append(tuple(sorted(members)))
    components.sort(key=lambda members: (members[0], len(members), members))
    return components


def _shortest_path(
    graph: Mapping[str, Iterable[str]],
    start: str,
    target: str,
    *,
    allowed: set[str] | None = None,
    max_nodes: int = 100_000,
) -> list[str] | None:
    queue = deque([start])
    parent: dict[str, str | None] = {start: None}
    while queue and len(parent) <= max_nodes:
        node = queue.popleft()
        if node == target:
            path: list[str] = []
            current: str | None = target
            while current is not None:
                path.append(current)
                current = parent[current]
            return list(reversed(path))
        for neighbor in sorted(graph.get(node, ())):
            if allowed is not None and neighbor not in allowed:
                continue
            if neighbor not in parent:
                parent[neighbor] = node
                queue.append(neighbor)
    return None


def _cycle_witness(
    graph: Mapping[str, Iterable[str]],
    members: tuple[str, ...],
) -> list[str]:
    allowed = set(members)
    for source in members:
        for target in sorted(set(graph.get(source, ())) & allowed):
            path = _shortest_path(graph, target, source, allowed=allowed)
            if path is not None:
                return [source, *path]
    return []


def _condensation(
    dependencies: Mapping[str, tuple[str, ...]],
    components: list[tuple[str, ...]],
) -> tuple[
    list[dict[str, object]],
    dict[str, set[str]],
    dict[str, str],
    dict[str, int],
]:
    component_id_by_path: dict[str, str] = {}
    members_by_id: dict[str, tuple[str, ...]] = {}
    for members in components:
        component_id = "component:" + hashlib.sha256(
            "\n".join(members).encode("utf-8")
        ).hexdigest()[:20]
        members_by_id[component_id] = members
        for path in members:
            component_id_by_path[path] = component_id
    component_graph: dict[str, set[str]] = {
        component_id: set() for component_id in members_by_id
    }
    for source, targets in dependencies.items():
        source_id = component_id_by_path[source]
        for target in targets:
            target_id = component_id_by_path[target]
            if source_id != target_id:
                component_graph[source_id].add(target_id)

    inbound: Counter[str] = Counter()
    dependents: dict[str, set[str]] = defaultdict(set)
    for targets in component_graph.values():
        inbound.update(targets)
    for source, targets in component_graph.items():
        for target in targets:
            dependents[target].add(source)
    remaining = {
        component_id: len(targets)
        for component_id, targets in component_graph.items()
    }
    layers = {component_id: 0 for component_id in component_graph}
    ready = [
        component_id for component_id, count in remaining.items() if count == 0
    ]
    heapq.heapify(ready)
    while ready:
        component_id = heapq.heappop(ready)
        for dependent in sorted(dependents.get(component_id, ())):
            layers[dependent] = max(layers[dependent], layers[component_id] + 1)
            remaining[dependent] -= 1
            if remaining[dependent] == 0:
                heapq.heappush(ready, dependent)
    payload = [
        {
            "component_id": component_id,
            "members": list(members_by_id[component_id]),
            "member_count": len(members_by_id[component_id]),
            "layer": layers[component_id],
            "dependency_components": sorted(component_graph[component_id])[:50],
            "dependency_component_count": len(component_graph[component_id]),
            "dependency_components_omitted": max(
                0, len(component_graph[component_id]) - 50
            ),
            "dependent_component_count": inbound[component_id],
            "cyclic": len(members_by_id[component_id]) > 1,
        }
        for component_id in sorted(
            component_graph,
            key=lambda item: (-layers[item], members_by_id[item]),
        )
    ]
    return payload, component_graph, component_id_by_path, layers


def _undirected(
    dependencies: Mapping[str, tuple[str, ...]],
) -> dict[str, dict[str, float]]:
    graph: dict[str, dict[str, float]] = {path: {} for path in dependencies}
    for source, targets in dependencies.items():
        for target in targets:
            graph[source][target] = graph[source].get(target, 0.0) + 1.0
            graph[target][source] = graph[target].get(source, 0.0) + 1.0
    return graph


def _communities(
    dependencies: Mapping[str, tuple[str, ...]],
    *,
    resolution: float = 1.0,
    max_rounds: int = 50,
) -> list[dict[str, object]]:
    graph = _undirected(dependencies)
    labels = {node: node for node in graph}
    degrees = {node: sum(edges.values()) for node, edges in graph.items()}
    total_degree = sum(degrees.values())
    totals = dict(degrees)
    rounds = 0
    if total_degree:
        for rounds in range(1, max_rounds + 1):
            moved = False
            for node in sorted(graph):
                current = labels[node]
                degree = degrees[node]
                weights: Counter[str] = Counter()
                for neighbor, weight in graph[node].items():
                    weights[labels[neighbor]] += weight
                totals[current] -= degree
                candidates = sorted(set(weights) | {current})
                scores = {
                    candidate: (
                        weights[candidate]
                        - resolution * degree * totals.get(candidate, 0.0) / total_degree
                    )
                    for candidate in candidates
                }
                best = min(
                    candidates,
                    key=lambda candidate: (-scores[candidate], candidate),
                )
                labels[node] = best
                totals[best] = totals.get(best, 0.0) + degree
                if best != current:
                    moved = True
            if not moved:
                break

    raw_groups: dict[str, set[str]] = defaultdict(set)
    for node, label in labels.items():
        raw_groups[label].add(node)
    connected_groups: list[tuple[str, ...]] = []
    for members in raw_groups.values():
        remaining = set(members)
        while remaining:
            seed = min(remaining)
            queue = deque([seed])
            component = {seed}
            remaining.remove(seed)
            while queue:
                node = queue.popleft()
                for neighbor in sorted(graph[node]):
                    if neighbor in remaining and neighbor in members:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)
            connected_groups.append(tuple(sorted(component)))
    connected_groups.sort(key=lambda members: (-len(members), members))

    community_by_node = {
        node: position
        for position, members in enumerate(connected_groups)
        for node in members
    }
    result: list[dict[str, object]] = []
    for members in connected_groups:
        member_set = set(members)
        internal_weight = 0.0
        boundary_weight = 0.0
        margins: list[float] = []
        for node in members:
            by_community: Counter[int] = Counter()
            for neighbor, weight in graph[node].items():
                by_community[community_by_node[neighbor]] += weight
                if neighbor in member_set:
                    internal_weight += weight
                else:
                    boundary_weight += weight
            degree = degrees[node]
            if degree == 0:
                margins.append(1.0)
            else:
                own = by_community[community_by_node[node]]
                alternative = max(
                    (
                        weight
                        for community, weight in by_community.items()
                        if community != community_by_node[node]
                    ),
                    default=0.0,
                )
                margins.append((own - alternative) / degree)
        community_id = "community:" + hashlib.sha256(
            "\n".join(members).encode("utf-8")
        ).hexdigest()[:20]
        result.append({
            "community_id": community_id,
            "members": list(members),
            "member_count": len(members),
            "internal_weight": round(internal_weight / 2.0, 6),
            "boundary_weight": round(boundary_weight, 6),
            "mean_assignment_margin": round(sum(margins) / len(margins), 6),
            "minimum_assignment_margin": round(min(margins), 6),
            "identity": "content-derived-from-sorted-members",
            "confidence": "deterministic-structural-heuristic",
        })
    for item in result:
        item["algorithm_rounds"] = rounds
    return result


def _pagerank(
    graph: Mapping[str, tuple[str, ...]],
    *,
    damping: float = 0.85,
    rounds: int = 30,
) -> dict[str, float]:
    nodes = sorted(graph)
    if not nodes:
        return {}
    count = len(nodes)
    rank = {node: 1.0 / count for node in nodes}
    inbound: dict[str, list[str]] = defaultdict(list)
    for source, targets in graph.items():
        for target in targets:
            inbound[target].append(source)
    for _ in range(rounds):
        dangling = sum(rank[node] for node in nodes if not graph[node]) / count
        updated = {}
        for node in nodes:
            contribution = sum(
                rank[source] / len(graph[source])
                for source in inbound.get(node, ())
                if graph[source]
            )
            updated[node] = (1.0 - damping) / count + damping * (
                dangling + contribution
            )
        rank = updated
    return rank


def _sampled_betweenness(
    graph: Mapping[str, tuple[str, ...]],
    *,
    max_sources: int = 32,
) -> tuple[dict[str, float], list[str]]:
    nodes = sorted(graph)
    if not nodes:
        return {}, []
    sample_count = min(max_sources, len(nodes))
    positions = sorted({
        min(len(nodes) - 1, (position * len(nodes)) // sample_count)
        for position in range(sample_count)
    })
    sources = [nodes[position] for position in positions]
    score: Counter[str] = Counter()
    for source in sources:
        stack: list[str] = []
        predecessors: dict[str, list[str]] = defaultdict(list)
        paths = Counter({source: 1.0})
        distance = {source: 0}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            stack.append(node)
            for target in graph[node]:
                if target not in distance:
                    distance[target] = distance[node] + 1
                    queue.append(target)
                if distance[target] == distance[node] + 1:
                    paths[target] += paths[node]
                    predecessors[target].append(node)
        dependency: Counter[str] = Counter()
        while stack:
            node = stack.pop()
            for predecessor in predecessors[node]:
                if paths[node]:
                    dependency[predecessor] += (
                        paths[predecessor] / paths[node]
                    ) * (1.0 + dependency[node])
            if node != source:
                score[node] += dependency[node]
    scale = 1.0 / max(1, len(sources))
    return {node: score[node] * scale for node in nodes}, sources


def _hotspots(
    index: RepositoryIndex,
    dependencies: Mapping[str, tuple[str, ...]],
    *,
    limit: int,
) -> tuple[list[dict[str, object]], list[str]]:
    reverse: Counter[str] = Counter()
    for targets in dependencies.values():
        reverse.update(targets)
    rank = _pagerank(dependencies)
    betweenness, samples = _sampled_betweenness(dependencies)
    max_rank = max(rank.values(), default=1.0)
    max_between = max(betweenness.values(), default=1.0) or 1.0
    max_in = max(reverse.values(), default=1)
    max_out = max((len(targets) for targets in dependencies.values()), default=1)
    values: list[dict[str, object]] = []
    for path in dependencies:
        fan_in = reverse[path]
        fan_out = len(dependencies[path])
        score = (
            0.45 * rank[path] / max_rank
            + 0.25 * betweenness[path] / max_between
            + 0.15 * math.log1p(fan_in) / math.log1p(max_in)
            + (
                0.15 * math.log1p(fan_out) / math.log1p(max_out)
                if max_out
                else 0.0
            )
        )
        values.append({
            "path": path,
            "score": round(score, 8),
            "pagerank": round(rank[path], 10),
            "sampled_betweenness": round(betweenness[path], 8),
            "fan_in": fan_in,
            "fan_out": fan_out,
            "symbol_count": len(index.symbols_for_path(path)),
            "source_sha256": index.files[path].sha256,
        })
    values.sort(key=lambda item: (-float(item["score"]), str(item["path"])))
    return values[:limit], samples


def _routes(
    dependencies: Mapping[str, tuple[str, ...]],
    component_graph: Mapping[str, set[str]],
    component_by_path: Mapping[str, str],
    component_layers: Mapping[str, int],
    *,
    limit: int,
) -> list[dict[str, object]]:
    members_by_component: dict[str, list[str]] = defaultdict(list)
    for path, component in component_by_path.items():
        members_by_component[component].append(path)
    inbound: Counter[str] = Counter()
    for targets in component_graph.values():
        inbound.update(targets)
    route_lengths: dict[str, int] = {}
    next_component: dict[str, str | None] = {}
    for component in sorted(
        component_graph,
        key=lambda item: (component_layers[item], item),
    ):
        targets = sorted(component_graph[component])
        selected = min(
            targets,
            key=lambda target: (-route_lengths[target], target),
        ) if targets else None
        next_component[component] = selected
        route_lengths[component] = 1 + (
            route_lengths[selected] if selected is not None else 0
        )

    entry_components = sorted(
        (component for component in component_graph if inbound[component] == 0),
        key=lambda component: members_by_component[component],
    )
    routes: list[dict[str, object]] = []
    for entry in entry_components:
        components: list[str] = []
        current: str | None = entry
        while current is not None:
            components.append(current)
            current = next_component[current]
        evidence_edges: list[dict[str, str]] = []
        for source_component, target_component in zip(components, components[1:]):
            candidates = sorted(
                (source, target)
                for source in members_by_component[source_component]
                for target in dependencies[source]
                if component_by_path[target] == target_component
            )
            if candidates:
                source, target = candidates[0]
                evidence_edges.append({"source": source, "target": target})
        routes.append({
            "route_id": "route:" + _canonical_sha(list(components))[:20],
            "entry_component": entry,
            "component_path": components,
            "component_count": len(components),
            "entry_files": sorted(members_by_component[entry]),
            "foundation_files": sorted(members_by_component[components[-1]]),
            "evidence_edges": evidence_edges,
            "policy": "longest-condensation-dependency-route-lexical-tiebreak",
        })
    routes.sort(key=lambda item: (
        -int(item["component_count"]),
        item["entry_files"],
    ))
    return routes[:limit]


def _commit(payload: dict[str, object]) -> dict[str, object]:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    receipt = payload["receipt"]
    assert isinstance(receipt, dict)
    receipt["architecture_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def build_verified_architecture(
    root: Path,
    index: RepositoryIndex,
    *,
    index_digest: str,
    max_components: int = 5_000,
    max_communities: int = 1_000,
    max_cycles: int = 1_000,
    max_dependency_edges: int = 100_000,
    max_hotspots: int = 100,
    max_routes: int = 100,
) -> dict[str, object]:
    """Build a freshness-checked architecture model with explicit heuristics."""
    root = root.expanduser().resolve(strict=True)
    component_limit = max(1, min(int(max_components), 20_000))
    community_limit = max(1, min(int(max_communities), 10_000))
    cycle_limit = max(1, min(int(max_cycles), 10_000))
    dependency_edge_limit = max(
        1, min(int(max_dependency_edges), 1_000_000)
    )
    hotspot_limit = max(1, min(int(max_hotspots), 1_000))
    route_limit = max(1, min(int(max_routes), 1_000))
    verified, sources, omissions = _verified_sources(root, index)
    dependencies = _verified_dependencies(index, verified)
    components = _strong_components(dependencies)
    (
        component_payload,
        component_graph,
        component_by_path,
        component_layers,
    ) = _condensation(
        dependencies,
        components,
    )
    community_payload = _communities(dependencies)
    hotspots, betweenness_sources = _hotspots(
        index,
        dependencies,
        limit=hotspot_limit,
    )
    routes = _routes(
        dependencies,
        component_graph,
        component_by_path,
        component_layers,
        limit=route_limit,
    )
    all_cyclic_components = [members for members in components if len(members) > 1]
    cycles = [
        {
            "cycle_id": "cycle:" + hashlib.sha256(
                "\n".join(members).encode("utf-8")
            ).hexdigest()[:20],
            "members": list(members),
            "member_count": len(members),
            "witness_path": _cycle_witness(dependencies, members),
            "source_sha256": {path: sources[path] for path in members},
        }
        for members in all_cyclic_components[:cycle_limit]
    ]
    reverse: Counter[str] = Counter()
    for targets in dependencies.values():
        reverse.update(targets)
    component_inbound: Counter[str] = Counter()
    for targets in component_graph.values():
        component_inbound.update(targets)
    entry_component_count = sum(
        component_inbound[component] == 0 for component in component_graph
    )
    dependency_edge_count = sum(len(targets) for targets in dependencies.values())
    dependency_edges: list[dict[str, str]] = []
    for source, targets in dependencies.items():
        for target in targets:
            if len(dependency_edges) >= dependency_edge_limit:
                break
            dependency_edges.append({"source": source, "target": target})
        if len(dependency_edges) >= dependency_edge_limit:
            break
    payload: dict[str, object] = {
        "schema_version": VERIFIED_ARCHITECTURE_SCHEMA_VERSION,
        "index_digest": index_digest,
        "direction": "importer-to-dependency",
        "sources": sources,
        "dependency_edges": dependency_edges,
        "components": component_payload[:component_limit],
        "communities": community_payload[:community_limit],
        "cycles": cycles,
        "entrypoints": [
            path for path in sorted(dependencies) if reverse[path] == 0
        ],
        "foundations": [
            path for path, targets in dependencies.items() if not targets
        ],
        "routes": routes,
        "hotspots": hotspots,
        "policy": {
            "components": "exact-strongly-connected-condensation",
            "layers": "longest-distance-from-foundation-component",
            "communities": (
                "deterministic-modularity-local-moving-with-connected-refinement"
            ),
            "community_resolution": 1.0,
            "community_stability": "assignment-weight-margin-not-change-stability",
            "hotspot_score": (
                "0.45*pagerank+0.25*sampled-betweenness+"
                "0.15*log-fan-in+0.15*log-fan-out; each term max-normalized"
            ),
            "betweenness_sources": betweenness_sources,
            "route_selection": "longest-condensation-route-with-lexical-tiebreak",
            "component_adjacency_output_limit": 50,
        },
        "truncation": {
            "components_omitted": max(0, len(component_payload) - component_limit),
            "communities_omitted": max(0, len(community_payload) - community_limit),
            "cycles_omitted": max(0, len(all_cyclic_components) - cycle_limit),
            "dependency_edges_omitted": max(
                0, dependency_edge_count - dependency_edge_limit
            ),
            "hotspots_omitted": max(0, len(dependencies) - hotspot_limit),
            "routes_omitted": max(0, entry_component_count - route_limit),
        },
        "receipt": {
            "freshness": "verified-against-indexed-source-sha256",
            "verified_file_count": len(verified),
            "verified_dependency_edge_count": dependency_edge_count,
            "source_manifest_sha256": _canonical_sha(sources),
            "omissions_by_reason": omissions,
            "remote_calls": 0,
            "heuristic_findings_are_not_defect_proofs": True,
            "commitment_scope": (
                "payload-excluding-generation-command-and-architecture-sha256"
            ),
        },
    }
    return _commit(payload)


def verify_architecture_commitment(payload: Mapping[str, object]) -> bool:
    """Verify a detached architecture receipt without workspace access."""
    try:
        candidate = copy.deepcopy(dict(payload))
        candidate.pop("generation", None)
        candidate.pop("command", None)
        if candidate.get("schema_version") != VERIFIED_ARCHITECTURE_SCHEMA_VERSION:
            return False
        receipt = candidate["receipt"]
        if not isinstance(receipt, dict):
            return False
        expected = str(receipt.pop("architecture_sha256"))
        canonical = json.dumps(
            candidate,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest() == expected
    except (KeyError, TypeError, ValueError):
        return False


__all__ = [
    "VERIFIED_ARCHITECTURE_SCHEMA_VERSION",
    "build_verified_architecture",
    "verify_architecture_commitment",
]
