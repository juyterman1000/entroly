"""Receipt-verified comparison of two repository architecture snapshots."""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Mapping

from .verified_architecture import (
    VERIFIED_ARCHITECTURE_SCHEMA_VERSION,
    verify_architecture_commitment,
)

ARCHITECTURE_DIFF_SCHEMA_VERSION = "entroly.verified-architecture-diff.v1"


def _list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _architecture_sha(payload: Mapping[str, object]) -> str:
    receipt = payload.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("architecture receipt is missing")
    value = receipt.get("architecture_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("architecture receipt commitment is invalid")
    return value


def _sources(payload: Mapping[str, object]) -> dict[str, str]:
    value = payload.get("sources")
    if not isinstance(value, Mapping):
        return {}
    return {
        str(path): str(digest)
        for path, digest in value.items()
        if isinstance(path, str) and isinstance(digest, str)
    }


def _edges(payload: Mapping[str, object]) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for raw in _list(payload.get("dependency_edges")):
        if not isinstance(raw, Mapping):
            continue
        source = raw.get("source")
        target = raw.get("target")
        if isinstance(source, str) and isinstance(target, str):
            result.add((source, target))
    return result


def _cycles(payload: Mapping[str, object]) -> dict[tuple[str, ...], str]:
    result: dict[tuple[str, ...], str] = {}
    for raw in _list(payload.get("cycles")):
        if not isinstance(raw, Mapping):
            continue
        members = raw.get("members")
        cycle_id = raw.get("cycle_id")
        if isinstance(members, list) and all(isinstance(item, str) for item in members):
            result[tuple(sorted(members))] = str(cycle_id)
    return result


def _layers(payload: Mapping[str, object]) -> dict[str, int]:
    result: dict[str, int] = {}
    for raw in _list(payload.get("components")):
        if not isinstance(raw, Mapping):
            continue
        members = raw.get("members")
        try:
            layer = int(raw.get("layer", -1))
        except (TypeError, ValueError):
            continue
        if layer < 0 or not isinstance(members, list):
            continue
        for member in members:
            if isinstance(member, str):
                result[member] = layer
    return result


def _communities(payload: Mapping[str, object]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for raw in _list(payload.get("communities")):
        if not isinstance(raw, Mapping):
            continue
        community_id = raw.get("community_id")
        members = raw.get("members")
        if not isinstance(community_id, str) or not isinstance(members, list):
            continue
        clean_members = sorted(item for item in members if isinstance(item, str))
        result[community_id] = {
            "members": clean_members,
            "mean_assignment_margin": raw.get("mean_assignment_margin"),
        }
    return result


def _community_changes(
    before: Mapping[str, object],
    after: Mapping[str, object],
    *,
    limit: int,
) -> tuple[list[dict[str, object]], int]:
    old = _communities(before)
    new = _communities(after)
    candidates: list[tuple[float, str, str]] = []
    for old_id, old_value in old.items():
        old_members = set(old_value["members"])
        for new_id, new_value in new.items():
            new_members = set(new_value["members"])
            union = old_members | new_members
            overlap = len(old_members & new_members) / len(union) if union else 1.0
            if overlap:
                candidates.append((overlap, old_id, new_id))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    old_used: set[str] = set()
    new_used: set[str] = set()
    matches: list[tuple[float, str, str]] = []
    for overlap, old_id, new_id in candidates:
        if old_id in old_used or new_id in new_used:
            continue
        old_used.add(old_id)
        new_used.add(new_id)
        matches.append((overlap, old_id, new_id))

    changes: list[dict[str, object]] = []
    for overlap, old_id, new_id in matches:
        old_members = set(old[old_id]["members"])
        new_members = set(new[new_id]["members"])
        changes.append({
            "before_community_id": old_id,
            "after_community_id": new_id,
            "status": "unchanged" if old_id == new_id else "membership-changed",
            "jaccard_overlap": round(overlap, 8),
            "added_members": sorted(new_members - old_members),
            "removed_members": sorted(old_members - new_members),
            "before_assignment_margin": old[old_id]["mean_assignment_margin"],
            "after_assignment_margin": new[new_id]["mean_assignment_margin"],
        })
    for old_id in sorted(set(old) - old_used):
        changes.append({
            "before_community_id": old_id,
            "after_community_id": None,
            "status": "dissolved-or-truncated",
            "jaccard_overlap": 0.0,
            "added_members": [],
            "removed_members": old[old_id]["members"],
            "before_assignment_margin": old[old_id]["mean_assignment_margin"],
            "after_assignment_margin": None,
        })
    for new_id in sorted(set(new) - new_used):
        changes.append({
            "before_community_id": None,
            "after_community_id": new_id,
            "status": "created-or-previously-truncated",
            "jaccard_overlap": 0.0,
            "added_members": new[new_id]["members"],
            "removed_members": [],
            "before_assignment_margin": None,
            "after_assignment_margin": new[new_id]["mean_assignment_margin"],
        })
    changes = [item for item in changes if item["status"] != "unchanged"]
    changes.sort(key=lambda item: (
        -float(item["jaccard_overlap"]),
        str(item["before_community_id"]),
        str(item["after_community_id"]),
    ))
    return changes[:limit], max(0, len(changes) - limit)


def _hotspot_ranks(payload: Mapping[str, object]) -> dict[str, int]:
    result: dict[str, int] = {}
    for position, raw in enumerate(_list(payload.get("hotspots")), start=1):
        if isinstance(raw, Mapping) and isinstance(raw.get("path"), str):
            result[str(raw["path"])] = position
    return result


def _route_ids(payload: Mapping[str, object]) -> set[str]:
    return {
        str(raw["route_id"])
        for raw in _list(payload.get("routes"))
        if isinstance(raw, Mapping) and isinstance(raw.get("route_id"), str)
    }


def _commit(payload: dict[str, object]) -> dict[str, object]:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    receipt = payload["receipt"]
    assert isinstance(receipt, dict)
    receipt["architecture_diff_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def build_verified_architecture_diff(
    before: Mapping[str, object],
    after: Mapping[str, object],
    *,
    limit: int = 5_000,
) -> dict[str, object]:
    """Compare two committed architecture payloads without trusting labels."""
    if before.get("schema_version") != VERIFIED_ARCHITECTURE_SCHEMA_VERSION:
        raise ValueError("before architecture schema is invalid")
    if after.get("schema_version") != VERIFIED_ARCHITECTURE_SCHEMA_VERSION:
        raise ValueError("after architecture schema is invalid")
    if not verify_architecture_commitment(before):
        raise ValueError("before architecture commitment is invalid")
    if not verify_architecture_commitment(after):
        raise ValueError("after architecture commitment is invalid")
    change_limit = max(1, min(int(limit), 50_000))
    old_sources = _sources(before)
    new_sources = _sources(after)
    added_files = sorted(set(new_sources) - set(old_sources))
    removed_files = sorted(set(old_sources) - set(new_sources))
    modified_files = sorted(
        path
        for path in set(old_sources) & set(new_sources)
        if old_sources[path] != new_sources[path]
    )
    old_edges = _edges(before)
    new_edges = _edges(after)
    added_edges = sorted(new_edges - old_edges)
    removed_edges = sorted(old_edges - new_edges)
    old_cycles = _cycles(before)
    new_cycles = _cycles(after)
    introduced_cycles = sorted(set(new_cycles) - set(old_cycles))
    resolved_cycles = sorted(set(old_cycles) - set(new_cycles))
    old_layers = _layers(before)
    new_layers = _layers(after)
    layer_moves = sorted(
        (
            {
                "path": path,
                "before_layer": old_layers[path],
                "after_layer": new_layers[path],
                "delta": new_layers[path] - old_layers[path],
            }
            for path in set(old_layers) & set(new_layers)
            if old_layers[path] != new_layers[path]
        ),
        key=lambda item: (-abs(int(item["delta"])), str(item["path"])),
    )
    community_changes, community_omitted = _community_changes(
        before, after, limit=change_limit
    )
    old_ranks = _hotspot_ranks(before)
    new_ranks = _hotspot_ranks(after)
    hotspot_moves = sorted(
        (
            {
                "path": path,
                "before_rank": old_ranks[path],
                "after_rank": new_ranks[path],
                "improvement": old_ranks[path] - new_ranks[path],
            }
            for path in set(old_ranks) & set(new_ranks)
            if old_ranks[path] != new_ranks[path]
        ),
        key=lambda item: (-abs(int(item["improvement"])), str(item["path"])),
    )
    old_routes = _route_ids(before)
    new_routes = _route_ids(after)
    before_truncation = (
        dict(before["truncation"])
        if isinstance(before.get("truncation"), Mapping)
        else {}
    )
    after_truncation = (
        dict(after["truncation"])
        if isinstance(after.get("truncation"), Mapping)
        else {}
    )
    input_truncated = any(
        int(value) > 0
        for value in (*before_truncation.values(), *after_truncation.values())
        if isinstance(value, (int, float))
    )
    counts = {
        "files_added": len(added_files),
        "files_removed": len(removed_files),
        "files_modified": len(modified_files),
        "dependency_edges_added": len(added_edges),
        "dependency_edges_removed": len(removed_edges),
        "cycles_introduced": len(introduced_cycles),
        "cycles_resolved": len(resolved_cycles),
        "layer_moves": len(layer_moves),
        "community_changes": len(community_changes) + community_omitted,
        "hotspot_rank_moves": len(hotspot_moves),
        "routes_added": len(new_routes - old_routes),
        "routes_removed": len(old_routes - new_routes),
    }
    payload: dict[str, object] = {
        "schema_version": ARCHITECTURE_DIFF_SCHEMA_VERSION,
        "before_architecture_sha256": _architecture_sha(before),
        "after_architecture_sha256": _architecture_sha(after),
        "input_truncation": {
            "before": before_truncation,
            "after": after_truncation,
            "absence_inconclusive": input_truncated,
        },
        "counts": counts,
        "files": {
            "added": added_files[:change_limit],
            "removed": removed_files[:change_limit],
            "modified": modified_files[:change_limit],
        },
        "dependency_edges": {
            "added": [
                {"source": source, "target": target}
                for source, target in added_edges[:change_limit]
            ],
            "removed": [
                {"source": source, "target": target}
                for source, target in removed_edges[:change_limit]
            ],
        },
        "cycles": {
            "introduced": [
                {"members": list(members), "cycle_id": new_cycles[members]}
                for members in introduced_cycles[:change_limit]
            ],
            "resolved": [
                {"members": list(members), "cycle_id": old_cycles[members]}
                for members in resolved_cycles[:change_limit]
            ],
        },
        "layer_moves": layer_moves[:change_limit],
        "community_changes": community_changes,
        "hotspot_rank_moves": hotspot_moves[:change_limit],
        "routes": {
            "added": sorted(new_routes - old_routes)[:change_limit],
            "removed": sorted(old_routes - new_routes)[:change_limit],
        },
        "truncation": {
            "files_added_omitted": max(0, len(added_files) - change_limit),
            "files_removed_omitted": max(0, len(removed_files) - change_limit),
            "files_modified_omitted": max(0, len(modified_files) - change_limit),
            "dependency_edges_added_omitted": max(0, len(added_edges) - change_limit),
            "dependency_edges_removed_omitted": max(0, len(removed_edges) - change_limit),
            "cycles_introduced_omitted": max(0, len(introduced_cycles) - change_limit),
            "cycles_resolved_omitted": max(0, len(resolved_cycles) - change_limit),
            "layer_moves_omitted": max(0, len(layer_moves) - change_limit),
            "community_changes_omitted": community_omitted,
            "hotspot_rank_moves_omitted": max(0, len(hotspot_moves) - change_limit),
            "routes_added_omitted": max(0, len(new_routes - old_routes) - change_limit),
            "routes_removed_omitted": max(0, len(old_routes - new_routes) - change_limit),
        },
        "receipt": {
            "input_commitments_verified": True,
            "community_matching": "greedy-maximum-jaccard-with-lexical-tiebreak",
            "truncated_input_warning": (
                "input architecture truncation can make absence inconclusive"
            ),
            "remote_calls": 0,
            "commitment_scope": (
                "payload-excluding-generation-command-and-architecture-diff-sha256"
            ),
        },
    }
    return _commit(payload)


def verify_architecture_diff_commitment(payload: Mapping[str, object]) -> bool:
    """Verify a detached architecture-diff receipt."""
    try:
        candidate = copy.deepcopy(dict(payload))
        candidate.pop("generation", None)
        candidate.pop("command", None)
        if candidate.get("schema_version") != ARCHITECTURE_DIFF_SCHEMA_VERSION:
            return False
        receipt = candidate["receipt"]
        if not isinstance(receipt, dict):
            return False
        expected = str(receipt.pop("architecture_diff_sha256"))
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
    "ARCHITECTURE_DIFF_SCHEMA_VERSION",
    "build_verified_architecture_diff",
    "verify_architecture_diff_commitment",
]
