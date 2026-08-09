from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from entroly.repository_intelligence import build_repository_index
from entroly.repository_intelligence.models import FileRecord, RepositoryIndex
from entroly.repository_intelligence.verified_architecture import (
    build_verified_architecture,
    verify_architecture_commitment,
)


def _write(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _project(root: Path) -> RepositoryIndex:
    _write(root, "app.py", "import service\nservice.execute()\n")
    _write(root, "api.py", "import service\nservice.execute()\n")
    _write(
        root,
        "service.py",
        "import data\ndef execute():\n    return data.load()\n",
    )
    _write(
        root,
        "data.py",
        "import service\ndef load():\n    return service.__name__\n",
    )
    _write(root, "isolated.py", "VALUE = 1\n")
    return build_repository_index(root)


def test_architecture_has_exact_layers_cycles_routes_and_stable_communities(
    tmp_path: Path,
) -> None:
    index = _project(tmp_path)
    payload = build_verified_architecture(
        tmp_path,
        index,
        index_digest="sha256:test",
    )

    assert payload["direction"] == "importer-to-dependency"
    assert payload["receipt"]["verified_file_count"] == 5
    assert payload["receipt"]["remote_calls"] == 0
    assert verify_architecture_commitment(payload)

    cycles = payload["cycles"]
    assert len(cycles) == 1
    assert cycles[0]["members"] == ["data.py", "service.py"]
    witness = cycles[0]["witness_path"]
    assert witness[0] == witness[-1]
    assert set(witness) == {"data.py", "service.py"}

    components = {
        tuple(item["members"]): item for item in payload["components"]
    }
    assert components[("data.py", "service.py")]["layer"] == 0
    assert components[("app.py",)]["layer"] == 1
    assert components[("api.py",)]["layer"] == 1
    assert components[("isolated.py",)]["layer"] == 0
    assert {"app.py", "api.py", "isolated.py"} <= set(payload["entrypoints"])

    app_route = next(
        route for route in payload["routes"] if route["entry_files"] == ["app.py"]
    )
    assert app_route["component_count"] == 2
    assert app_route["evidence_edges"] == [
        {"source": "app.py", "target": "service.py"},
    ]
    assert all(
        community["identity"] == "content-derived-from-sorted-members"
        for community in payload["communities"]
    )


def test_architecture_omits_stale_sources_and_receipt_detects_tampering(
    tmp_path: Path,
) -> None:
    index = _project(tmp_path)
    (tmp_path / "data.py").write_text("VALUE = 2\n", encoding="utf-8")
    payload = build_verified_architecture(
        tmp_path,
        index,
        index_digest="sha256:test",
    )
    assert payload["receipt"]["omissions_by_reason"] == {"stale-source": 1}
    assert "data.py" not in payload["sources"]
    assert all(
        edge["source"] != "data.py" and edge["target"] != "data.py"
        for edge in payload["dependency_edges"]
    )
    assert verify_architecture_commitment(payload)
    tampered = copy.deepcopy(payload)
    tampered["entrypoints"].append("invented.py")
    assert not verify_architecture_commitment(tampered)


def test_architecture_is_root_independent_for_identical_sources(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = build_verified_architecture(
        first_root,
        _project(first_root),
        index_digest="sha256:same",
    )
    second = build_verified_architecture(
        second_root,
        _project(second_root),
        index_digest="sha256:same",
    )
    assert first == second


def test_architecture_handles_dependency_chains_beyond_python_recursion_limit(
    tmp_path: Path,
) -> None:
    file_count = 1_100
    files: dict[str, FileRecord] = {}
    dependencies: dict[str, tuple[str, ...]] = {}
    raw = b"# verified\n"
    digest = hashlib.sha256(raw).hexdigest()
    for position in range(file_count):
        path = f"src/f{position:04d}.py"
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        files[path] = FileRecord(
            path=path,
            language="python",
            sha256=digest,
            byte_length=len(raw),
            line_count=1,
            is_test=False,
        )
        dependencies[path] = (
            (f"src/f{position + 1:04d}.py",)
            if position + 1 < file_count
            else ()
        )
    index = RepositoryIndex(
        root=str(tmp_path),
        files=files,
        file_dependencies=dependencies,
    )
    payload = build_verified_architecture(
        tmp_path,
        index,
        index_digest="sha256:deep",
        max_hotspots=10,
        max_routes=2,
        max_communities=20,
    )
    layers = {
        item["members"][0]: item["layer"] for item in payload["components"]
    }
    assert "src/f0000.py" in layers, (
        len(payload["sources"]),
        payload["receipt"]["omissions_by_reason"],
        len(payload["components"]),
        list(layers)[:3],
    )
    assert layers["src/f0000.py"] == file_count - 1
    assert layers[f"src/f{file_count - 1:04d}.py"] == 0
    assert payload["routes"][0]["component_count"] == file_count
    assert verify_architecture_commitment(payload)


def test_architecture_bounds_serialized_edges_without_changing_full_graph_counts(
    tmp_path: Path,
) -> None:
    index = _project(tmp_path)
    payload = build_verified_architecture(
        tmp_path,
        index,
        index_digest="sha256:bounded",
        max_dependency_edges=1,
    )
    assert len(payload["dependency_edges"]) == 1
    assert payload["receipt"]["verified_dependency_edge_count"] == 4
    assert payload["truncation"]["dependency_edges_omitted"] == 3
    assert verify_architecture_commitment(payload)
