from pathlib import Path

from entroly.repo_map import build_repo_map


def test_repo_map_covers_three_repos():
    root = Path(__file__).resolve().parents[1]
    grouped = build_repo_map(root)

    python_paths = {entry.path for entry in grouped["python"]}
    rust_paths = {entry.path for entry in grouped["rust-core"]}
    wasm_paths = {entry.path for entry in grouped["wasm"]}

    assert "entroly/server.py" in python_paths
    assert "entroly-core/src/cogops.rs" in rust_paths
    assert "entroly-wasm/js/server.js" in wasm_paths


def test_repo_map_includes_new_sync_and_map_modules():
    root = Path(__file__).resolve().parents[1]
    grouped = build_repo_map(root)
    python_entries = {entry.path: entry for entry in grouped["python"]}

    assert python_entries["entroly/change_listener.py"].category == "python-cogops"
    assert python_entries["entroly/repo_map.py"].category == "python-cogops"
