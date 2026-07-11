from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from entroly.models.registry import get_model_registry


ROOT = Path(__file__).resolve().parent.parent
PYTHON_SNAPSHOT = ROOT / "entroly" / "models" / "registry.json"
RUST_SNAPSHOT = ROOT / "entroly-core" / "model_registry.json"
SYNC_SCRIPT = ROOT / "scripts" / "sync_model_registry.py"


def test_python_and_rust_registry_snapshots_are_byte_identical():
    python_bytes = PYTHON_SNAPSHOT.read_bytes()
    rust_bytes = RUST_SNAPSHOT.read_bytes()

    assert rust_bytes == python_bytes
    assert hashlib.sha256(rust_bytes).hexdigest() == get_model_registry().base_registry_digest


def test_registry_snapshot_has_unique_model_ids():
    payload = json.loads(PYTHON_SNAPSHOT.read_text(encoding="utf-8"))
    ids = [model["id"] for model in payload["models"]]

    assert ids
    assert len(ids) == len(set(ids))


def test_sync_tool_check_mode_accepts_committed_snapshot():
    spec = importlib.util.spec_from_file_location("sync_model_registry", SYNC_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.synchronize(check=True) == 0
