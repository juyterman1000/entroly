"""Local transport for compact Work Graph handoffs between agent processes."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from .work_graph_review import _graph_source_state, _handoff_graph, _observation, _source_state
from .work_graph_store import WorkGraphStore

_MAX_HANDOFF_BYTES = 256 * 1024
_MAX_INBOX_ITEMS = 128


def _inbox(store: WorkGraphStore) -> Path:
    folder = store.repo_dir / "handoffs"
    folder.mkdir(mode=0o700, exist_ok=True)
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("unsafe Work Graph handoff inbox")
    return folder


def save_handoff(
    store: WorkGraphStore, handoff: dict[str, Any], continuation_proof: dict[str, Any]
) -> str:
    """Persist a bounded receipt for another local host; no transcript is stored."""
    if not store.load().verify_handoff(handoff):
        raise ValueError("cannot save a handoff that does not match the current graph")
    payload = {
        "schema_version": 1,
        "handoff": handoff,
        "continuation_proof": continuation_proof,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) > _MAX_HANDOFF_BYTES:
        raise ValueError("handoff exceeds local inbox size limit")
    name = hashlib.sha256(str(handoff["payload_commitment"]).encode("utf-8")).hexdigest()
    folder = _inbox(store)
    target = folder / f"{name}.json"
    temporary = folder / f".{name}.{uuid.uuid4().hex}.tmp"
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as out:
            out.write(encoded)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return f"handoff:{name}"


def pending_handoffs(*, root: Path, store: WorkGraphStore, to_agent: str) -> dict[str, Any]:
    """Return current receipts addressed to one agent and mark stale ones."""
    if not to_agent or len(to_agent) > 512 or "\x00" in to_agent:
        raise ValueError("to_agent must be a bounded agent identifier")
    folder = _inbox(store)
    files = sorted(
        (file for file in folder.glob("*.json") if not file.is_symlink() and file.is_file()),
        key=lambda file: file.stat().st_mtime_ns, reverse=True,
    )
    graph = store.load()
    current = _source_state(store.repo_id, _observation(root))
    entries = []
    for file in files[:_MAX_INBOX_ITEMS]:
        if file.is_symlink() or not file.is_file() or file.stat().st_size > _MAX_HANDOFF_BYTES:
            continue
        try:
            payload = json.loads(file.read_text(encoding="utf-8"))
            handoff = payload["handoff"]
            if payload.get("schema_version") != 1 or handoff.get("to_agent") != to_agent:
                continue
            try:
                sealed = _handoff_graph(graph, handoff)
                valid = current == _graph_source_state(sealed, store.repo_id)
            except ValueError:
                valid = False
            entries.append({
                "ref": "handoff:" + file.stem,
                "current": valid,
                "handoff": handoff,
                "continuation_proof": payload["continuation_proof"],
            })
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return {"status": "ok", "kind": "work_pending_handoffs", "to_agent": to_agent, "handoffs": entries}
