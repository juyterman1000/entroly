"""Reversible response contracts for supported agent integrations.

These contracts are instructions, not output truncators.  They never delete a
model response or change a provider's maximum-token setting.  Agent bundles may
read the active contract and follow it; receipts make that activation visible.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "entroly.response-contract.v1"
CONTRACTS: dict[str, dict[str, Any]] = {
    "off": {
        "description": "No Entroly response-shaping instruction.",
        "instruction": "",
    },
    "concise": {
        "description": "Lead with the outcome and omit routine narration.",
        "instruction": (
            "Lead with the result. Keep routine updates short, omit repeated context, "
            "and expand only when the user or task risk needs detail. Preserve errors, "
            "uncertainty, evidence, and required next actions."
        ),
    },
    "minimal": {
        "description": "Use the shortest complete answer for low-risk work.",
        "instruction": (
            "For low-risk routine work, answer in the shortest complete form. Never "
            "compress away failures, uncertainty, evidence boundaries, or user actions."
        ),
    },
    "evidence": {
        "description": "Prioritize receipts, verification, and explicit claim boundaries.",
        "instruction": (
            "Lead with the verified outcome. Distinguish measured usage from estimates, "
            "name failed gates and pass-throughs, retain recovery handles, and do not "
            "claim quality or savings without a matched baseline."
        ),
    },
}


def _state_root(scope: str) -> Path:
    if scope == "user":
        return Path.home() / ".entroly"
    if scope != "project":
        raise ValueError("scope must be 'project' or 'user'")
    from .config import _project_checkpoint_dir

    return _project_checkpoint_dir()


def contract_path(scope: str = "project") -> Path:
    return _state_root(scope) / "response-contract.json"


def _digest(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid response contract at {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported response contract at {path}")
    if value.get("name") not in CONTRACTS:
        raise ValueError(f"unknown response contract at {path}")
    return value


def load_contract(scope: str = "project", *, fall_back_to_user: bool = True) -> dict[str, Any]:
    path = contract_path(scope)
    value = _read(path)
    if value is None and scope == "project" and fall_back_to_user:
        value = _read(contract_path("user"))
        if value is not None:
            value = dict(value)
            value["resolved_scope"] = "user"
    if value is None:
        value = {
            "schema_version": SCHEMA_VERSION,
            "name": "off",
            "description": CONTRACTS["off"]["description"],
            "instruction": "",
            "scope": scope,
            "resolved_scope": "default",
        }
    return value


def set_contract(name: str, *, scope: str = "project") -> dict[str, Any]:
    if name not in CONTRACTS:
        raise ValueError(f"unknown response contract {name!r}; choose from {', '.join(CONTRACTS)}")
    path = contract_path(scope)
    previous = _read(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if path.exists():
        backup = path.with_name(
            f"{path.name}.backup-{time.strftime('%Y%m%d%H%M%S')}-{time.time_ns()}"
        )
        shutil.copy2(path, backup)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "description": CONTRACTS[name]["description"],
        "instruction": CONTRACTS[name]["instruction"],
        "scope": scope,
        "updated_at_unix": int(time.time()),
    }
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        if os.name != "nt":
            temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return {
        "schema_version": "entroly.response-contract-change.v1",
        "action": "disable" if name == "off" else "set",
        "scope": scope,
        "name": name,
        "path": str(path),
        "backup": str(backup) if backup else None,
        "previous_digest": _digest(previous),
        "new_digest": _digest(payload),
        "reversible": True,
        "claim_boundary": "This changes agent instructions only; it is not measured token savings.",
    }


def environment_contract() -> dict[str, str]:
    """Return a minimal environment pointer for wrapped CLI agents."""
    project = contract_path("project")
    user = contract_path("user")
    selected = project if project.exists() else user if user.exists() else None
    return {"ENTROLY_RESPONSE_CONTRACT": str(selected)} if selected else {}


__all__ = [
    "CONTRACTS",
    "SCHEMA_VERSION",
    "contract_path",
    "environment_contract",
    "load_contract",
    "set_contract",
]
