"""A recovering agent must acknowledge the trust level before it may act.

``work_resume`` already labels what it returns ``untrusted_recovered_work_state``
and records ``unknown:previous-agent-intent``. Labelling is not a control: an
agent can read the label and immediately claim work as though the reconstructed
state were observed fact. The label describes the risk; nothing made anyone
carry it.

This is the control. Recovery arms a gate; mutating operations refuse while the
gate is armed; acknowledging it disarms it. The acknowledgement is not a
formality -- it is the moment responsibility transfers from "Entroly inferred
this" to "the agent accepted this", and it is recorded.

The token is a digest of the recovered state itself, which gives two properties
worth more than a random nonce:

- acknowledging state A does not authorise acting on state B. If the worktree
  moves between resume and acknowledgement, the token no longer matches and the
  agent is sent back to look again.
- re-resuming unchanged state yields the same token, so a retry or a duplicate
  resume does not demand a second acknowledgement for the same facts.

Fail-closed. If the gate cannot be persisted, recovery fails rather than
returning reconstructed state that nothing can hold anyone to. A gate that
silently stops working is worse than no gate, because the label keeps implying
a control that is no longer there.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

_MARKER_NAME = "pending-recovery-ack.json"
_TRUST_LABEL = "untrusted_recovered_work_state"


class RecoveryAcknowledgementRequired(RuntimeError):
    """Raised when a mutating operation runs while a recovery is unacknowledged."""

    def __init__(self, token: str) -> None:
        super().__init__(
            "recovered work state has not been acknowledged; call "
            "work_acknowledge_recovery with the token from work_resume before "
            "claiming or mutating work"
        )
        self.token = token


def recovery_token(resume_view: Any) -> str:
    """Deterministic identity for one recovered state.

    Canonical JSON so that key order, which carries no meaning, cannot make two
    identical recoveries look different and demand a second acknowledgement.
    """
    raw = json.dumps(
        resume_view, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(raw.encode("utf-8", errors="surrogatepass")).hexdigest()
    return f"recovery:{digest[:32]}"


def _marker_path(state_root: str | os.PathLike[str]) -> Path:
    return Path(state_root) / _MARKER_NAME


def arm(state_root: str | os.PathLike[str], token: str, unknowns: list[str]) -> dict[str, Any]:
    """Record that recovered state is outstanding, and describe what is unknown.

    Written atomically: a half-written marker would be unparseable, and a
    gate that cannot be read is treated as armed, so a torn write would wedge
    the repository rather than merely losing a gate.
    """
    root = Path(state_root)
    root.mkdir(parents=True, exist_ok=True)
    record = {"token": token, "trust": _TRUST_LABEL, "unknowns": sorted(set(unknowns))}
    payload = json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8")

    handle, temp_name = tempfile.mkstemp(dir=str(root), prefix=".ack-", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, _marker_path(root))
    except OSError:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        # Deliberately not swallowed. See the module docstring: recovery that
        # cannot arm its gate must fail, not proceed ungated.
        raise
    return dict(record, required=True)


def pending(state_root: str | os.PathLike[str]) -> dict[str, Any] | None:
    """The outstanding acknowledgement, if any.

    An unreadable or malformed marker counts as armed. The alternative is to
    treat corruption as consent.
    """
    path = _marker_path(state_root)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return {"token": "", "trust": _TRUST_LABEL, "unknowns": ["unknown:gate-unreadable"]}
    try:
        record = json.loads(raw)
    except ValueError:
        return {"token": "", "trust": _TRUST_LABEL, "unknowns": ["unknown:gate-corrupt"]}
    if not isinstance(record, dict) or not isinstance(record.get("token"), str):
        return {"token": "", "trust": _TRUST_LABEL, "unknowns": ["unknown:gate-malformed"]}
    return record


def require_acknowledged(state_root: str | os.PathLike[str]) -> None:
    """Refuse the caller while recovered state is outstanding."""
    outstanding = pending(state_root)
    if outstanding is not None:
        raise RecoveryAcknowledgementRequired(str(outstanding.get("token", "")))


def acknowledge(state_root: str | os.PathLike[str], token: str) -> dict[str, Any]:
    """Accept responsibility for one specific recovered state.

    Refuses a token that does not match the outstanding one, so acknowledging
    stale state cannot authorise acting on current state.
    """
    outstanding = pending(state_root)
    if outstanding is None:
        return {"acknowledged": True, "already_clear": True}
    expected = str(outstanding.get("token", ""))
    supplied = str(token).strip()
    if not expected or supplied != expected:
        raise ValueError(
            "acknowledgement token does not match the outstanding recovered "
            "state; call work_resume again and acknowledge what it returns"
        )
    try:
        _marker_path(state_root).unlink()
    except FileNotFoundError:
        pass
    return {
        "acknowledged": True,
        "token": expected,
        "trust": _TRUST_LABEL,
        "unknowns": outstanding.get("unknowns", []),
    }
