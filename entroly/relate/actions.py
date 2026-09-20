from __future__ import annotations

import re
import shlex
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedAction:
    raw: str
    scope: str
    resource: str = ""
    risk: str = "low"
    allowed_by_envelope: bool = False
    reasons: tuple[str, ...] = ()


_SECRET = re.compile(r"(^|/|\s)(\.env(?:\.[\w.-]+)?|id_rsa|id_ed25519|.*\.pem|.*\.key)(\s|$)")


def normalize_action(
    raw: str,
    *,
    allowed_capabilities: tuple[str, ...] = ("read", "execute:test"),
    network_approved: bool = False,
    destructive_approved: bool = False,
) -> NormalizedAction:
    reasons: list[str] = []
    try:
        parts = shlex.split(raw)
    except ValueError:
        return NormalizedAction(raw, "unknown", risk="high", reasons=("parse_error",))
    if not parts:
        return NormalizedAction(raw, "unknown", risk="low", reasons=("empty",))
    text = raw.lower()
    scope, resource, risk = "unknown", "", "medium"
    if "| sh" in text or "| bash" in text:
        return NormalizedAction(raw, "network:execute", risk="critical", reasons=("remote_pipe_to_shell",))
    if any(p in {"rm", "rmdir"} for p in parts) or "rm -rf" in text:
        return NormalizedAction(raw, "delete", risk="critical", reasons=("destructive_delete",))
    if parts[:2] == ["git", "push"]:
        return NormalizedAction(raw, "remote_write", risk="critical", reasons=("git_push", "network"))
    if parts[0] in {"curl", "wget"}:
        return NormalizedAction(raw, "network", risk="high", reasons=("network_transfer",))
    if _SECRET.search(raw):
        return NormalizedAction(raw, "read:secret", resource=raw, risk="critical", reasons=("protected_secret_path",))
    if parts[0] == "git" and len(parts) > 1 and parts[1] in {"status", "diff", "log"}:
        scope, risk = "read", "low"
    elif parts[0] in {"cat", "sed", "rg", "grep"}:
        scope, risk = "read", "low"
    elif parts[0] in {"npm", "pytest", "python"} and any(x in text for x in ["test", "pytest", "--noemit", "tsc"]):
        scope, risk = "execute:test", "medium"
    elif parts[0] == "chmod":
        scope, risk = "permission_change", "critical"
    allowed = (
        scope in allowed_capabilities
        and not (scope.startswith("network") and not network_approved)
        and not (scope in {"delete", "permission_change"} and not destructive_approved)
    )
    if not allowed:
        reasons.append("not_in_capability_envelope")
    return NormalizedAction(raw, scope, resource, risk, allowed, tuple(reasons))
