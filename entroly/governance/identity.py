"""
Governance Identity — Agent identity creation, verification, and resolution.
=============================================================================

Provides the identity plane for the enterprise control plane:
  - HMAC-SHA256 signed identity tokens via Rust engine (entroly_core)
  - Identity resolution from headers, environment, or explicit creation
  - Identity verification against operator key
  - Ephemeral/time-limited credentials
  - Organization and user delegation chain

Token computation is performed by the Rust engine to guarantee identical
results across Python, Node/WASM, and native distributions.  Falls back to
a pure-Python HMAC implementation when the native build is unavailable
(development environments without `maturin develop`).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import time
from typing import Any, Mapping, Sequence

from ..native_status import usable_core
from .domain import AgentIdentity, _new_id, _now
from .events import (
    GovernanceEventBus,
    IdentityResolvedEvent,
    TracingContext,
    get_event_bus,
)

logger = logging.getLogger(__name__)

# ── Rust Engine Integration ───────────────────────────────────────────────────

# Routed through the shared capability gate, not a bare `import entroly_core`.
#
# A direct import decides on its own whether the native core is usable, so this
# module would call into a core the rest of the process has already refused --
# below the declared minimum version, or missing symbols other modules need.
# `usable_core()` is the single answer to "may this process call into
# entroly_core?", and a mixed process is worse than either pure mode: that is
# the split that produced `ContextFragment.__new__() got an unexpected keyword
# argument 'recency_score'`.
#
# The symbols are then fetched individually. They are new to this release, so a
# core that is otherwise perfectly usable can still lack them; absence must fall
# back rather than raise at first call.
_core = usable_core()
_rust_compute_token = getattr(_core, "governance_compute_identity_token", None)
_rust_verify_token = getattr(_core, "governance_verify_identity_token", None)
_RUST_AVAILABLE = _rust_compute_token is not None and _rust_verify_token is not None

if _RUST_AVAILABLE:
    logger.debug("governance.identity: using Rust engine for token computation")
else:
    logger.debug(
        "governance.identity: native identity tokens unavailable — using Python "
        "HMAC fallback. Run `maturin develop --release` in entroly-core/ to "
        "enable native tokens."
    )


IDENTITY_SCHEMA_VERSION = "entroly.governance.identity.v1"
_IDENTITY_ENV = "ENTROLY_AGENT_IDENTITY"
_IDENTITY_KEY_ENV = "ENTROLY_IDENTITY_KEY"
_IDENTITY_TTL_ENV = "ENTROLY_IDENTITY_TTL_SECONDS"
_DEFAULT_TTL = 86400  # 24 hours

_SCOPE_RE = re.compile(
    r"^(read|write|execute|deploy|admin|review|approve)"
    r"(?::([A-Za-z0-9_./*-]+))?$"
)

KNOWN_AGENT_TYPES = frozenset({
    "claude-code", "codex", "cursor", "windsurf", "aider", "cline",
    "copilot", "hermes", "openclaw", "opencode", "human", "service",
    "ci-bot", "unknown",
})


# ── Exceptions ───────────────────────────────────────────────────────

class IdentityError(RuntimeError):
    """Raised when agent identity cannot be established or verified."""


class IdentityExpiredError(IdentityError):
    """Raised when an identity token has expired."""


# ── Identity Token Operations ────────────────────────────────────────

def _identity_payload_json(identity_data: dict[str, Any]) -> str:
    """Build the canonical AgentIdentityPayload JSON for the Rust engine.

    Field names must match `AgentIdentityPayload` in governance.rs exactly
    so the Rust HMAC computation is identical to the Python fallback.
    """
    scopes = identity_data.get("scopes", [])
    if isinstance(scopes, (set, frozenset)):
        scopes = sorted(scopes)

    payload = {
        "agent_id": str(identity_data.get("agent_id", "")),
        "agent_type": str(identity_data.get("agent_type", "")),
        "created_at_ms": int(identity_data.get("created_at", 0) * 1000),
        "model": str(identity_data.get("model", "")),
        "organization": str(identity_data.get("organization", "")),
        "scopes": sorted(scopes),
        "session_id": str(identity_data.get("session_id", "")),
        "team": str(identity_data.get("team", "")),
        "user": str(identity_data.get("user", "")),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _python_compute_token(payload_json: str, key: str) -> str:
    """Pure-Python HMAC fallback (mirrors the Rust two-pass construction)."""
    if not key:
        return ""
    payload_hash = hashlib.sha256(payload_json.encode("utf-8")).digest()
    combined = key.encode("utf-8") + payload_hash
    token_bytes = hashlib.sha256(combined).hexdigest()
    return f"egov1:{token_bytes}"


def compute_identity_token(identity_data: dict[str, Any], key: str) -> str:
    """Compute a signed identity token — delegates to Rust engine when available.

    Both the Rust and Python implementations produce identical tokens for
    the same inputs.  The Rust path is preferred (consistent with Node/WASM).
    """
    if not key:
        return ""
    payload_json = _identity_payload_json(identity_data)
    if _RUST_AVAILABLE:
        try:
            return _rust_compute_token(payload_json, key)
        except Exception as exc:
            logger.warning("Rust token computation failed (%s); using Python fallback", exc)
    return _python_compute_token(payload_json, key)


def verify_token(identity: AgentIdentity, key: str) -> bool:
    """Verify an identity token against the operator key.

    Delegates to the Rust engine for constant-time comparison when available.
    """
    if not key or not identity.identity_token:
        return False
    payload_json = _identity_payload_json(identity.to_dict())
    if _RUST_AVAILABLE:
        try:
            return _rust_verify_token(payload_json, identity.identity_token, key)
        except Exception as exc:
            logger.warning("Rust token verification failed (%s); using Python fallback", exc)
    # Python fallback: constant-time comparison
    expected = _python_compute_token(payload_json, key)
    return hmac.compare_digest(identity.identity_token, expected)


def is_expired(identity: AgentIdentity, ttl: float | None = None) -> bool:
    """Check if an identity has expired based on TTL."""
    if ttl is None:
        ttl_str = os.environ.get(_IDENTITY_TTL_ENV, "")
        ttl = float(ttl_str) if ttl_str else _DEFAULT_TTL
    return (time.time() - identity.created_at) > ttl


# ── Identity Creation ────────────────────────────────────────────────

def create_identity(
    *,
    agent_id: str,
    agent_type: str = "unknown",
    organization: str = "",
    user: str = "",
    team: str = "",
    session_id: str = "",
    model: str = "",
    scopes: Sequence[str] | None = None,
    key: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentIdentity:
    """Create a new agent identity, optionally signed with the operator key.

    If ``key`` is provided or ENTROLY_IDENTITY_KEY is set, the identity
    token is HMAC-SHA256 signed.  Otherwise the identity is unsigned.
    """
    if key is None:
        key = os.environ.get(_IDENTITY_KEY_ENV, "")

    if scopes is None:
        scopes = ["read"]

    for scope in scopes:
        if not _SCOPE_RE.match(scope):
            raise IdentityError(
                f"Invalid scope {scope!r}. Expected format: "
                "'action' or 'action:path' where action is one of: "
                "read, write, execute, deploy, admin, review, approve"
            )

    if not session_id:
        session_id = _new_id()

    identity_data = {
        "agent_id": agent_id,
        "agent_type": agent_type,
        "organization": organization,
        "user": user,
        "team": team,
        "session_id": session_id,
        "model": model,
        "created_at": _now(),
        "scopes": sorted(scopes),
    }

    token = ""
    if key:
        token = compute_identity_token(identity_data, key)

    return AgentIdentity(
        agent_id=agent_id,
        agent_type=agent_type,
        organization=organization,
        user=user,
        team=team,
        session_id=session_id,
        model=model,
        created_at=identity_data["created_at"],
        identity_token=token,
        scopes=frozenset(scopes),
        metadata=metadata or {},
    )


# ── Identity Resolution ─────────────────────────────────────────────

def resolve_identity(
    *,
    header_value: str | None = None,
    env_value: str | None = None,
    bus: GovernanceEventBus | None = None,
) -> AgentIdentity:
    """Resolve agent identity from available sources.

    Priority:
      1. Explicit header value (X-Entroly-Agent-Identity)
      2. Environment variable (ENTROLY_AGENT_IDENTITY)
      3. Anonymous fallback (read-only)

    Emits an IdentityResolvedEvent on the governance bus.
    """
    if bus is None:
        bus = get_event_bus()

    raw = header_value or env_value or os.environ.get(_IDENTITY_ENV, "")
    source = "anonymous"

    if not raw:
        logger.info("No agent identity found; falling back to anonymous (read-only)")
        identity = AgentIdentity.anonymous()
    else:
        try:
            identity = AgentIdentity.from_dict(json.loads(raw))
            source = "header" if header_value else "environment"
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Failed to parse agent identity: %s; falling back to anonymous", exc)
            identity = AgentIdentity.anonymous()
            source = "parse_error"

        # Verify token if key is available
        key = os.environ.get(_IDENTITY_KEY_ENV, "")
        if key and identity.identity_token:
            if not verify_token(identity, key):
                logger.warning(
                    "Agent identity token verification failed for %s; "
                    "falling back to anonymous",
                    identity.agent_id,
                )
                identity = AgentIdentity.anonymous()
                source = "verification_failed"

        # Check expiration
        if identity.agent_id != "anonymous" and is_expired(identity):
            logger.warning(
                "Agent identity expired for %s; falling back to anonymous",
                identity.agent_id,
            )
            identity = AgentIdentity.anonymous()
            source = "expired"

    # Emit event
    bus.emit(IdentityResolvedEvent(
        tracing=TracingContext(
            agent_id=identity.agent_id,
            session_id=identity.session_id,
            organization_id=identity.organization,
            user_id=identity.user,
        ),
        payload={
            "source": source,
            "agent_id": identity.agent_id,
            "agent_type": identity.agent_type,
            "verified": identity.is_verified,
            "scopes": sorted(identity.scopes),
        },
    ))

    return identity


__all__ = [
    "IDENTITY_SCHEMA_VERSION",
    "KNOWN_AGENT_TYPES",
    "IdentityError",
    "IdentityExpiredError",
    "compute_identity_token",
    "verify_token",
    "is_expired",
    "create_identity",
    "resolve_identity",
]
