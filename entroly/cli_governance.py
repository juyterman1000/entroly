"""Operator interface for the governance control plane.

`entroly/governance/` shipped identity, policy, authorization, audit and event
modules with tests, but nothing outside the package imported it: the repository's
own reachability check listed all seven modules as unreachable, and the
architecture notes are explicit that "a test that imports a module directly does
not prove a user can reach it". This module is that product path.

Every subcommand is a thin, faithful projection of the governance API — no
parallel logic, no second source of truth. Where a value is not measured it is
reported as absent rather than defaulted, and every decision carries the reason
the policy engine gave.
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import Any

# ── Output helpers ────────────────────────────────────────────────────────────


def _emit(payload: dict[str, Any], *, as_json: bool) -> int:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            print(f"  {key}:")
            print("    " + json.dumps(value, indent=2, default=str).replace("\n", "\n    "))
        else:
            print(f"  {key}: {value}")
    return 0


def _error(message: str, *, as_json: bool) -> int:
    """Report a refusal without inventing a result."""
    if as_json:
        print(json.dumps({"status": "error", "reason": message}, indent=2))
    else:
        print(f"  error: {message}")
    return 1


# ── identity ──────────────────────────────────────────────────────────────────


def _cmd_identity(args: Any) -> int:
    from .governance.identity import (
        IdentityError,
        create_identity,
        is_expired,
        resolve_identity,
    )

    as_json = bool(getattr(args, "json_output", False))
    action = getattr(args, "identity_action", "show")

    if action == "create":
        scopes = [s for s in (getattr(args, "scope", None) or "").split(",") if s]
        try:
            identity = create_identity(
                agent_id=args.agent_id,
                agent_type=getattr(args, "agent_type", "") or "unknown",
                organization=getattr(args, "organization", "") or "",
                user=getattr(args, "user", "") or "",
                model=getattr(args, "model", "") or "",
                scopes=scopes or None,
            )
        except IdentityError as exc:
            # A rejected scope must still answer in the requested format. This
            # escaped past `--json` and printed nothing to stdout, so a caller
            # parsing the output got an empty stream instead of a refusal --
            # while `--risk banana` on the sibling command returned structured
            # JSON. Same class of error, two different contracts.
            return _error(str(exc), as_json=as_json)
        payload = _identity_payload(identity, is_expired)
        payload["claim_boundary"] = (
            "An identity asserts who is acting. It does not authorize an action; "
            "run `entroly govern policy check` for that."
        )
        return _emit(payload, as_json=as_json)

    try:
        identity = resolve_identity(env_value=os.environ.get("ENTROLY_AGENT_IDENTITY"))
    except IdentityError as exc:
        return _error(f"identity could not be resolved: {exc}", as_json=as_json)
    return _emit(_identity_payload(identity, is_expired), as_json=as_json)


def _identity_payload(identity: Any, is_expired: Any) -> dict[str, Any]:
    expired = is_expired(identity)
    return {
        "agent_id": getattr(identity, "agent_id", None),
        "agent_type": getattr(identity, "agent_type", None),
        "organization": getattr(identity, "organization", None) or None,
        "user": getattr(identity, "user", None) or None,
        "model": getattr(identity, "model", None) or None,
        "scopes": list(getattr(identity, "scopes", ()) or ()),
        "expired": expired,
        # The credential itself is never printed; report only that one
        # exists. The field is `identity_token` -- reading `token` returned
        # None and reported every signed identity as unsigned.
        "token_present": bool(getattr(identity, "identity_token", None)),
    }


# ── policy ────────────────────────────────────────────────────────────────────


def _cmd_policy(args: Any) -> int:
    from .governance.domain import RiskLevel
    from .governance.identity import IdentityError, resolve_identity
    from .governance.policy import (
        PolicyError,
        evaluate,
        load_policies,
        policy_source_status,
    )

    as_json = bool(getattr(args, "json_output", False))
    action = getattr(args, "policy_action", "list")

    try:
        policies = load_policies(getattr(args, "policy_file", None))
    except PolicyError as exc:
        return _error(f"policies could not be loaded: {exc}", as_json=as_json)

    if action == "list":
        # Echoing the requested path as "source" made a typo'd filename read as
        # "your file loaded and contains one policy". Two fallbacks are silent
        # -- a missing file and a missing PyYAML -- so ask the policy engine
        # which source is in force rather than inferring it from the path.
        return _emit(
            {
                "policy_count": len(policies),
                **policy_source_status(getattr(args, "policy_file", None)),
                "policies": [
                    {
                        # Field names taken from the Policy dataclass. Guessing
                        # `scope`/`effect` rendered every policy as blank, which
                        # reads as "no rules" rather than "wrong key".
                        "id": getattr(p, "id", None),
                        "name": getattr(p, "name", None),
                        "version": getattr(p, "version", None),
                        "agent_type_pattern": getattr(p, "agent_type_pattern", None),
                        "allowed_scopes": list(getattr(p, "allowed_scopes", ()) or ()),
                        "denied_paths": list(getattr(p, "denied_paths", ()) or ()),
                        "max_risk_level": getattr(
                            getattr(p, "max_risk_level", None), "value", None
                        ),
                        "budget_limit_usd": getattr(p, "budget_limit_usd", None),
                    }
                    for p in policies
                ],
            },
            as_json=as_json,
        )

    # check
    try:
        identity = resolve_identity(env_value=os.environ.get("ENTROLY_AGENT_IDENTITY"))
    except IdentityError as exc:
        return _error(f"identity could not be resolved: {exc}", as_json=as_json)

    try:
        risk = RiskLevel(getattr(args, "risk", "low"))
    except ValueError:
        return _error(
            f"unknown risk level {getattr(args, 'risk', None)!r}; "
            f"expected one of {[r.value for r in RiskLevel]}",
            as_json=as_json,
        )

    decision = evaluate(
        identity,
        args.scope,
        resource=getattr(args, "resource", "") or "",
        risk_level=risk,
        policies=policies,
    )
    return _emit(
        {
            "agent_id": getattr(identity, "agent_id", None),
            "scope": args.scope,
            "resource": getattr(args, "resource", "") or None,
            "risk_level": risk.value,
            "allowed": bool(getattr(decision, "allowed", False)),
            # PolicyDecision carries `requires_approval` and the matched
            # `policy`; there is no `verdict` field. Emitting one produced a
            # permanently empty key, which reads as "no verdict" rather than
            # "wrong name".
            "requires_approval": bool(getattr(decision, "requires_approval", False)),
            "reason": getattr(decision, "reason", None),
            "matched_policy": getattr(getattr(decision, "policy", None), "id", None),
            "matched_policy_name": getattr(getattr(decision, "policy", None), "name", None),
            "claim_boundary": (
                "This is the policy verdict for the identity and scope given. It "
                "is not evidence that the action was performed or verified."
            ),
        },
        as_json=as_json,
    )


# ── audit ─────────────────────────────────────────────────────────────────────


def _cmd_audit(args: Any) -> int:
    from .governance.audit import get_audit_log

    as_json = bool(getattr(args, "json_output", False))
    action = getattr(args, "audit_action", "tail")
    log = get_audit_log()

    if action == "verify":
        # `verify_chain` returns `(ok, detail)`. `bool(tuple)` is True for any
        # non-empty tuple, so unpacking is load-bearing: taking the truthiness
        # of the pair would report a broken chain as intact.
        ok, detail = log.verify_chain()
        payload = {
            "chain_intact": bool(ok),
            "detail": detail,
            "jsonl_path": str(getattr(log, "jsonl_path", "") or ""),
            "db_path": str(getattr(log, "db_path", "") or ""),
            # Measured, not assumed. Against a chain of five records this
            # detects a payload edit and a deletion from the middle, and does
            # NOT detect tail truncation, an edit whose chain was recomputed,
            # or a wholly fabricated self-consistent log -- it reported "Chain
            # intact" over attacker-written records granting admin. The chain
            # is unkeyed SHA-256, so write access to the file is enough to
            # forge a consistent history. An earlier version of this string
            # claimed entries "were not altered after the fact", which is the
            # exact overclaim the trust invariants forbid.
            "detects": ["payload edited in place", "record removed from the middle"],
            "does_not_detect": [
                "records truncated from the end",
                "an edit whose chain hash was recomputed",
                "a fabricated log that is internally consistent",
            ],
            "claim_boundary": (
                "The chain is unkeyed, so this detects accidental corruption and "
                "naive edits, not a writer who recomputes it. It also does not "
                "prove every action was recorded."
            ),
        }
        _emit(payload, as_json=as_json)
        # Fail closed: a broken chain must be a non-zero exit for CI use.
        return 0 if ok else 2

    # `int(x or 20)` turned an explicit `--limit 0` into 20: the caller asked
    # for nothing and got twenty. `query(limit=0)` returns zero records, so the
    # CLI was the only layer misreading it.
    raw_limit = getattr(args, "limit", None)
    limit = 20 if raw_limit is None else int(raw_limit)
    if limit < 0:
        # SQLite reads a negative LIMIT as unbounded, so `--limit -1` silently
        # dumped the entire audit log -- the opposite of asking for a bound,
        # on the one store that grows without end.
        return _error(
            f"--limit must be zero or greater, got {limit}. A negative limit "
            f"returns the entire audit log rather than bounding it.",
            as_json=as_json,
        )
    records = list(log.query(limit=limit))
    return _emit(
        {
            "returned": len(records),
            "limit": limit,
            "jsonl_path": str(getattr(log, "jsonl_path", "") or ""),
            "records": [
                r if isinstance(r, dict) else getattr(r, "__dict__", {}) for r in records
            ],
        },
        as_json=as_json,
    )


# ── status ────────────────────────────────────────────────────────────────────


def _cmd_status(args: Any) -> int:
    """One view of whether the control plane is actually engaged.

    Each field is reported from the live objects. A capability that is not
    configured reads as absent, never as a zero that could be mistaken for a
    measurement.
    """
    from .governance.audit import get_audit_log
    from .governance.authorization import get_authorization_service
    from .governance.identity import IdentityError, is_expired, resolve_identity
    from .governance.policy import PolicyError, load_policies

    as_json = bool(getattr(args, "json_output", False))

    identity_payload: dict[str, Any] | None
    try:
        identity = resolve_identity(env_value=os.environ.get("ENTROLY_AGENT_IDENTITY"))
        identity_payload = _identity_payload(identity, is_expired)
    except IdentityError as exc:
        identity_payload = {"resolved": False, "reason": str(exc)[:200]}

    try:
        policy_count: int | None = len(load_policies(None))
        policy_error = None
    except PolicyError as exc:
        policy_count, policy_error = None, str(exc)[:200]

    service = get_authorization_service()
    stats = service.stats  # a property returning an AuthorizationStats dataclass()

    log = get_audit_log()

    return _emit(
        {
            "identity": identity_payload,
            "policies_loaded": policy_count,
            "policy_error": policy_error,
            "authorization": (
                dataclasses.asdict(stats) if dataclasses.is_dataclass(stats)
                else getattr(stats, "__dict__", {})
            ),
            "audit_chain_intact": log.verify_chain()[0],
            "audit_jsonl": str(getattr(log, "jsonl_path", "") or ""),
            "claim_boundary": (
                "Reports the state of the local control plane only. It is not an "
                "attestation that every agent action passed through it."
            ),
        },
        as_json=as_json,
    )


# ── entry point ───────────────────────────────────────────────────────────────


_DISPATCH = {
    "identity": _cmd_identity,
    "policy": _cmd_policy,
    "audit": _cmd_audit,
    "status": _cmd_status,
}


def cmd_govern(args: Any) -> int:
    """`entroly govern` — operator interface to the governance control plane."""
    group = getattr(args, "govern_group", None)
    handler = _DISPATCH.get(group)
    if handler is None:
        return _error(
            f"unknown governance group {group!r}; expected one of "
            f"{sorted(_DISPATCH)}",
            as_json=bool(getattr(args, "json_output", False)),
        )
    return handler(args)


__all__ = ["cmd_govern"]
