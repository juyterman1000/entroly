"""The governance control plane must be reachable from a real entry point.

`entroly/governance/` shipped with passing tests and no product path: nothing
outside the package imported it, so the repository's own reachability check
listed all seven modules as unreachable and the total rose from 37 to 44. The
architecture notes state the rule this violated directly — "a test that imports
a module directly does not prove a user can reach it."

`tests/test_governance.py` passed the whole time, because importing a module is
exactly what it does.

These tests assert the property instead: that the package is reachable from the
declared console entry points, and that `entroly govern` actually answers. They
also pin the four API mismatches found by *running* the command, each of which
was invisible to a direct-import test:

* `AuthorizationService.stats` is a property, not a method;
* `GovernanceAuditLog.verify_chain()` returns `(ok, detail)`, so `bool(...)` on
  the pair is always True — a broken chain would have reported intact;
* the identity credential field is `identity_token`, not `token`;
* `Policy` exposes `id`/`allowed_scopes`/`max_risk_level`, not `scope`/`effect`.
"""
from __future__ import annotations

import dataclasses
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args: str, env_extra: dict[str, str] | None = None, tmp_path: Path | None = None):
    import os

    env = dict(os.environ)
    env["ENTROLY_DISABLE_UPDATE_CHECK"] = "1"
    env["NO_COLOR"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if tmp_path is not None:
        env["ENTROLY_DIR"] = str(tmp_path / "state")
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-m", "entroly", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )


def _json_tail(stdout: str) -> dict:
    """CLI banners may precede JSON; take the last *top-level* object.

    `rfind("{")` lands on a nested brace and decodes a fragment, so this anchors
    on a brace that starts its own line — where the emitted document begins.
    """
    lines = stdout.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].startswith("{"):
            return json.loads("\n".join(lines[index:]))
    raise AssertionError(f"no top-level JSON object in output:\n{stdout[-800:]}")


def test_governance_is_reachable_from_a_real_entry_point():
    """The repository's own checker must not list governance as unreachable."""
    graph = REPO_ROOT / "scripts" / "codebase_graph.py"
    if not graph.exists():
        pytest.skip("codebase_graph.py is absent")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "graph.json"
        subprocess.run(
            [sys.executable, str(graph), "--json", str(out)],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=600,
        )
        # Written to a file rather than scraped from stdout, which carries a
        # human summary. Skipping on a parse failure would make this gate
        # vacuous exactly when it matters most.
        assert out.exists(), "codebase_graph.py produced no JSON artifact"
        unreachable = json.loads(out.read_text(encoding="utf-8")).get("unreachable") or []
    stranded = sorted(m for m in unreachable if "governance" in str(m))
    assert not stranded, (
        f"governance modules are unreachable from the declared entry points: "
        f"{stranded}. They ship, they are tested, and no user can invoke them."
    )


def test_entroly_govern_answers(tmp_path):
    """`entroly govern status` must run and report real control-plane state."""
    result = _run_cli("govern", "status", "--json", tmp_path=tmp_path)
    assert result.returncode == 0, f"govern status failed:\n{result.stdout}\n{result.stderr}"
    payload = _json_tail(result.stdout)

    for key in ("identity", "policies_loaded", "authorization", "audit_chain_intact"):
        assert key in payload, f"missing {key!r} in: {sorted(payload)}"
    assert isinstance(payload["authorization"], dict)
    assert isinstance(payload["audit_chain_intact"], bool), (
        "audit_chain_intact must be a bool; verify_chain returns a tuple and "
        "leaking it here would make a broken chain read as intact"
    )
    assert "claim_boundary" in payload


def test_policy_denies_by_default_with_a_reason(tmp_path):
    """Fail-closed, and say why — a denial without a reason is not auditable."""
    result = _run_cli(
        "govern", "policy", "check", "tool:write",
        "--resource", "src/app.py", "--risk", "high", "--json",
        tmp_path=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    payload = _json_tail(result.stdout)
    assert payload["allowed"] is False, f"write at high risk was allowed: {payload}"
    assert payload.get("reason"), "denial carried no reason"
    assert payload.get("matched_policy"), "denial named no policy"


def test_a_signed_identity_reports_its_token_and_an_unsigned_one_does_not(tmp_path):
    """`token_present` reads `identity_token`; `token` is always None.

    Reading the wrong attribute reported every signed identity as unsigned —
    a credential check that silently always says "absent".
    """
    unsigned = _json_tail(
        _run_cli("govern", "identity", "create", "--agent-id", "t", "--json",
                 tmp_path=tmp_path).stdout
    )
    signed = _json_tail(
        _run_cli("govern", "identity", "create", "--agent-id", "t", "--json",
                 env_extra={"ENTROLY_IDENTITY_KEY": "test-key-only"},
                 tmp_path=tmp_path).stdout
    )
    assert unsigned["token_present"] is False
    assert signed["token_present"] is True, (
        "a keyed identity reported no token; the field is `identity_token`"
    )
    # The credential itself must never be printed.
    assert "identity_token" not in signed and "token" not in signed


def test_the_governance_api_shapes_the_cli_depends_on():
    """Pin the four shapes that broke when assumed rather than read."""
    from entroly.governance.audit import GovernanceAuditLog
    from entroly.governance.authorization import AuthorizationService
    from entroly.governance.domain import AgentIdentity, Policy

    assert isinstance(
        inspect.getattr_static(AuthorizationService, "stats"), property
    ), "AuthorizationService.stats became a method; the CLI reads it as a property"

    signature = inspect.signature(GovernanceAuditLog.verify_chain)
    assert "tuple" in str(signature.return_annotation), (
        "verify_chain no longer returns a tuple; the CLI unpacks (ok, detail) "
        "and bool() on the pair would always be True"
    )

    identity_fields = {f.name for f in dataclasses.fields(AgentIdentity)}
    assert "identity_token" in identity_fields

    policy_fields = {f.name for f in dataclasses.fields(Policy)}
    assert {"id", "allowed_scopes", "max_risk_level"} <= policy_fields
