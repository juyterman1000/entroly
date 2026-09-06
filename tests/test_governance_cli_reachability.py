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


def test_audit_records_stay_inside_the_configured_state_directory(tmp_path):
    """`ENTROLY_DIR` must scope governance too, or isolation is a fiction.

    Running the documented `entroly govern audit verify` is what exposed this:
    it printed a path under the real `~/.entroly` while `ENTROLY_DIR` pointed
    at a temp directory. Twenty-four other modules honour that variable;
    governance was the one subsystem that could not be scoped to a project, so
    every test and sandbox was writing into the user's actual home.
    """
    result = _run_cli("govern", "audit", "verify", "--json", tmp_path=tmp_path)
    assert result.returncode in (0, 2), result.stderr
    payload = _json_tail(result.stdout)
    state_dir = str(tmp_path / "state")
    assert payload["jsonl_path"].startswith(state_dir), (
        f"audit log escaped ENTROLY_DIR: {payload['jsonl_path']} not under {state_dir}"
    )
    assert payload["db_path"].startswith(state_dir)


def _seed_audit_records(tmp_path, count: int) -> None:
    """Write records into the same state directory the CLI will read."""
    from entroly.governance.audit import GovernanceAuditLog

    log = GovernanceAuditLog(audit_dir=tmp_path / "state" / "governance" / "audit")
    for index in range(count):
        log.append(
            event_id=f"seed-{index}", event_type="permission.denied",
            agent_id="a", payload={"scope": "write"}, allowed=False,
        )


def test_an_explicit_zero_limit_returns_nothing(tmp_path):
    """`--limit 0` meant twenty.

    The handler read `int(limit or 20)`, so a legitimate zero was
    indistinguishable from an unset flag. `query(limit=0)` returns no records,
    which made the CLI the only layer misreading it.
    """
    _seed_audit_records(tmp_path, 5)
    payload = _json_tail(
        _run_cli("govern", "audit", "tail", "--limit", "0", "--json", tmp_path=tmp_path).stdout
    )
    assert payload["limit"] == 0
    assert payload["returned"] == 0, "an explicit zero limit returned records"

    bounded = _json_tail(
        _run_cli("govern", "audit", "tail", "--limit", "2", "--json", tmp_path=tmp_path).stdout
    )
    assert bounded["returned"] == 2, "the seeded records are not readable at all"


def test_a_negative_limit_is_refused_not_treated_as_unbounded(tmp_path):
    """SQLite reads a negative LIMIT as *no limit*.

    `--limit -1` therefore dumped the whole audit log -- the exact opposite of
    asking for a bound, on the one store designed to grow without end.
    """
    _seed_audit_records(tmp_path, 5)
    result = _run_cli("govern", "audit", "tail", "--limit", "-1", "--json", tmp_path=tmp_path)
    assert result.returncode != 0, "a negative limit was accepted"
    payload = _json_tail(result.stdout)
    assert payload["status"] == "error"
    assert "records" not in payload, "a refused request still returned audit records"


def test_a_rejected_scope_still_answers_in_json(tmp_path):
    """`--json` must produce JSON on the failure path too.

    An invalid scope raised past the handler and printed nothing to stdout,
    while an invalid `--risk` on the sibling command returned a structured
    error. A caller parsing `--json` got an empty stream rather than a refusal.
    """
    result = _run_cli(
        "govern", "identity", "create", "--agent-id", "a",
        "--scope", "root:/", "--json", tmp_path=tmp_path,
    )
    assert result.returncode != 0
    payload = _json_tail(result.stdout)
    assert payload["status"] == "error"
    assert "scope" in payload["reason"].lower()


def test_an_operator_policy_file_is_actually_read(tmp_path):
    """A control plane that cannot read its own policy file governs nothing.

    `governance/policy.py` is the only yaml importer in the package, and it
    degrades to built-in deny-by-default when the import fails -- logged, never
    raised. PyYAML was in no dependency group and is not pulled in
    transitively, so on a default `pip install entroly` every operator's
    policies.yaml was read by nothing while `load_policies` returned as though
    it had loaded.
    """
    state = tmp_path / "state" / "governance"
    state.mkdir(parents=True)
    (state / "policies.yaml").write_text(
        'policies:\n'
        '  - id: team-write\n'
        '    name: team can write\n'
        '    version: "2"\n'
        '    agent_type_pattern: "*"\n'
        '    allowed_scopes: [read, write]\n'
        '    max_risk_level: medium\n',
        encoding="utf-8",
    )
    payload = _json_tail(_run_cli("govern", "policy", "list", "--json", tmp_path=tmp_path).stdout)
    assert payload["yaml_available"] is True, "PyYAML is missing; policy files cannot be read"
    assert payload["using_builtin_fallback"] is False, (
        f"the operator's policy file was ignored: {payload.get('fallback_reason')}"
    )
    assert [p["id"] for p in payload["policies"]] == ["team-write"]


def test_a_silently_ignored_policy_file_says_so(tmp_path):
    """Falling back to read-only is right; doing it silently is not."""
    payload = _json_tail(_run_cli("govern", "policy", "list", "--json", tmp_path=tmp_path).stdout)
    assert payload["source_exists"] is False
    assert payload["using_builtin_fallback"] is True
    assert payload["fallback_reason"], (
        "the built-in policy was substituted with no reason given, which is "
        "indistinguishable from the operator's file having loaded"
    )


def test_audit_dir_precedence_resolves_at_call_time(monkeypatch, tmp_path):
    """Explicit argument > ENTROLY_AUDIT_DIR > ENTROLY_DIR > home.

    Each rung was wrong. The default was `Path("~/...").expanduser()` evaluated
    at import, so relocating HOME afterwards did nothing; and the environment
    outranked the explicit constructor argument, so `GovernanceAuditLog(
    audit_dir=tmp)` silently wrote elsewhere whenever the variable happened to
    be set -- voiding a caller's isolation without a word.
    """
    from entroly.governance.audit import GovernanceAuditLog

    explicit = tmp_path / "explicit"
    monkeypatch.setenv("ENTROLY_AUDIT_DIR", str(tmp_path / "env"))
    assert GovernanceAuditLog(audit_dir=explicit).jsonl_path.parent == explicit
    assert GovernanceAuditLog().jsonl_path.parent == tmp_path / "env"

    monkeypatch.delenv("ENTROLY_AUDIT_DIR")
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))
    assert GovernanceAuditLog().jsonl_path.parent == tmp_path / "state" / "governance" / "audit"

    # Resolved per call, not once at import: a later HOME must be respected.
    monkeypatch.delenv("ENTROLY_DIR")
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    assert GovernanceAuditLog().jsonl_path.is_relative_to(home)


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
