from __future__ import annotations

import sqlite3

import pytest

from entroly.server import _apply_mcp_access_policy, create_mcp_server
from entroly.session_attach import (
    DEFAULT_SCOPES,
    AttachmentError,
    AttachmentStore,
    install_attachment,
    parse_ttl,
    resolve_tool_scopes,
    uninstall_attachment,
)


def test_attachment_grant_is_scoped_hashed_expiring_and_revocable(tmp_path):
    store = AttachmentStore(tmp_path / "state")
    issued = store.create(
        client="codex",
        project_root=tmp_path,
        scopes=["context", "verify"],
        ttl_seconds=60,
        session_id="task-127",
        now=1000,
    )
    token = issued.token_file.read_text(encoding="utf-8").strip()

    assert issued.grant.scopes == ("context", "verify")
    assert "optimize_context" in issued.grant.tools
    assert "prepare_task_dream" in issued.grant.tools
    assert "remember_fragment" not in issued.grant.tools
    assert token not in store.db_path.read_bytes().decode("latin-1")
    assert store.authorize(issued.grant.grant_id, token, tool="optimize_context", now=1010)
    with pytest.raises(AttachmentError, match="outside"):
        store.authorize(issued.grant.grant_id, token, tool="remember_fragment", now=1011)
    with pytest.raises(AttachmentError, match="invalid"):
        store.authorize(issued.grant.grant_id, "wrong-token", now=1012)
    with pytest.raises(AttachmentError, match="expired"):
        store.authorize(issued.grant.grant_id, token, now=1060)

    revoked = store.revoke(issued.grant.grant_id, now=1020)
    assert revoked.revoked_at == 1020
    assert not issued.token_file.exists()
    with pytest.raises(AttachmentError, match="revoked"):
        store.authorize(issued.grant.grant_id, token, now=1021)

    with sqlite3.connect(store.db_path) as connection:
        events = connection.execute(
            "SELECT event FROM attachment_events WHERE grant_id = ? ORDER BY event_id",
            (issued.grant.grant_id,),
        ).fetchall()
    assert events == [
        ("created",),
        ("authorization_denied",),
        ("authorization_denied",),
        ("authorization_denied",),
        ("revoked",),
        ("authorization_denied",),
    ]


def test_attachment_install_commands_never_contain_raw_token(tmp_path):
    store = AttachmentStore(tmp_path / "state")
    for client in ("claude", "codex", "openclaw"):
        issued = store.create(client=client, project_root=tmp_path, ttl_seconds=60)
        token = issued.token_file.read_text(encoding="utf-8").strip()
        rendered = " ".join(part for command in issued.install_commands for part in command)

        assert token not in rendered
        assert issued.grant.grant_id in rendered
        assert str(issued.token_file) in rendered
        assert f"ENTROLY_SOURCE={tmp_path.resolve()}" in rendered


def test_mcp_access_policy_removes_ungranted_tools_and_reauthorizes_each_call():
    mcp_module = pytest.importorskip("mcp.server.fastmcp")
    mcp = mcp_module.FastMCP("attachment-test")

    @mcp.tool()
    def allowed(value: int) -> int:
        return value + 1

    @mcp.tool()
    def denied() -> str:
        return "never"

    @mcp.resource("test://status")
    def status() -> str:
        return "secret-free but unscoped"

    @mcp.prompt()
    def workflow() -> str:
        return "unscoped"

    checks = []
    _apply_mcp_access_policy(
        mcp,
        allowed_tools={"allowed"},
        authorize_tool=checks.append,
    )

    assert set(mcp._tool_manager._tools) == {"allowed"}
    assert mcp._resource_manager._resources == {}
    assert mcp._prompt_manager._prompts == {}
    assert mcp._tool_manager._tools["allowed"].fn(4) == 5
    assert mcp._tool_manager._tools["allowed"].fn(9) == 10
    assert checks == ["allowed", "allowed"]

    with pytest.raises(RuntimeError, match="unavailable"):
        _apply_mcp_access_policy(
            mcp,
            allowed_tools={"missing"},
            authorize_tool=None,
        )


def test_default_attachment_scope_matches_the_real_mcp_surface(tmp_path, monkeypatch):
    pytest.importorskip("mcp.server.fastmcp")
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))
    _scopes, tools = resolve_tool_scopes(DEFAULT_SCOPES)

    mcp, _engine = create_mcp_server(
        allowed_tools=set(tools),
        authorize_tool=lambda _name: None,
    )

    assert set(mcp._tool_manager._tools) == set(tools)
    assert mcp._resource_manager._resources == {}
    assert mcp._resource_manager._templates == {}
    assert mcp._prompt_manager._prompts == {}


def test_failed_client_install_rolls_back_and_revokes_grant(tmp_path):
    store = AttachmentStore(tmp_path / "state")
    issued = store.create(client="codex", project_root=tmp_path, ttl_seconds=60)
    commands = []

    def failing_runner(command, **_kwargs):
        commands.append(tuple(command))
        if command[:3] == ("codex", "mcp", "add"):
            return __import__("subprocess").CompletedProcess(command, 2, "", "bad config")
        return __import__("subprocess").CompletedProcess(command, 0, "", "")

    with pytest.raises(AttachmentError, match="rolled back and grant revoked"):
        install_attachment(issued, store=store, runner=failing_runner)

    assert any(command[:3] == ("codex", "mcp", "remove") for command in commands)
    assert store.get(issued.grant.grant_id).status == "revoked"
    assert not issued.token_file.exists()


def test_client_uninstall_failure_is_actionable_after_access_is_revoked(tmp_path):
    store = AttachmentStore(tmp_path / "state")
    issued = store.create(client="claude", project_root=tmp_path, ttl_seconds=60)
    revoked = store.revoke(issued.grant.grant_id)

    def failing_runner(command, **_kwargs):
        return __import__("subprocess").CompletedProcess(command, 3, "", "entry locked")

    with pytest.raises(AttachmentError, match="access is revoked") as exc_info:
        uninstall_attachment(revoked, runner=failing_runner)

    assert "run manually" in str(exc_info.value)
    assert "claude mcp remove" in str(exc_info.value)
    assert store.get(issued.grant.grant_id).status == "revoked"


def test_attachment_scope_and_ttl_validation():
    assert parse_ttl("30m") == 1800
    scopes, tools = resolve_tool_scopes([])
    assert scopes
    assert "optimize_context" in tools
    assert "remember_fragment" not in tools
    with pytest.raises(ValueError, match="unknown"):
        resolve_tool_scopes(["root"])
    with pytest.raises(ValueError, match="30 days"):
        parse_ttl("31d")
