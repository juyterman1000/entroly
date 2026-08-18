"""Which runaway-session protection is in force, and on which surface.

`entroly/session_rescue.py` compacts an append-only agent conversation before
it crosses the provider context limit: it defers while a warm provider cache
would be sacrificed, freezes compacted bytes so later turns do not rewrite the
prefix, and persists every omitted span before mutating the outbound copy.

`SessionRescueController` is pure policy -- it imports nothing HTTP-aware -- so
it was never proxy-*coupled*, only proxy-*exposed*, because the proxy happened
to be its first caller. `entroly.rescue_session` is the surface-neutral entry
point that pip, npm-via-Python, SDK and provider-SDK callers use to run the same
policy on their own conversations.

What remains proxy-only is being automatic: rescue rewrites the outbound
provider request, and the proxy is the only surface Entroly owns that sees one.
MCP tools (`remember_fragment`, `entroly_retrieve`, `optimize_context`,
`get_stats`, `analyze_codebase_health`, `smart_read`) are invoked with their own
arguments and never receive the host's transcript, so an MCP host that wants
rescue has to pass the conversation in.

Two failure modes these tests exist to prevent: a user running long sessions
over MCP who believes automatic protection is running, and a pip or SDK user
told they lack a capability their install has always contained.

These tests do NOT assert that rescue happens automatically under MCP. It
cannot, and a test demanding it would be demanding a lie.
"""

from __future__ import annotations

import pytest

from entroly.runtime_capabilities import (
    render_capabilities_text,
    runtime_capabilities,
)


# ── capability report ────────────────────────────────────────────────────────

def test_session_protection_separates_automatic_from_available() -> None:
    """The distinction is automatic vs callable, not proxy vs nothing.

    Reporting `active_modes: ["proxy"]` alone would tell a pip or SDK user they
    do not have a capability their install has always contained.
    """
    session = runtime_capabilities()["session_protection"]

    assert session["implemented"] is True
    assert session["automatic_modes"] == ["proxy"]
    assert "sdk" in session["callable_from"]
    assert session["entry_point"] == "entroly.rescue_session"
    assert session["enable_with"] == "entroly proxy"


def test_report_explains_why_only_the_proxy_is_automatic() -> None:
    """A boundary without a reason reads as an oversight and invites a "fix"."""
    session = runtime_capabilities()["session_protection"]

    reason = session["reason_not_automatic_elsewhere"].lower()
    assert "outbound" in reason
    assert "conversation" in reason


def test_report_keeps_the_two_properties_that_distinguish_it_from_compaction() -> None:
    """Recoverability and prefix stability are the whole differentiation.

    Summarizing compaction is lossy and rewrites the prefix, which throws away
    the warm provider cache. Losing either property in a refactor would remove
    the reason to prefer this over the harness's own compaction.
    """
    session = runtime_capabilities()["session_protection"]

    assert session["omissions_recoverable"] is True
    assert session["prefix_cache_stable"] is True


def test_human_output_names_both_the_automatic_mode_and_the_call() -> None:
    text = render_capabilities_text(runtime_capabilities())

    assert "Runaway-session rescue" in text
    assert "entroly proxy" in text
    assert "entroly.rescue_session" in text


def test_added_block_stays_privacy_safe() -> None:
    """The report must not gain path/error keys as it grows (see
    test_runtime_capabilities.test_report_is_stable_conservative_and_privacy_safe).
    """
    session = runtime_capabilities()["session_protection"]

    forbidden = {"path", "error", "exception", "traceback"}
    assert not any(str(key).casefold() in forbidden for key in session)


def test_render_tolerates_a_report_without_the_block() -> None:
    """Older callers may hold a report produced before this block existed."""
    report = runtime_capabilities()
    report.pop("session_protection")

    text = render_capabilities_text(report)

    assert "Runaway-session rescue" not in text
    assert "Engine:" in text


# ── MCP startup notice ───────────────────────────────────────────────────────

def _announce(monkeypatch: pytest.MonkeyPatch) -> None:
    from entroly import server

    monkeypatch.setattr(server, "_SESSION_PROTECTION_ANNOUNCED", False)
    server._announce_session_protection_mode()


def test_mcp_startup_states_the_mode_and_the_remedy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.delenv("ENTROLY_SESSION_RESCUE", raising=False)

    _announce(monkeypatch)

    err = capsys.readouterr().err
    assert "not automatic on the MCP surface" in err
    assert "entroly proxy" in err


def test_startup_notice_never_writes_to_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The MCP server speaks JSON-RPC on stdout; one stray line desyncs it."""
    monkeypatch.delenv("ENTROLY_SESSION_RESCUE", raising=False)

    _announce(monkeypatch)

    captured = capsys.readouterr()
    assert captured.out == "", (
        f"MCP stdout must carry protocol only, got: {captured.out!r}"
    )
    assert captured.err


def test_notice_is_printed_once_per_process(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    from entroly import server

    monkeypatch.delenv("ENTROLY_SESSION_RESCUE", raising=False)
    monkeypatch.setattr(server, "_SESSION_PROTECTION_ANNOUNCED", False)

    server._announce_session_protection_mode()
    server._announce_session_protection_mode()

    assert capsys.readouterr().err.count("not automatic on the MCP surface") == 1


def test_operator_who_disabled_rescue_is_not_nagged(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Telling someone how to enable what they switched off is noise."""
    monkeypatch.setenv("ENTROLY_SESSION_RESCUE", "0")

    _announce(monkeypatch)

    assert capsys.readouterr().err == ""


# ── the capability itself, off the proxy ─────────────────────────────────────

def _controller(tmp_path):
    from entroly.compression_retrieval_store_secure import CompressionRetrievalStore
    from entroly.session_rescue import SessionRescueController

    return SessionRescueController(
        recovery_store=CompressionRetrievalStore(tmp_path / "recovery.json")
    )


def _runaway(turns: int = 40) -> list[dict[str, str]]:
    """An append-only agent loop: short asks, bulky tool output.

    Tool output is what compacts -- see `rescue_session`. A fixture built from
    prose would return `pressure-observed` and prove nothing about the policy.
    """
    messages = [{"role": "system", "content": "You are a coding agent."}]
    for turn in range(turns):
        messages.append({"role": "user", "content": f"run step {turn}"})
        messages.append(
            {
                "role": "tool",
                "content": "\n".join(
                    f"[INFO] compiled crate_{i} v0.1.{i} in {i}ms" for i in range(300)
                ),
            }
        )
    return messages


def _over_watermark(messages) -> int:
    from entroly.session_rescue import estimate_message_tokens

    return max(1, int(estimate_message_tokens(messages) / 0.95))


def test_sdk_caller_gets_the_same_rescue_the_proxy_gets(tmp_path) -> None:
    """A pip/SDK user must reach the real policy, not a lesser reimplementation."""
    from entroly.session_rescue import estimate_message_tokens, rescue_session

    messages = _runaway()
    before = estimate_message_tokens(messages)

    result = rescue_session(
        "sdk-session",
        messages,
        context_window=_over_watermark(messages),
        controller=_controller(tmp_path),
    )

    assert result.action == "emergency-rescue"
    assert result.tokens_saved > 0
    assert estimate_message_tokens(result.messages) < before


def test_every_omission_is_receipted_before_the_copy_changes(tmp_path) -> None:
    """Recoverability is the property that separates this from summarizing."""
    from entroly.session_rescue import rescue_session

    messages = _runaway()

    result = rescue_session(
        "receipted-session",
        messages,
        context_window=_over_watermark(messages),
        controller=_controller(tmp_path),
    )

    assert result.tokens_saved > 0
    assert result.recovery_receipts


def test_prose_over_the_watermark_reports_instead_of_paraphrasing(tmp_path) -> None:
    """Nothing safely compressible found is an answer, not a malfunction.

    The compactor does not summarise reasoning. A caller must be able to tell
    "I compacted nothing" from "I compacted something", which is why `action`
    exists and why a silent zero would be the wrong contract.
    """
    from entroly.session_rescue import rescue_session

    messages = [{"role": "system", "content": "You are a coding agent."}]
    for turn in range(60):
        messages.append({"role": "user", "content": f"step {turn}: " + "detail " * 400})
        messages.append(
            {"role": "assistant", "content": f"result {turn}: " + "output " * 400}
        )

    result = rescue_session(
        "prose-session",
        messages,
        context_window=_over_watermark(messages),
        controller=_controller(tmp_path),
    )

    assert result.action == "pressure-observed"
    assert result.tokens_saved == 0


def test_below_the_watermark_the_conversation_is_untouched(tmp_path) -> None:
    """Safe to call every turn -- and calling every turn is how it sees growth."""
    from entroly.session_rescue import rescue_session

    messages = [{"role": "user", "content": "small"}]

    result = rescue_session(
        "quiet-session",
        messages,
        context_window=200_000,
        controller=_controller(tmp_path),
    )

    assert result.messages == messages


def test_caller_list_is_never_mutated(tmp_path) -> None:
    """The caller keeps its own transcript; only the outbound copy changes."""
    from entroly.session_rescue import estimate_message_tokens, rescue_session

    messages = _runaway()
    snapshot = [dict(message) for message in messages]

    rescue_session(
        "immutable-session",
        messages,
        context_window=max(1, int(estimate_message_tokens(messages) / 0.95)),
        controller=_controller(tmp_path),
    )

    assert messages == snapshot


def test_default_controller_is_shared_so_frozen_bytes_survive_turns() -> None:
    """Per-call controllers would recompress the prefix and void the cache.

    The freeze that keeps the prompt prefix byte-stable lives in per-conversation
    controller state, so the process-wide instance is load-bearing rather than a
    convenience.
    """
    from entroly.session_rescue import default_controller

    assert default_controller() is default_controller()


def test_rescue_session_is_importable_from_the_package_root() -> None:
    """`from entroly import rescue_session` is the documented SDK path."""
    import entroly

    assert hasattr(entroly, "rescue_session")
