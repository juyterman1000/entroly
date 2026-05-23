"""Tests for the gh#44 codex tracking fix.

Covers the fail-open preflight, the verified codex auth.json / config.toml
schema parsing, and the empirical post-launch watchdog.

Schema under test is verified against openai/codex source:
  auth.json : keys `auth_mode` (canonical), `OPENAI_API_KEY`, `tokens`
  config    : key  `forced_login_method` (chatgpt|api), `model_provider`
"""

from __future__ import annotations

import json

import pytest

from entroly.cli import (
    _codex_preflight,
    _PREFLIGHT,
    _proxy_request_count,
    _read_codex_config,
    _wrap_watchdog_report,
)


@pytest.fixture
def codex_home(tmp_path, monkeypatch):
    """Isolated fake ~/.codex; OPENAI_API_KEY cleared by default."""
    home = tmp_path / "codex_home"
    home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(home))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return home


def _write_auth(home, obj):
    (home / "auth.json").write_text(json.dumps(obj), encoding="utf-8")


def _write_cfg(home, text):
    (home / "config.toml").write_text(text, encoding="utf-8")


# ── _read_codex_config ──────────────────────────────────────────────


class TestReadCodexConfig:
    def test_missing_file_returns_empty(self, codex_home):
        assert _read_codex_config(codex_home) == {}

    def test_parses_valid_toml(self, codex_home):
        _write_cfg(codex_home, 'model_provider = "myprov"\n'
                               'forced_login_method = "chatgpt"\n')
        cfg = _read_codex_config(codex_home)
        assert cfg["model_provider"] == "myprov"
        assert cfg["forced_login_method"] == "chatgpt"

    def test_malformed_toml_does_not_raise(self, codex_home):
        _write_cfg(codex_home, "this is = = not [ valid toml ][[")
        # Must degrade to {} rather than explode the wrap command.
        assert _read_codex_config(codex_home) == {}


# ── _codex_preflight ────────────────────────────────────────────────


class TestCodexPreflight:
    def test_no_config_at_all_is_quiet(self, codex_home):
        # No auth.json, no config, no API key → no confident claim.
        assert _codex_preflight(9377) == []

    def test_chatgpt_via_canonical_auth_mode(self, codex_home):
        _write_auth(codex_home, {"auth_mode": "chatgpt", "tokens": {"id": "x"}})
        hints = _codex_preflight(9377)
        assert any("ChatGPT account" in h for h in hints)
        assert any("codex logout" in h for h in hints)

    def test_chatgpt_via_tokens_fallback_when_no_auth_mode(self, codex_home):
        # Older codex builds without the auth_mode field.
        _write_auth(codex_home, {"tokens": {"id_token": "x"}})
        assert any("ChatGPT account" in h for h in _codex_preflight(9377))

    def test_api_key_in_auth_json_suppresses_chatgpt_warning(self, codex_home):
        _write_auth(codex_home, {"auth_mode": "apikey",
                                 "OPENAI_API_KEY": "sk-test"})
        assert not any("ChatGPT account" in h for h in _codex_preflight(9377))

    def test_env_api_key_suppresses_warning_even_with_tokens(
        self, codex_home, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        _write_auth(codex_home, {"tokens": {"id_token": "x"}})
        assert not any("ChatGPT account" in h for h in _codex_preflight(9377))

    def test_forced_login_method_chatgpt_in_config(self, codex_home):
        _write_cfg(codex_home, 'forced_login_method = "chatgpt"\n')
        assert any("ChatGPT account" in h for h in _codex_preflight(9377))

    def test_custom_model_provider_warns_with_port(self, codex_home):
        _write_cfg(codex_home, 'model_provider = "azure"\n')
        hints = _codex_preflight(9377)
        joined = "\n".join(hints)
        assert 'model_provider = "azure"' in joined
        assert "localhost:9377/v1" in joined

    def test_model_provider_openai_is_not_flagged(self, codex_home):
        _write_cfg(codex_home, 'model_provider = "openai"\n')
        assert not any("model_provider" in h for h in _codex_preflight(9377))

    def test_malformed_auth_json_does_not_raise(self, codex_home):
        (codex_home / "auth.json").write_text("{ not json", encoding="utf-8")
        # Should not raise; just yields no confident chatgpt claim.
        assert _codex_preflight(9377) == []

    def test_preflight_registered_for_codex_only(self):
        assert "codex" in _PREFLIGHT
        assert _PREFLIGHT["codex"] is _codex_preflight


# ── _proxy_request_count ────────────────────────────────────────────


class TestProxyRequestCount:
    def test_unreachable_proxy_returns_none(self):
        # Nothing is listening on this port → None (never raises).
        assert _proxy_request_count(59999) is None


# ── _wrap_watchdog_report ───────────────────────────────────────────


SPEC = {"name": "OpenAI Codex CLI", "env_key": "OPENAI_BASE_URL",
        "env_val": "http://localhost:{port}/v1"}


class TestWatchdogReport:
    def test_silent_when_baseline_unknown(self, capsys, monkeypatch):
        monkeypatch.setattr("entroly.cli._proxy_request_count", lambda p: 5)
        _wrap_watchdog_report("codex", SPEC, 9377, before=None)
        assert capsys.readouterr().out == ""

    def test_silent_when_traffic_flowed(self, capsys, monkeypatch):
        monkeypatch.setattr("entroly.cli._proxy_request_count", lambda p: 9)
        _wrap_watchdog_report("codex", SPEC, 9377, before=3)
        assert capsys.readouterr().out == ""

    def test_silent_when_after_unmeasurable(self, capsys, monkeypatch):
        monkeypatch.setattr("entroly.cli._proxy_request_count", lambda p: None)
        _wrap_watchdog_report("codex", SPEC, 9377, before=3)
        assert capsys.readouterr().out == ""

    def test_warns_on_zero_delta(self, capsys, monkeypatch):
        monkeypatch.setattr("entroly.cli._proxy_request_count", lambda p: 3)
        _wrap_watchdog_report("codex", SPEC, 9377, before=3)
        out = capsys.readouterr().out
        assert "0 requests" in out
        assert "dashboard will be empty" in out

    def test_zero_delta_reuses_codex_hints(
        self, capsys, monkeypatch, codex_home
    ):
        # Ground truth says nothing flowed AND codex is in chatgpt mode:
        # the watchdog should surface the same targeted remediation.
        (codex_home / "config.toml").write_text(
            'forced_login_method = "chatgpt"\n', encoding="utf-8"
        )
        monkeypatch.setattr("entroly.cli._proxy_request_count", lambda p: 7)
        _wrap_watchdog_report("codex", SPEC, 9377, before=7)
        out = capsys.readouterr().out
        assert "ChatGPT account" in out
        assert "codex logout" in out

    def test_zero_delta_generic_when_no_specific_hint(
        self, capsys, monkeypatch
    ):
        monkeypatch.setattr("entroly.cli._proxy_request_count", lambda p: 1)
        spec = {"name": "Aider", "env_key": "OPENAI_API_BASE",
                "env_val": "http://localhost:{port}/v1"}
        _wrap_watchdog_report("aider", spec, 9377, before=1)
        out = capsys.readouterr().out
        assert "0 requests" in out
        assert "OPENAI_API_BASE" in out  # generic fallback names the var
