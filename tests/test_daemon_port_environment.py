"""`entroly daemon` must honour the same port variables as `entroly proxy`.

`entroly proxy --help` documents its defaults as "9377, or ENTROLY_PROXY_PORT"
and "9378, or ENTROLY_DASHBOARD_PORT". The daemon starts the same proxy and
dashboard but hard-coded 9377 and 9378, so with ENTROLY_PROXY_PORT=39377 set
(because 9377 is taken) it still reported "proxy:9377 dashboard:9378" and
tried to bind the port the user had moved away from.
"""

from __future__ import annotations

import pytest

import entroly.cli as cli
from entroly.cli_parser import build_cli_parser


def _parse(*argv: str):
    parser = build_cli_parser(
        version=cli.__version__,
        wrap_agent_names=cli._wrap_agent_names,
        reason_options=cli._EXIT_REASON_OPTIONS,
        benefit_options=cli._EXIT_BENEFIT_OPTIONS,
        surface_options=cli._EXIT_SURFACE_OPTIONS,
        duration_options=cli._EXIT_DURATION_OPTIONS,
    )
    return parser.parse_args(["daemon", *argv])


class _Stop(Exception):
    pass


def _ports(monkeypatch, *argv: str) -> dict:
    import entroly.daemon as daemon_module

    seen = {}

    def fake_daemon(**kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(daemon_module, "EntrolyDaemon", fake_daemon)
    monkeypatch.setattr(daemon_module, "_install_log_buffer", lambda: None)
    with pytest.raises(_Stop):
        cli.cmd_daemon(_parse(*argv))
    return seen


def test_the_daemon_uses_the_port_variables(monkeypatch) -> None:
    monkeypatch.setenv("ENTROLY_PROXY_PORT", "39377")
    monkeypatch.setenv("ENTROLY_DASHBOARD_PORT", "39378")

    seen = _ports(monkeypatch)

    assert (seen["proxy_port"], seen["dashboard_port"]) == (39377, 39378)


def test_flags_still_win_over_the_variables(monkeypatch) -> None:
    monkeypatch.setenv("ENTROLY_PROXY_PORT", "39377")

    seen = _ports(monkeypatch, "--proxy-port", "40001")

    assert seen["proxy_port"] == 40001


def test_without_either_the_defaults_are_unchanged(monkeypatch) -> None:
    monkeypatch.delenv("ENTROLY_PROXY_PORT", raising=False)
    monkeypatch.delenv("ENTROLY_DASHBOARD_PORT", raising=False)

    seen = _ports(monkeypatch)

    assert (seen["proxy_port"], seen["dashboard_port"], seen["mcp_port"]) == (9377, 9378, 9379)
