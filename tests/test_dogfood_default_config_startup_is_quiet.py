"""A default start must not warn about Entroly's own defaults.

`entroly go` on a fresh project printed, between "Indexed 149 files" and
"Local services ready!":

    Auto-recovery will not run on streaming requests: it retries on a
    verification verdict, and witness_mode=audit produces one only on the
    non-streaming path. Set ENTROLY_WITNESS_MODE=annotate (or strict) to
    cover streaming.

Nothing in that sentence was chosen by the user. Auto-recovery defaults on and
WITNESS defaults to audit, so every first run opened with a warning about a
setting the user had never heard of. The fact it states is true and stays
reported where it is useful: `/stats` keeps `active_streaming: false` and
`streaming_blocked_by`. The line stays a warning for an operator who turned
auto-recovery on explicitly, because then it contradicts something they asked
for.
"""
from __future__ import annotations

import logging

import pytest

from entroly.proxy import create_proxy_app
from entroly.proxy_config import ProxyConfig

_MESSAGE = "Auto-recovery will not run on streaming requests"


def _start(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, **env: str):
    for name in ("ENTROLY_AUTO_RECOVERY", "ENTROLY_WITNESS", "ENTROLY_WITNESS_MODE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    caplog.set_level(logging.INFO, logger="entroly.proxy")
    app = create_proxy_app(
        object(), ProxyConfig(), start_dashboard=False, start_autotune=False
    )
    records = [r for r in caplog.records if _MESSAGE in r.getMessage()]
    return app.state.proxy, records


def test_default_start_does_not_warn_about_default_settings(monkeypatch, caplog) -> None:
    proxy, records = _start(monkeypatch, caplog)

    assert records, "the fact is still logged, at a level a default run does not print"
    assert all(r.levelno < logging.WARNING for r in records), [
        (r.levelname, r.getMessage()) for r in records
    ]
    # The reachability report is unchanged: /stats still names the blocker.
    assert proxy._auto_recovery_streaming_blocked_by == "witness_mode=audit"


def test_explicitly_enabled_recovery_still_warns(monkeypatch, caplog) -> None:
    proxy, records = _start(monkeypatch, caplog, ENTROLY_AUTO_RECOVERY="1")

    assert proxy._auto_recovery_streaming_blocked_by == "witness_mode=audit"
    assert any(r.levelno == logging.WARNING for r in records), (
        "an operator who asked for auto-recovery must hear that streaming "
        "requests will not get it"
    )
