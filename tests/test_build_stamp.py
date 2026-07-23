"""get_stats carries a build stamp so an agent can tell if a fix is live.

Dogfooding had to cross-check the server process start time against the
compiled core's mtime by hand to know whether a shipped fix was running. The
build stamp exposes the engine version, native-engine mode, and — when the
native core is present — its build time, directly in get_stats.
"""

from __future__ import annotations

import platform

from entroly.server import EntrolyEngine


class _FakeEngine:
    _use_rust = False

    def _stats_python(self) -> dict:
        return {"session": {}}


def test_build_stamp_reports_version_and_engine_mode() -> None:
    stamp = EntrolyEngine._build_stamp(_FakeEngine())  # type: ignore[arg-type]
    assert stamp["native_engine"] is False
    assert stamp["python"] == platform.python_version()
    assert stamp.get("entroly_version")  # non-empty version string


def test_get_stats_includes_build_when_engineless() -> None:
    stats = EntrolyEngine.get_stats(_FakeEngine())  # type: ignore[arg-type]
    assert "build" in stats
    assert stats["build"]["python"] == platform.python_version()
