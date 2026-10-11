"""Auto-recovery must report where it can actually run.

Recovery retries a request with exact CCR originals when verification rejects
an answer produced from compressed context. It therefore only runs where a
verification verdict exists, and the two response paths do not agree on that:

    non-streaming   _forward_response                  gated on _witness_enabled
    streaming       _buffered_witness_stream_response   reached only when the
                                                        mode is annotate/strict

The default mode is "audit" (proxy.py promotes "off" to "audit" unless
ENTROLY_WITNESS=0), so a streaming request takes the plain path, never reaches
a verdict, and never attempts recovery. Coding agents stream, so that is the
common case -- and `/stats` reported `enabled: true` beside `attempted: 0`
with nothing naming the reason, which reads as "nothing needed recovering".

These tests pin the reachability report, not the recovery behaviour itself.
Enabling recovery on the default streaming path is a product decision about
buffering; stating honestly that it is unreachable is not.
"""
from __future__ import annotations

import pytest

from entroly.proxy import create_proxy_app
from entroly.proxy_config import ProxyConfig


def _proxy(**config):
    app = create_proxy_app(
        object(), ProxyConfig(**config), start_dashboard=False, start_autotune=False
    )
    return app.state.proxy


def test_default_mode_cannot_recover_on_the_streaming_path() -> None:
    proxy = _proxy()
    assert proxy._witness_mode == "audit", "default promotes off -> audit"
    assert proxy._auto_recovery_enabled, "ENTROLY_AUTO_RECOVERY defaults on"
    assert proxy._auto_recovery_streaming_blocked_by == "witness_mode=audit"


@pytest.mark.parametrize("mode", ["annotate", "strict"])
def test_verifying_modes_cover_both_paths(mode: str) -> None:
    proxy = _proxy(witness_mode=mode)
    assert proxy._auto_recovery_streaming_blocked_by is None


def test_stats_separates_the_two_paths() -> None:
    """`enabled` is the operator's setting, not a statement of reachability."""
    proxy = _proxy()
    buffered = bool(proxy._auto_recovery_enabled and proxy._witness_enabled)
    streaming = bool(
        proxy._auto_recovery_enabled
        and proxy._witness_mode in {"annotate", "strict"}
    )
    assert buffered is True
    assert streaming is False, (
        "a single `enabled` flag cannot describe both paths, which is what "
        "made the inert streaming case invisible"
    )


def test_disabling_recovery_reports_no_blocker() -> None:
    """An operator who turned it off should not be told something blocked it."""
    import os

    os.environ["ENTROLY_AUTO_RECOVERY"] = "0"
    try:
        proxy = _proxy()
        assert proxy._auto_recovery_enabled is False
        assert proxy._auto_recovery_streaming_blocked_by is None
    finally:
        os.environ.pop("ENTROLY_AUTO_RECOVERY", None)
