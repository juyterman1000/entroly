from __future__ import annotations

import scripts.verify_release_readiness as gate


class FakeNative:
    available = True
    version = gate.__version__
    missing_symbols = ()
    version_ok = True
    error = None


def test_release_gate_requires_exact_tag_clean_tree_and_native_capabilities(monkeypatch) -> None:
    responses = {
        ("describe", "--tags", "--exact-match", "HEAD"): (
            0,
            f"entroly-v{gate.__version__}",
        ),
        ("status", "--porcelain"): (0, ""),
    }
    monkeypatch.setattr(gate, "_git", lambda *args: responses[args])
    monkeypatch.setattr(gate, "native_status", lambda symbols: FakeNative())
    assert gate.release_readiness() == []


def test_same_version_stale_native_wheel_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        gate,
        "_git",
        lambda *args: (
            (0, f"entroly-v{gate.__version__}")
            if args[0] == "describe"
            else (0, "")
        ),
    )
    stale = FakeNative()
    stale.missing_symbols = ("extract_skeleton",)
    monkeypatch.setattr(gate, "native_status", lambda symbols: stale)
    failures = gate.release_readiness()
    assert any("capability-stale" in failure for failure in failures)
