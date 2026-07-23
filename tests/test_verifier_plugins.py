"""Tests for the entroly.verifier plugin entry-point seam (entroly/plugins.py).

Trust contract under test: plugins are additive + attributed, a raising plugin
is recorded and skipped (never breaks verification), results are sanitized and
bounded, and ENTROLY_PLUGINS=0 disables discovery entirely.
"""

from __future__ import annotations

import entroly.plugins as plugins


class _FakeEntryPoint:
    def __init__(self, name, fn=None, raises=None):
        self.name = name
        self._fn = fn
        self._raises = raises

    def load(self):
        if self._raises:
            raise self._raises
        return self._fn


def _install(monkeypatch, eps):
    monkeypatch.setattr(plugins, "entry_points", lambda group: eps)
    monkeypatch.setattr(plugins, "_cache", None)


def test_discovers_and_runs_a_plugin(monkeypatch):
    def verify(evidence, claim):
        return {"verdict": "supported", "confidence": 0.9, "note": "ok"}

    _install(monkeypatch, [_FakeEntryPoint("goodv", verify)])
    out = plugins.run_verifier_plugins("the sky is blue", "sky is blue")
    assert set(out) == {"goodv"}
    assert out["goodv"]["verdict"] == "supported"
    assert out["goodv"]["confidence"] == 0.9
    assert out["goodv"]["status"] == "ok"
    assert "elapsed_ms" in out["goodv"]


def test_raising_plugin_is_recorded_not_fatal(monkeypatch):
    def bad(evidence, claim):
        raise RuntimeError("plugin exploded")

    def good(evidence, claim):
        return {"verdict": "uncertain"}

    _install(monkeypatch, [_FakeEntryPoint("bad", bad), _FakeEntryPoint("good", good)])
    out = plugins.run_verifier_plugins("e", "c")
    assert out["bad"]["status"] == "error"
    assert "plugin exploded" in out["bad"]["error"]
    # The healthy plugin still ran — one failure never blocks the rest.
    assert out["good"]["verdict"] == "uncertain"


def test_unloadable_and_noncallable_plugins_are_skipped(monkeypatch):
    _install(monkeypatch, [
        _FakeEntryPoint("broken", raises=ImportError("missing dep")),
        _FakeEntryPoint("notcallable", fn="just a string"),
    ])
    assert plugins.discover_verifiers(refresh=True) == {}


def test_results_are_sanitized_and_bounded(monkeypatch):
    def noisy(evidence, claim):
        big = {f"k{i}": "x" * 10_000 for i in range(100)}
        big["obj"] = object()
        return big

    _install(monkeypatch, [_FakeEntryPoint("noisy", noisy)])
    out = plugins.run_verifier_plugins("e", "c")["noisy"]
    assert out.get("_truncated") is True
    # Values are stringified and capped, keeping the report JSON-safe/bounded.
    assert all(len(v) <= 500 for v in out.values() if isinstance(v, str))
    import json
    json.dumps(out)  # must serialize


def test_kill_switch_disables_discovery(monkeypatch):
    def verify(evidence, claim):
        return {"verdict": "supported"}

    _install(monkeypatch, [_FakeEntryPoint("goodv", verify)])
    monkeypatch.setenv("ENTROLY_PLUGINS", "0")
    assert plugins.discover_verifiers(refresh=True) == {}
    assert plugins.run_verifier_plugins("e", "c") == {}


def test_non_dict_result_is_wrapped(monkeypatch):
    _install(monkeypatch, [_FakeEntryPoint("scalar", lambda e, c: 0.7)])
    out = plugins.run_verifier_plugins("e", "c")["scalar"]
    assert out["status"] == "ok"
    assert out["result"] == "0.7"
