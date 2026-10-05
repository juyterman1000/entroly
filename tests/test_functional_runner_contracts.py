"""Legacy functional runners must propagate both checks and unexpected errors."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "filename", ["test_functional.py", "test_intensive_functional.py", "test_deep_functional.py"]
)
@pytest.mark.parametrize("error", [AssertionError("broken contract"), RuntimeError("engine failed")])
def test_standalone_functional_runner_reports_uncaught_failure(filename, error, monkeypatch):
    spec = importlib.util.spec_from_file_location("functional_runner", Path(__file__).with_name(filename))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "real_sources", lambda: [])
    cases = [name for name in vars(module) if name.startswith("test_")]
    assert cases
    for name in cases:
        monkeypatch.setattr(module, name, lambda: None)

    def fail():
        raise error

    monkeypatch.setattr(module, cases[0], fail)
    assert module.run() is False
    assert module.failed == 1


def test_failed_check_fails_a_pytest_function(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "deep_runner", Path(__file__).with_name("test_deep_functional.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class BrokenEngine:
        def ingest_fragment(self, *_args, **_kwargs):
            return {"status": "error"}

    monkeypatch.setattr(module, "fresh_engine", lambda: (BrokenEngine(), "unused"))
    with pytest.raises(AssertionError):
        module.test_unicode_binary()
