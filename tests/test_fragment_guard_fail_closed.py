from __future__ import annotations

import logging

from entroly.adaptive_pruner import FragmentGuard


class _FailingGuard:
    def review_code(self, content: str, source: str):
        raise RuntimeError(f"sensitive backend detail: {content}")


def test_fragment_guard_backend_failure_is_not_reported_clean(caplog):
    guard = FragmentGuard()
    guard._guard = _FailingGuard()

    with caplog.at_level(logging.ERROR, logger="entroly.adaptive_pruner"):
        issues = guard.scan("SECRET_MUST_NOT_LEAK", source="file:broken.py")

    assert issues == [
        "code quality scan failed (RuntimeError); treat fragment as unverified"
    ]
    assert "SECRET_MUST_NOT_LEAK" not in issues[0]
    assert "backend failed while scanning file:broken.py" in caplog.text


def test_fragment_guard_empty_content_does_not_call_backend():
    guard = FragmentGuard()
    guard._guard = _FailingGuard()

    assert guard.scan("", source="file:empty.py") == []
