"""Pytest tests for the RAVS archetype classifier and router."""
from __future__ import annotations

import pytest
from entroly.ravs.router import BayesianRouter, classify_archetype


# ── classify_archetype ────────────────────────────────────────────────

def test_classify_read_file():
    assert classify_archetype("read the file utils.py") in ("file_read", "code_read", "read", "inspect")


def test_classify_run_tests():
    arch = classify_archetype("run pytest")
    assert "test" in arch or arch in ("run", "shell", "exec")


def test_classify_returns_string():
    assert isinstance(classify_archetype("explain this code"), str)
    assert len(classify_archetype("explain this code")) > 0


def test_classify_all_prompts_return_something():
    prompts = [
        "explain this code",
        "read the file utils.py",
        "run pytest",
        "refactor the auth module",
        "what is 2+2",
        "show me the git log",
        "check the build output",
        "look at the error in proxy.py",
        "add a unit test for the router",
        "update the README",
        "fix the authentication bug",
        "describe how RAVS works",
        "install the dependencies",
        "format the code with black",
    ]
    for p in prompts:
        result = classify_archetype(p)
        assert isinstance(result, str) and result, f"empty archetype for: {p!r}"


# ── BayesianRouter ────────────────────────────────────────────────────

@pytest.fixture
def router():
    return BayesianRouter(enabled=True)


def test_router_route_returns_decision(router):
    d = router.route("claude-3-opus-20240229", "run pytest")
    assert hasattr(d, "use_original")
    assert hasattr(d, "risk_level")
    assert hasattr(d, "reason")


def test_router_high_risk_keeps_original(router):
    d = router.route("claude-3-opus-20240229", "fix the authentication bug")
    assert d.use_original is True


def test_router_disabled_always_keeps_original():
    r = BayesianRouter(enabled=False)
    d = r.route("claude-3-opus-20240229", "run pytest")
    assert d.use_original is True


def test_router_stats_returns_dict(router):
    router.route("claude-3-opus-20240229", "run pytest")
    stats = router.stats()
    assert isinstance(stats, dict)


def test_router_risk_level_is_string(router):
    d = router.route("claude-3-opus-20240229", "explain this code")
    assert isinstance(d.risk_level, str)


if __name__ == "__main__":
    # Quick manual smoke test — not run by pytest
    r = BayesianRouter(enabled=True)
    prompts = [
        "explain this code", "read the file utils.py", "run pytest",
        "refactor the auth module", "fix the authentication bug",
    ]
    print("=== RAVS Prompt-Level Archetype Router ===\n")
    print(f"{'Prompt':<45} {'Archetype':<18} {'Risk':<10} {'Decision'}")
    print("-" * 105)
    for msg in prompts:
        arch = classify_archetype(msg)
        d = r.route("claude-3-opus-20240229", msg)
        swap = "SWAP → Haiku" if not d.use_original else "keep Opus"
        print(f"  {msg:<43} {arch:<18} {d.risk_level:<10} {swap:<15} ({d.reason})")
    print(f"\nRouter stats: {r.stats()}")
