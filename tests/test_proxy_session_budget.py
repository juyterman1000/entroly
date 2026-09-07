"""The proxy must stop a session that has spent its policy budget.

A per-request cost clamp cannot stop a run that loops on a paid model: every
individual call looks affordable no matter how many follow it. Only a total
that accumulates across the session and refuses at a cap can, and
`governance.authorization` has owned that logic -- with a running total, a
fail-closed comparison, and a denial naming the numbers -- while nothing called
it. These tests wire it to the request path and pin the behaviour that makes it
safe to ship.

The default-off case is the one that matters most. The built-in
deny-by-default policy caps at $0.00, so a budget check that ran unconditionally
would refuse the first request of every default install. Enforcement requires
both an explicit switch and a policy naming a positive budget.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _fresh_authorization_service():
    """Every test starts from a zero ledger.

    The authorization service is a process-global singleton holding the running
    spend total, so a dollar recorded by one test would otherwise be inherited
    by the next and refuse its requests. Reset on the way out too, so these
    tests cannot leak spend into unrelated suites.
    """
    from entroly.governance.authorization import reset_authorization_service

    reset_authorization_service()
    yield
    reset_authorization_service()


@pytest.fixture()
def proxy(monkeypatch, tmp_path):
    """A proxy instance with isolated state and no enforcement configured."""
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("ENTROLY_ENFORCE_BUDGET", raising=False)
    from entroly.proxy import ProxyConfig, PromptCompilerProxy

    return PromptCompilerProxy(object(), ProxyConfig())


def _enforcing_proxy(monkeypatch, tmp_path):
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("ENTROLY_ENFORCE_BUDGET", "1")
    from entroly.proxy import ProxyConfig, PromptCompilerProxy

    return PromptCompilerProxy(object(), ProxyConfig())


def test_budget_enforcement_is_off_by_default(proxy):
    """No switch, no refusal — the common install must be untouched."""
    assert proxy._budget_enforced is False
    assert proxy._budget_refusal() is None


def test_enforcement_without_a_declared_budget_does_not_block(monkeypatch, tmp_path):
    """The built-in policy caps at $0.00; that must not mean "refuse everything".

    Asking for enforcement while no policy declares a budget is a
    misconfiguration. Refusing every request would be catastrophic and passing
    silently is indistinguishable from a working cap, so the proxy allows the
    request and warns once.
    """
    p = _enforcing_proxy(monkeypatch, tmp_path)
    assert p._budget_enforced is True
    # No policies.yaml exists under the isolated ENTROLY_DIR, so the built-in
    # $0.00 deny-by-default policy applies.
    assert p._budget_refusal() is None, (
        "a $0.00 built-in policy caused a refusal; every default install with "
        "the switch on would fail its first request"
    )

    # Spend must be on the clock for this to prove anything. `check_budget`
    # compares with a strict `>`, so at zero spend a $0.00 cap refuses nothing
    # and the guard above looks unnecessary. It becomes load-bearing the moment
    # a single cent is recorded: without it, every subsequent request against
    # the built-in policy is refused for exceeding a budget nobody set.
    from entroly.governance.authorization import get_authorization_service

    get_authorization_service().record_spend(0.01)
    assert p._budget_refusal() is None, (
        "one cent of spend against the $0.00 built-in policy caused a refusal; "
        "an install with the switch on and no declared budget would be bricked"
    )


def test_a_session_over_its_budget_is_refused_with_the_numbers(monkeypatch, tmp_path):
    """Once spend passes the cap the proxy refuses, and says by how much."""
    p = _enforcing_proxy(monkeypatch, tmp_path)

    from entroly.governance import authorization as auth
    from entroly.governance.domain import Policy, RiskLevel

    funded = Policy(
        id="test-budget", name="test", version="1", agent_type_pattern="*",
        allowed_scopes=frozenset({"read", "write", "execute"}),
        denied_paths=(), requires_approval_for=frozenset(),
        max_risk_level=RiskLevel.HIGH, max_files_per_change=50,
        max_lines_per_change=5000, budget_limit_usd=1.00,
    )
    monkeypatch.setattr(auth, "find_matching_policy", lambda *a, **k: funded, raising=False)
    monkeypatch.setattr(
        "entroly.governance.policy.find_matching_policy", lambda *a, **k: funded
    )

    service = auth.get_authorization_service()
    assert p._budget_refusal() is None, "a fresh session must not be refused"

    service.record_spend(1.50)  # over the $1.00 cap
    refusal = p._budget_refusal()
    assert refusal is not None, "spending past the cap did not refuse the request"
    assert refusal.status_code == 402

    import json
    payload = json.loads(refusal.body)
    assert payload["error"] == "budget_exceeded"
    assert "1.50" in payload["detail"] and "1.00" in payload["detail"], (
        f"the refusal must name spend and limit, got: {payload['detail']}"
    )
    assert "claim_boundary" in payload, "a refusal must state what it measured"
    assert p._budget_denials == 1


def test_unpriced_usage_is_not_debited(proxy, monkeypatch):
    """An event with no rate behind it must not move the running total.

    Debiting a guess would put an invented number into the total that later
    refuses real requests. The cap has to be as honest as the ledger it reads.
    """
    proxy._budget_enforced = True
    recorded: list[float] = []

    class _Svc:
        def record_spend(self, cost_usd): recorded.append(cost_usd)

    monkeypatch.setattr(
        "entroly.governance.authorization.get_authorization_service", lambda: _Svc()
    )

    class _Event:
        pricing_source = "unpriced:no-catalog-entry"
        cost_micro_usd = 5_000_000  # $5, but unpriced — a fabricated figure

    proxy._debit_session_budget(_Event())
    assert recorded == [], "an unpriced event was debited against the budget"

    class _Priced:
        pricing_source = "explicit"
        cost_micro_usd = 250_000  # $0.25

    proxy._debit_session_budget(_Priced())
    assert recorded == [0.25]
    assert proxy._budget_recorded_usd == pytest.approx(0.25)


def test_a_broken_governance_path_allows_the_request(proxy, monkeypatch):
    """Availability beats enforcement when the check itself fails.

    A budget check that cannot run must not take the proxy down. It also must
    not look like a working cap, which is why it warns.
    """
    proxy._budget_enforced = True
    monkeypatch.setattr(
        "entroly.governance.authorization.get_authorization_service",
        lambda: (_ for _ in ()).throw(RuntimeError("governance unavailable")),
    )
    assert proxy._budget_refusal() is None
