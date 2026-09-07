"""The proxy must stop a session that has spent its policy budget.

A per-request cost clamp cannot stop a run that loops on a paid model: every
individual call looks affordable no matter how many follow it. Only a total
that accumulates across the session and refuses at a cap can, and
`governance.authorization` has owned that logic -- with a running total, a
fail-closed comparison, and a denial naming the numbers -- while nothing called
it. These tests wire it to the request path and pin the behaviour that makes it
safe to ship.

Writing `budget_limit_usd` into a policy is the opt-in. There is no second
switch, because a flag nobody sets leaves the cap doing nothing for exactly the
operators who asked for one. Safety comes from the policy instead: the built-in
deny-by-default policy caps at $0.00, and a non-positive budget disables
enforcement, so an install without a policies.yaml is untouched.
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
    monkeypatch.delenv("ENTROLY_ENFORCE_BUDGET", raising=False)
    from entroly.proxy import ProxyConfig, PromptCompilerProxy

    return PromptCompilerProxy(object(), ProxyConfig())


def test_enforcement_is_on_by_default_but_a_default_install_is_untouched(proxy):
    """The cap is armed without configuration, and still refuses nothing.

    Both halves matter. An opt-in switch nobody sets is indistinguishable from
    an unwired feature -- an operator who writes `budget_limit_usd` into a
    policy has already said what they want, and requiring a second flag means
    the cap silently does nothing for exactly the people who asked for it.

    Safety comes from the policy instead: the built-in deny-by-default policy
    caps at $0.00, and a non-positive budget disables enforcement, so an
    install with no policies.yaml behaves exactly as before.
    """
    assert proxy._budget_enforced is True, "the cap must be armed by default"
    assert proxy._budget_refusal() is None, (
        "a default install with no declared budget was refused"
    )


def test_enforcement_can_be_switched_off(monkeypatch, tmp_path):
    """An escape hatch has to exist for anyone who needs the old behaviour."""
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("ENTROLY_ENFORCE_BUDGET", "0")
    from entroly.proxy import ProxyConfig, PromptCompilerProxy

    p = PromptCompilerProxy(object(), ProxyConfig())
    assert p._budget_enforced is False
    assert p._budget_refusal() is None


def test_enforcement_without_a_declared_budget_does_not_block(monkeypatch, tmp_path):
    """The built-in policy caps at $0.00; that must not mean "refuse everything".

    This is the ordinary path for an install with no policies.yaml, not a
    misconfiguration, and it is why the cap can be armed by default at all.
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


def test_the_first_identity_resolution_is_audited(tmp_path, monkeypatch):
    """Attaching the subscriber after construction misses the founding record.

    `from_environment` resolves the identity, and that emits
    `identity.resolved`. Installing the subscriber afterwards means the event
    has already been delivered to an empty handler list, so it is gone.

    In a process that builds the service once -- the normal case -- that event
    is the entire record of which agent is acting. Measured before the fix:
    zero records existed after the only service creation the process made, so
    the audit trail opened with the establishing entry already missing.
    """
    monkeypatch.setenv("ENTROLY_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))

    import entroly.governance.audit as audit_module
    from entroly.governance.authorization import get_authorization_service
    from entroly.governance.events import reset_event_bus

    monkeypatch.setattr(audit_module, "_global_log", None, raising=False)

    # The bus must be reset, not just the service. It is a separate global
    # that outlives both, and it carries the installed subscriber. Without
    # this, an earlier test in the same process has already attached one, so
    # the subscriber is present no matter when this call installs it -- and
    # the ordering this test exists to pin becomes unobservable.
    reset_event_bus()
    try:
        get_authorization_service()  # the only creation, as in a real process
    finally:
        reset_event_bus()  # leave no half-wired bus for the next test

    types = {r["event_type"] for r in audit_module.get_audit_log().query(limit=20)}
    assert "identity.resolved" in types, (
        f"identity resolution was not audited (recorded: {sorted(types) or 'nothing'}); "
        "the subscriber attaches after the event has already been dispatched"
    )


def test_the_audit_subscriber_installs_once_per_bus(tmp_path, monkeypatch):
    """Re-creating the service must not double-write every audit record.

    `subscribe` appends unconditionally, so a second install attaches a second
    wildcard handler and each event is appended twice. Recreating the service
    is ordinary -- `get_authorization_service(reset=True)` does it -- so this
    is reachable in a normal process, not just in tests.

    The assertion reads the JSONL chain, not `query`. The two sinks in `append`
    handle the duplicate differently: SQLite drops it (`INSERT OR IGNORE` on
    `event_id`) so `query` looks correct, while the JSONL keeps it. Since
    `verify_chain` walks the JSONL, the inflated copy is the one that gets
    certified intact -- a `query`-based assertion passes while the audit trail
    is wrong.
    """
    monkeypatch.setenv("ENTROLY_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))

    import entroly.governance.audit as audit_module
    from entroly.governance.authorization import get_authorization_service
    from entroly.governance.domain import RiskLevel

    monkeypatch.setattr(audit_module, "_global_log", None, raising=False)

    get_authorization_service()
    service = get_authorization_service(reset=True)  # second install attempt
    service.check("write", resource="src/app.py", risk_level=RiskLevel.HIGH)

    # A total count would be wrong: creating the service resolves identity,
    # which is itself an auditable event, so the number legitimately varies.
    # A duplicate subscriber appends the *same* event twice, so the defect is
    # precisely a repeated event_id in the chain.
    import json
    from collections import Counter

    log = audit_module.get_audit_log()
    lines = log.jsonl_path.read_text(encoding="utf-8").splitlines()
    counts = Counter(json.loads(ln)["event_id"] for ln in lines if ln.strip())
    repeated = [eid for eid, n in counts.items() if n > 1]

    assert not repeated, (
        f"{len(repeated)} event(s) appear more than once in the audit chain "
        f"({repeated[:3]}); the subscriber is attached more than once, so "
        f"verify_chain certifies {len(lines)} records as intact while query "
        "reports the real count"
    )


def test_the_budget_gate_does_not_read_policies_from_disk(proxy, monkeypatch):
    """The per-request gate must not stat and re-parse policies.yaml.

    `load_policies` is uncached: it resolves the path, stats it, and on a hit
    reads and parses the YAML -- every call. This gate runs on every proxied
    request and is on by default, so calling it here would put disk I/O in the
    hot path of every install.

    Correctness matters more than the cost. `check_budget` evaluates against
    the snapshot the service loaded at construction, so a gate reading a fresh
    copy could see a positive budget that the decision below never applies --
    two answers to "what is the limit" from one request.
    """
    from entroly.governance import policy as policy_module

    calls: list[object] = []
    real = policy_module.load_policies
    monkeypatch.setattr(
        policy_module, "load_policies",
        lambda *a, **k: (calls.append(a), real(*a, **k))[1],
    )

    proxy._budget_enforced = True
    proxy._budget_refusal()

    assert calls == [], (
        f"the budget gate called load_policies {len(calls)}x for one request; "
        "policies.yaml is re-read and re-parsed on every proxied request"
    )


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


def test_governance_decisions_are_recorded_without_wiring_anything(tmp_path, monkeypatch):
    """The audit trail must record by default, not on request.

    `install_audit_subscriber` shipped and nothing called it, so the audit log
    stayed empty unless an operator attached it to the bus themselves. That is
    worse than having no audit trail: `govern audit verify` reports an intact
    chain over a log nothing ever wrote to, which reads as evidence of clean
    operation.

    Obtaining the authorization service now attaches it, so a denial recorded
    here proves the path is live end to end.
    """
    monkeypatch.setenv("ENTROLY_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))

    import entroly.governance.audit as audit_module
    from entroly.governance.authorization import get_authorization_service
    from entroly.governance.domain import RiskLevel

    monkeypatch.setattr(audit_module, "_global_log", None, raising=False)

    service = get_authorization_service()
    # A write at high risk is denied by the built-in read-only policy, which
    # emits a governance event the subscriber should persist.
    service.check("write", resource="src/app.py", risk_level=RiskLevel.HIGH)

    log = audit_module.get_audit_log()
    records = list(log.query(limit=20))
    assert records, (
        "no governance event was recorded; the audit subscriber is not "
        "attached, so `audit verify` would report an intact empty chain"
    )
    ok, detail = log.verify_chain()
    assert ok, f"recorded chain does not verify: {detail}"
