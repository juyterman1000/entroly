"""
Tests for Entroly Enterprise Governance Layer.
==============================================

Verifies:
  1. AgentIdentity creation, serialization, and HMAC token signing/verification
  2. Policy evaluation (allow, deny, path restriction, approval required)
  3. AuthorizationService end-to-end gating
  4. GovernanceAuditLog tamper-evident chain verification
  5. GovernanceEventBus publishing and subscription
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from entroly.governance.domain import (
    AgentIdentity,
    GovernancePolicy,
    Policy,
    RiskLevel,
)
from entroly.governance.identity import (
    compute_identity_token,
    create_identity,
    verify_token,
)
from entroly.governance.policy import (
    evaluate,
)
from entroly.governance.authorization import (
    AuthorizationService,
)
from entroly.governance.audit import (
    GovernanceAuditLog,
)
from entroly.governance.events import (
    GovernanceEvent,
    GovernanceEventBus,
    IdentityResolvedEvent,
    TracingContext,
)


def test_agent_identity_token_roundtrip():
    secret_key = "test-secret-key-123"
    identity = create_identity(
        agent_id="agent-007",
        scopes=["read", "write:tests/*"],
        key=secret_key,
    )

    assert identity.agent_id == "agent-007"
    assert identity.identity_token.startswith("egov1:")
    assert verify_token(identity, secret_key) is True

    # Tampered key should fail verification
    assert verify_token(identity, "wrong-key") is False


def test_agent_identity_anonymous():
    anon = AgentIdentity.anonymous()
    assert anon.agent_id == "anonymous"
    assert anon.has_scope("read") is True
    assert anon.has_scope("write:src/main.rs") is False


def test_policy_allow_and_deny():
    policy = GovernancePolicy(
        id="policy-001",
        name="Standard Developer Policy",
        allowed_scopes=frozenset({"read", "write:tests/*"}),
        denied_paths=("*.key", "secrets/**", "prod/*"),
        max_risk_level=RiskLevel.MEDIUM,
    )

    dev_identity = create_identity(
        agent_id="dev-1",
        scopes=["read", "write:tests/*"],
    )

    # 1. Allowed write
    dec1 = evaluate(
        dev_identity,
        "write:tests/*",
        resource="tests/test_foo.py",
        policies=[policy],
    )
    assert dec1.allowed is True

    # 2. Denied because path matches denied_paths
    dec2 = evaluate(
        dev_identity,
        "read",
        resource="secrets/token.key",
        policies=[policy],
    )
    assert dec2.allowed is False

    # 3. Denied because risk level exceeded
    dec3 = evaluate(
        dev_identity,
        "write:tests/*",
        resource="tests/test_foo.py",
        risk_level=RiskLevel.HIGH,
        policies=[policy],
    )
    assert dec3.allowed is False


def test_policy_requires_approval():
    policy = GovernancePolicy(
        id="policy-crit",
        name="Production Policy",
        allowed_scopes=frozenset({"read", "deploy"}),
        requires_approval_for=frozenset({"deploy"}),
    )
    ops_identity = create_identity(
        agent_id="ops-agent",
        scopes=["admin"],
    )

    dec = evaluate(
        ops_identity,
        "deploy",
        resource="deploy/values.yaml",
        policies=[policy],
    )
    assert dec.requires_approval is True


def test_authorization_service_gate():
    policy = GovernancePolicy(
        id="p1",
        name="Policy1",
        allowed_scopes=frozenset({"read", "write:scratch/*"}),
    )
    identity = create_identity(agent_id="bot-1", scopes=["read", "write:scratch/*"])

    auth_service = AuthorizationService(
        identity=identity,
        policies=[policy],
    )

    # Authorized operation
    dec1 = auth_service.check("read", resource="README.md")
    assert dec1.allowed is True

    # Denied operation
    dec2 = auth_service.check("write:src/critical.py", resource="src/critical.py")
    assert dec2.allowed is False


def test_audit_chain_tamper_detection():
    with tempfile.TemporaryDirectory() as tmpdir:
        audit_dir = Path(tmpdir) / "audit"
        log = GovernanceAuditLog(audit_dir=audit_dir)

        log.append(
            event_id="evt-1",
            event_type="agent.registered",
            agent_id="agent-1",
            payload={"action": "register", "team": "core"},
            allowed=True,
        )
        log.append(
            event_id="evt-2",
            event_type="policy.evaluated",
            agent_id="agent-1",
            payload={"scope": "read:*", "verdict": "allow"},
            allowed=True,
        )
        log.append(
            event_id="evt-3",
            event_type="action.authorized",
            agent_id="agent-1",
            payload={"resource": "src/engine.py"},
            allowed=True,
        )

        # Verification passes on valid log
        valid, msg = log.verify_chain()
        assert valid is True
        assert "intact" in msg.lower()

        # Tampering with middle entry in jsonl
        lines = log.jsonl_path.read_text(encoding="utf-8").splitlines()
        import json
        entry1 = json.loads(lines[1])
        entry1["payload"]["verdict"] = "tampered_value"
        lines[1] = json.dumps(entry1)
        log.jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Verification fails!
        valid_tampered, err_msg = log.verify_chain()
        assert valid_tampered is False
        assert "chain broken" in err_msg.lower()


def test_governance_event_bus():
    bus = GovernanceEventBus()
    received = []

    def handler(evt):
        received.append(evt)

    bus.subscribe("identity.resolved", handler)

    bus.emit(IdentityResolvedEvent(
        tracing=TracingContext(agent_id="agent-1"),
        payload={"msg": "hello"},
    ))

    assert len(received) == 1
    assert received[0].tracing.agent_id == "agent-1"
