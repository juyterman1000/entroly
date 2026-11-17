"""Regression: strict authorization must contribute to governance statistics."""
import pytest

from entroly.governance.authorization import AuthorizationService, PolicyDeniedError
from entroly.governance.domain import AgentIdentity, Policy


def test_require_permission_counts_allow_and_deny():
    identity = AgentIdentity.anonymous()
    policy = Policy(id="test-read-only", allowed_scopes=frozenset({"read"}))
    service = AuthorizationService(identity=identity, policies=[policy])

    service.require_permission("read")
    with pytest.raises(PolicyDeniedError):
        service.require_permission("write")

    assert service.stats.checks == 2
    assert service.stats.allowed == 1
    assert service.stats.denied == 1


def test_require_permission_does_not_double_count():
    identity = AgentIdentity.anonymous()
    policy = Policy(id="test-read-only", allowed_scopes=frozenset({"read"}))
    service = AuthorizationService(identity=identity, policies=[policy])

    service.check("read")
    service.require_permission("read")

    assert service.stats.checks == 2
    assert service.stats.allowed == 2
