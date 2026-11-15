"""Commit-point checks for the signed Work Graph handoff acceptance."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from entroly.context_receipts.models import stable_json
from entroly.receipt_attestation import AttestationKey, AttestationLog
from entroly import work_graph_handoff_acceptance as acceptance


@pytest.mark.parametrize("change", ["graph", "expiry"])
def test_handoff_acceptance_rechecks_at_commit(tmp_path, monkeypatch, change):
    pytest.importorskip("cryptography")
    revision = (1, "graph")
    expected = {"audience": "test", "successor_id": "successor"}
    ref = "handoff:" + "a" * 64
    clock_ms = 1000
    graph_binding = revision

    class Store:
        repo_dir = tmp_path

        @contextmanager
        def lock(self):
            yield

        def _load_unlocked(self):
            return SimpleNamespace(
                revision=graph_binding[0], graph_commitment=graph_binding[1]
            )

    attempts = 0

    def challenge(*_):
        nonlocal attempts, clock_ms, graph_binding
        attempts += 1
        if attempts == 2:
            if change == "graph":
                graph_binding = (2, "later")
            else:
                clock_ms = 3000
        return expected, revision

    monkeypatch.setattr(acceptance, "_challenge_with_graph", challenge)
    monkeypatch.setattr(acceptance.time, "time_ns", lambda: clock_ms * 1_000_000)
    key = AttestationKey.generate()
    monkeypatch.setenv("ENTROLY_HANDOFF_ISSUER", "fixture")
    monkeypatch.setenv("ENTROLY_HANDOFF_PUBLIC_KEY", key.public_hex())
    path = tmp_path / "signed.json"
    assertion = AttestationLog(key).append(
        {
            **expected,
            "issuer": "fixture",
            "issued_at_ms": 900,
            "expires_at_ms": 2000,
        }
    )
    path.write_text(stable_json(assertion.to_dict()), encoding="utf-8")
    message = "revision changed" if change == "graph" else "expired during"
    with pytest.raises(ValueError, match=message):
        acceptance.accept(tmp_path, Store(), ref, path)
    assert not list((tmp_path / "handoffs").glob("*.accepted"))

    graph_binding, clock_ms = revision, 1000
    receipt = acceptance.accept(tmp_path, Store(), ref, path)
    assert (receipt["issued_at_ms"], receipt["expires_at_ms"]) == (900, 2000)
    assert acceptance.accept(tmp_path, Store(), ref, path) == receipt
