"""Fail-closed audit for the groundedness/verification surface.

CLAUDE.md trust invariant: "Fail-closed verification: WITNESS, RAVS, and
native-status checks must degrade safely, not silently claim confidence."

These tests pin the property that a check which did not run is never
reported as a check that passed. Two regressions are covered:

1. `verify_response` (MCP tool, entroly/server.py) read every signal back
   out of its own result dict with `.get("risk_score", 0)`. Each signal was
   raising an AttributeError, so each was scored as ZERO risk — i.e. as
   positive evidence of safety — and every response, including fabricated
   ones, came back `fused_risk=0.0, verdict="pass"`.

2. `VerificationEngine._check_staleness` swallowed a `ValueError` from an
   unparseable `last_checked`, so `check_belief` returned status="verified"
   with no issues for a belief whose freshness was never established.
"""

from __future__ import annotations

import asyncio
import json
import textwrap
from datetime import datetime, timedelta, timezone

import pytest

from entroly.vault import VaultConfig, VaultManager
from entroly.verification_engine import VerificationEngine


# ── 1. verify_response must not certify what it did not check ────────

CONTEXT = "The build fails when the incremental cache is enabled."
FABRICATED = "The build is orchestrated by the Zylophane quantum scheduler at 4400 THz."


def _call_verify_response(monkeypatch, tmp_path, **kwargs) -> dict:
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "entroly"))
    from entroly.server import create_mcp_server

    mcp, _engine = create_mcp_server()

    async def _run():
        return await mcp.call_tool("verify_response", kwargs)

    out = asyncio.run(_run())
    raw = out[0][0].text if isinstance(out, tuple) else out[0].text
    return json.loads(raw)


def test_verify_response_flags_a_fabricated_response(monkeypatch, tmp_path):
    """A response with no support in context must not come back "pass"."""
    report = _call_verify_response(
        monkeypatch, tmp_path, response=FABRICATED, context=CONTEXT
    )
    assert report["verdict"] != "pass", report
    assert report["fused_risk"] > 0.15, report


def test_verify_response_witness_signal_actually_runs(monkeypatch, tmp_path):
    """The primary WITNESS signal must produce a score, not an exception.

    Guards the attribute contract between server.py and witness.py:
    WitnessResult exposes `certificates`/`summary_score` and Certificate
    exposes `claim_text`/`risk`/`label` — not `total_claims`/`text`/`score`.
    """
    report = _call_verify_response(
        monkeypatch, tmp_path, response=FABRICATED, context=CONTEXT
    )
    witness = report["witness"]
    assert "status" not in witness, f"WITNESS did not run: {witness}"
    assert "witness" in report["signals_scored"], report["signals_scored"]
    assert 0.0 <= witness["risk_score"] <= 1.0


def test_verify_response_witness_risk_is_complement_of_groundedness(
    monkeypatch, tmp_path
):
    """Polarity guard: summary_score is groundedness, risk is 1 - it.

    Feeding summary_score straight into the 0.80-weighted risk term
    inverts the tool — a perfectly grounded response scores ~0.8 risk.
    """
    report = _call_verify_response(
        monkeypatch, tmp_path, response=CONTEXT, context=CONTEXT
    )
    witness = report["witness"]
    assert "status" not in witness, f"WITNESS did not run: {witness}"
    assert witness["risk_score"] == pytest.approx(
        1.0 - witness["groundedness"], abs=1e-4
    )
    # A verbatim restatement of the context is grounded -> low risk.
    assert witness["risk_score"] < 0.15, witness
    assert report["verdict"] == "pass", report


def test_verify_response_unavailable_signal_is_not_scored_as_zero_risk(
    monkeypatch, tmp_path
):
    """A raising signal must be excluded, never zero-filled.

    Zero-filling is the fail-open: it makes "the check crashed" arithmetically
    identical to "the check found no risk".
    """
    import entroly.ravs.epr as epr_mod

    def _boom(*a, **k):
        raise RuntimeError("epr exploded")

    monkeypatch.setattr(epr_mod, "compute_epr", _boom)

    report = _call_verify_response(
        monkeypatch, tmp_path, response=FABRICATED, context=CONTEXT
    )
    assert "epr" in report["signals_unavailable"], report
    assert "epr" not in report["signals_scored"], report
    assert "epr" not in report["signal_weights"], report
    # The broken signal must not have dragged the fused risk toward safe.
    assert report["verdict"] != "pass", report


def test_verify_response_without_witness_cannot_return_pass(monkeypatch, tmp_path):
    """Losing the primary groundedness signal must block a "pass" verdict."""
    import entroly.witness as witness_mod

    class _Broken:
        def __init__(self, *a, **k):
            raise RuntimeError("witness engine unavailable")

    monkeypatch.setattr(witness_mod, "WitnessAnalyzer", _Broken)

    report = _call_verify_response(
        monkeypatch, tmp_path, response=CONTEXT, context=CONTEXT
    )
    assert report["witness"]["status"] == "unavailable", report
    assert report["verdict"] == "unverified", report
    assert report["verdict"] != "pass"
    assert "witness" in report["signals_unavailable"], report
    assert "not" in report["recommendation"].lower()


def test_verify_response_with_no_working_signal_reports_maximum_risk(
    monkeypatch, tmp_path
):
    """If nothing ran, unknown is not safe."""
    import entroly.witness as witness_mod

    class _Broken:
        def __init__(self, *a, **k):
            raise RuntimeError("witness engine unavailable")

    monkeypatch.setattr(witness_mod, "WitnessAnalyzer", _Broken)

    report = _call_verify_response(
        monkeypatch, tmp_path, response=FABRICATED, context=CONTEXT
    )
    if not report["signals_scored"]:
        assert report["fused_risk"] == 1.0, report
    assert report["verdict"] != "pass", report


# ── 2. A belief whose freshness check could not run is not "verified" ──


def _write_belief(beliefs_dir, name: str, claim_id: str, last_checked: str) -> None:
    beliefs_dir.mkdir(parents=True, exist_ok=True)
    (beliefs_dir / f"{name}.md").write_text(
        textwrap.dedent(
            f"""\
            ---
            claim_id: {claim_id}
            entity: {name}
            status: verified
            confidence: 0.9
            last_checked: {last_checked}
            sources:
              - {name}.py:1
            ---

            The {name} module does the thing.
            """
        ),
        encoding="utf-8",
    )


@pytest.fixture()
def engine_and_beliefs(tmp_path):
    vault = VaultManager(VaultConfig(base_path=str(tmp_path / "vault")))
    vault.ensure_structure()
    return VerificationEngine(vault), vault.config.path / "beliefs"


@pytest.mark.parametrize(
    "bad_timestamp",
    ["not-a-timestamp", "2024-13-45T99:99:99+00:00", "yesterday", "0"],
)
def test_unparseable_last_checked_is_not_reported_verified(
    engine_and_beliefs, bad_timestamp
):
    engine, beliefs_dir = engine_and_beliefs
    _write_belief(beliefs_dir, "corrupt", "c_corrupt", bad_timestamp)

    result = engine.check_belief("c_corrupt")
    assert result["status"] != "verified", result
    assert result["issues"], result


def test_unparseable_last_checked_appears_in_the_stale_report(engine_and_beliefs):
    engine, beliefs_dir = engine_and_beliefs
    _write_belief(beliefs_dir, "corrupt", "c_corrupt", "not-a-timestamp")

    report = engine.full_verification_pass()
    assert report.total_beliefs_checked == 1
    assert [s.claim_id for s in report.stale_beliefs] == ["c_corrupt"], report.stale_beliefs


def test_parseable_timestamps_still_classify_correctly(engine_and_beliefs):
    """The fix must not turn every belief stale."""
    engine, beliefs_dir = engine_and_beliefs
    now = datetime.now(timezone.utc)
    _write_belief(beliefs_dir, "fresh", "c_fresh", now.isoformat())
    _write_belief(
        beliefs_dir, "old", "c_old", (now - timedelta(days=30)).isoformat()
    )

    assert engine.check_belief("c_fresh")["status"] == "verified"
    assert engine.check_belief("c_old")["status"] == "needs_attention"


# ── Signals 2-4: correct API, and a reason that is true ────────────────────


def test_unavailable_signals_report_a_true_reason_not_an_attribute_error():
    """A signal that cannot measure must say why, accurately.

    All three of ECE/EPR/spectral had drifted out of API compatibility --
    `ece.evaluate` does not exist (it is `evaluate_uncertainty`), and EPRSignal
    and SpectralSignal are dataclasses, not dicts, so `.get(...)` raised. The
    reported reason was therefore an AttributeError string, which reads as a
    trivial code bug rather than the real situation: these signals need token
    logprobs or extractable entities, and this tool receives plain text.
    """
    from entroly.ravs.ece import EpistemicCascadeEngine
    from entroly.ravs.epr import compute_epr
    from entroly.ravs.spectral import compute_spectral_consistency

    # The real APIs exist under these names.
    assert hasattr(EpistemicCascadeEngine, "evaluate_uncertainty")
    assert not hasattr(EpistemicCascadeEngine, "evaluate"), (
        "an `evaluate` method appeared; re-check which one verify_response calls"
    )

    # They return dataclasses, so `.get` is not available -- the original bug.
    epr = compute_epr("some response text")
    assert not hasattr(epr, "get")
    assert hasattr(epr, "has_logprobs")

    spec = compute_spectral_consistency("ctx", "resp")
    assert not hasattr(spec, "get")
    assert hasattr(spec, "n_ctx_entities") and hasattr(spec, "score")


def test_spectral_argument_order_is_context_then_response():
    """`compute_spectral_consistency(context, response)` is not symmetric.

    verify_response passed them reversed. Measured on one pair the two orders
    gave 0.2000 and 0.5000, so the swap silently changed the number -- latent
    only because the call was raising before it could be used.
    """
    from entroly.ravs.spectral import compute_spectral_consistency

    context = (
        "The RefundProcessor validates DeclinedTransaction objects and "
        "PaymentGateway.charge calls StripeAdapter.submit."
    )
    response = "ZylophaneScheduler reconciles ParallelUniverseBatch records."

    forward = compute_spectral_consistency(context, response)
    reversed_ = compute_spectral_consistency(response, context)
    assert (forward.n_ctx_entities, forward.n_resp_entities) != (
        reversed_.n_ctx_entities,
        reversed_.n_resp_entities,
    ), "the two orders are indistinguishable here; pick a pair that separates them"
