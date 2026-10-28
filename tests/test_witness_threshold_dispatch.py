"""Explicit thresholds must survive optional native analysis dispatch.

These are policy-fidelity checks over identical certificates, not a native
scorer parity test or a semantic risk guarantee. No model calls are needed.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json

import pytest

import entroly.witness as witness
from entroly.witness_calibration import ThresholdSet


CONTEXT = "The library stores JSON."
OUTPUT = "Entroly was founded in 1723."


def _result() -> witness.WitnessResult:
    # Nonzero evidence avoids the separate zero-evidence escalation rule.
    return witness.WitnessResult(
        output=OUTPUT,
        certificates=[witness.Certificate(
            claim_id=0,
            claim_text=OUTPUT,
            label="unknown",
            support_strength=0.25,
            contradiction_strength=0.0,
            evidence_adequacy=0.5,
            risk=0.55,
            proof_path=[],
        )],
        summary_score=0.45,
        n_grounded=0,
        n_unsupported=0,
        n_contradicted=0,
        n_unknown=1,
        latency_ms=0.0,
    )


def _native_payload(result: witness.WitnessResult, mode: str, profile: str) -> str:
    # Deliberately distinct: accepting this policy must not pass the fidelity
    # test by accident. Only native *certificates* should be used with overrides.
    return json.dumps({
        "witness": asdict(result),
        "policy": asdict(witness.WitnessRewrite(
            output="native-policy-sentinel",
            changed=True,
            mode=mode,
            profile=profile,
            flagged_count=1,
            suppressed_count=0,
            warned_count=1,
        )),
    })


@pytest.mark.parametrize("profile", ["chat", "qa", "summary", "auto"])
@pytest.mark.parametrize("mode", ["strict", "audit", "annotate", "off"])
@pytest.mark.parametrize("values", [
    (0.8, 0.9, 1.0),  # PASS at the fixture's score
    (0.2, 0.8, 0.9),  # HEDGE
    (0.1, 0.3, 0.8),  # WARN
    (0.0, 0.0, 0.0),  # SUPPRESS (except the existing summary downgrade)
])
def test_explicit_thresholds_use_shared_policy_with_native_certificates(
    monkeypatch, profile, mode, values,
):
    calls = []
    result = _result()

    def native(context, output, native_mode, native_profile, *settings):
        calls.append((context, output, native_mode, native_profile, settings))
        return _native_payload(result, native_mode, native_profile)

    monkeypatch.setattr(witness, "_rust_witness_analyze", native)
    thresholds = ThresholdSet(*values)
    analyzer = witness.WitnessAnalyzer(
        profile=profile, thresholds=thresholds,
        use_nli=False, use_local_nli=False, use_stave=False,
    )
    analyzed, rewrite = analyzer.analyze_and_rewrite(
        CONTEXT, OUTPUT, retrieved=["Additional evidence."], mode=mode,
    )
    assert analyzed == result
    assert rewrite == witness.apply_witness_policy(
        OUTPUT, result, mode=mode, profile=profile, thresholds=thresholds,
    )
    assert calls == [(
        CONTEXT + "\n\nAdditional evidence.", OUTPUT, "audit", profile,
        (0.62, 0.70, 0.72),
    )]


def test_no_override_preserves_opt_in_native_policy(monkeypatch):
    calls = []

    def native(context, output, mode, profile, *settings):
        calls.append(mode)
        return _native_payload(_result(), mode, profile)

    monkeypatch.setattr(witness, "_rust_witness_analyze", native)
    analyzer = witness.WitnessAnalyzer(
        profile="chat", use_nli=False, use_local_nli=False, use_stave=False,
    )
    _, rewrite = analyzer.analyze_and_rewrite(CONTEXT, OUTPUT, mode="strict")
    assert calls == ["strict"]
    assert rewrite.output == "native-policy-sentinel"


def test_explicit_thresholds_survive_native_failure(monkeypatch):
    calls = []

    def unavailable(*args):
        calls.append(args[2])
        raise RuntimeError("native backend unavailable")

    monkeypatch.setattr(witness, "_rust_witness_analyze", unavailable)
    monkeypatch.setattr(witness, "_rust_witness_claims", None)
    analyzer = witness.WitnessAnalyzer(
        profile="chat", thresholds=ThresholdSet(0, 0, 0),
        use_nli=False, use_local_nli=False, use_stave=False,
    )
    result, rewrite = analyzer.analyze_and_rewrite(CONTEXT, OUTPUT, mode="strict")
    assert calls == ["audit"]
    assert result.flagged()
    assert rewrite.suppressed_count == len(result.flagged())
    assert rewrite == witness.apply_witness_policy(
        OUTPUT, result, mode="strict", profile="chat", thresholds=analyzer.thresholds,
    )


@pytest.mark.parametrize("risk,action", [
    (0.0, "pass"),
    (0.3, "hedge"),
    (0.55, "warn"),
    (0.8, "suppress"),
    (1.0, "suppress"),
])
def test_native_dispatch_preserves_exact_override_boundaries(monkeypatch, risk, action):
    result = _result()
    result = replace(result, certificates=[replace(result.certificates[0], risk=risk)])
    calls = []

    def native(context, output, mode, profile, *settings):
        calls.append(mode)
        return _native_payload(result, mode, profile)

    monkeypatch.setattr(witness, "_rust_witness_analyze", native)
    analyzer = witness.WitnessAnalyzer(
        profile="chat", thresholds=ThresholdSet(0.3, 0.55, 0.8),
        use_nli=False, use_local_nli=False, use_stave=False,
    )
    analyzed, rewrite = analyzer.analyze_and_rewrite(CONTEXT, OUTPUT, mode="strict")
    assert analyzed == result
    assert calls == ["audit"]
    assert rewrite.suppressed_count == int(action == "suppress")
    assert rewrite.warned_count == int(action in {"hedge", "warn"})
    if action == "pass":
        assert rewrite.output == OUTPUT and not rewrite.changed
    elif action == "hedge":
        assert "[Entroly WITNESS notes (unverified" in rewrite.output
    elif action == "warn":
        assert "[Entroly WITNESS warning:" in rewrite.output
    else:
        assert OUTPUT not in rewrite.output


@pytest.mark.parametrize("profile", ["chat", "qa"])
@pytest.mark.parametrize("value", [0.0, 1.0])
def test_preinstalled_native_backend_honors_explicit_policy(monkeypatch, profile, value):
    core = pytest.importorskip("entroly_core")
    native = getattr(core, "py_witness_analyze", None)
    if native is None:
        pytest.skip("installed engine does not expose WITNESS")
    completed = []

    def recorded_native(*args):
        payload = native(*args)
        completed.append(payload)
        return payload

    monkeypatch.setattr(witness, "_rust_witness_analyze", recorded_native)
    thresholds = ThresholdSet(value, value, value)
    analyzer = witness.WitnessAnalyzer(
        profile=profile, thresholds=thresholds,
        use_nli=False, use_local_nli=False, use_stave=False,
    )

    def unexpected_fallback(*args, **kwargs):
        pytest.fail("installed-native diagnostic must not fall back to Python scoring")

    monkeypatch.setattr(analyzer, "_certify_claim", unexpected_fallback)
    result, rewrite = analyzer.analyze_and_rewrite(CONTEXT, OUTPUT, mode="strict")
    assert len(completed) == 1, "native analysis must complete successfully"
    assert result == witness._witness_result_from_dict(json.loads(completed[0])["witness"])
    assert result.certificates, "fixture must exercise a native certificate"
    assert rewrite == witness.apply_witness_policy(
        OUTPUT, result, mode="strict", profile=profile, thresholds=thresholds,
    )
