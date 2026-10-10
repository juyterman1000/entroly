import pytest

from benchmarks.context_assurance import run_benchmark, source_digest


def test_frozen_local_controls_and_long_turn_instrumentation():
    pytest.importorskip(
        "tiktoken", reason="frozen protocol requires exact o200k_base counting"
    )
    result = run_benchmark(performance_samples=2)
    assert result["false_passes_declared_obligations"] == 0
    assert result["inexact_recoveries"] == 0
    assert result["provider_observed_usage"] is None
    assert result["provider_cost"] is None
    assert result["risk_calibration"].startswith("unavailable")
    assert {row["turns"] for row in result["summaries"]} == {20, 50, 100}
    for report in result["summaries"]:
        assert report["model_self_divergence"] == 0
        assert report["risk_bound"] is None
        if report["strategy"] in {"full", "receipt_exact_recovery"}:
            assert report["decision_divergence_regret"] == 0
            assert report["cumulative_missed_context_debt"] == 0
        else:
            assert report["decision_divergence_regret"] == report["trials"]
            assert report["cumulative_missed_context_debt"] == report["trials"]


def test_protocol_refuses_to_relabel_heuristic_tokens_as_o200k(monkeypatch):
    import benchmarks.context_assurance as harness

    monkeypatch.setattr(harness, "_encoding", lambda: None)
    with pytest.raises(RuntimeError, match="tokenizer"):
        harness.run_benchmark(performance_samples=1)


def test_protocol_identity_survives_windows_checkout_line_endings():
    original = b'{\n  "budget":20\n}\n'
    assert source_digest(original) == source_digest(original.replace(b"\n", b"\r\n"))
    assert source_digest(original) != source_digest(original.replace(b"20", b"21"))
