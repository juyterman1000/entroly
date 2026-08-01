from __future__ import annotations

from bench import accuracy


def test_entirely_fitting_context_is_identity() -> None:
    text = "small context"
    assert accuracy._entroly_compress(text, 100, "context") == text


def test_run_mode_separates_emitted_context_from_provider_usage(monkeypatch) -> None:
    monkeypatch.setattr(
        accuracy,
        "_call_llm_detailed",
        lambda *_args, **_kwargs: accuracy.LLMCallResult(
            text="answer",
            input_tokens=50,
            output_tokens=5,
            total_tokens=55,
            latency_ms=7.5,
        ),
    )
    items = [
        {
            "context": "x" * 40,
            "question": "question",
            "expected": "answer",
            "metadata": {},
        }
    ]
    result = accuracy._run_mode(items, "needle", "gpt-4o-mini", "baseline", None)
    assert result.avg_emitted_context_tokens == 10
    assert result.avg_tokens == 10
    assert result.avg_provider_input_tokens == 50
    assert result.avg_provider_output_tokens == 5
    assert result.avg_provider_total_tokens == 55
    assert result.details[0]["emitted_context_tokens"] == 10
    assert result.details[0]["provider_total_tokens"] == 55


def test_compressed_context_savings_do_not_use_provider_total(monkeypatch) -> None:
    monkeypatch.setattr(
        accuracy,
        "_compress_messages_modal",
        lambda _messages, _budget, **_kwargs: [
            {"role": "system", "content": "Context:\nshort"},
            {"role": "user", "content": "question"},
        ],
    )
    monkeypatch.setattr(
        accuracy,
        "_call_llm_detailed",
        lambda *_args, **_kwargs: accuracy.LLMCallResult(
            text="answer",
            input_tokens=1_000,
            output_tokens=10,
            total_tokens=1_010,
            latency_ms=1.0,
        ),
    )
    result = accuracy._run_mode(
        [{"context": "x" * 400, "question": "question", "expected": "answer"}],
        "needle",
        "gpt-4o-mini",
        "entroly",
        10,
    )
    assert result.avg_tokens == 2
    assert result.avg_provider_total_tokens == 1_010
