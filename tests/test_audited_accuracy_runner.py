from __future__ import annotations

from bench import audited_accuracy_runner as runner
from bench.benchmark_evidence import ProviderUsage


def test_runner_separates_emitted_and_provider_tokens(monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "select_with_audit",
        lambda _fragments, budget, _query: {
            "selected": [
                {
                    "source": "sample:x",
                    "content": "Rhijn",
                    "token_count": 2,
                    "source_spans": [],
                }
            ],
            "candidates": [],
            "metrics": {"scope": "candidate_units", "verdict": "sufficient"},
            "emitted_tokens": 2,
            "selection_mode": "atomic_audited",
        },
    )

    def evaluator(_query: str, context: str, _model: str, _seed: int):
        return runner.EvaluationResult(
            correct="Rhijn" in context,
            response="Rhijn" if "Rhijn" in context else "wrong",
            usage=ProviderUsage(input_tokens=50, output_tokens=5, total_tokens=55),
        )

    sample = runner.QASample(
        sample_id="x",
        dataset="squad",
        split="holdout",
        query="What is the Dutch name?",
        context="The Dutch name is Rhijn.",
        answers=("Rhijn",),
    )
    evidence, _ = runner.run_sample(
        sample, budget=10, model="model", seed=0, evaluator=evaluator
    )
    assert evidence.emitted_context_tokens == 2
    assert evidence.prompt_input_tokens == 50
    assert evidence.provider_total_tokens == 55
    assert evidence.answer_present_post_trim
