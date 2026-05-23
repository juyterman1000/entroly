from __future__ import annotations

import json

import pytest

from starlette.testclient import TestClient
from starlette.applications import Starlette
from starlette.routing import Route

from entroly.proxy import (
    PromptCompilerProxy,
    _extract_text_from_sse,
    _witness_feedback_route,
    _witness_list,
    _witness_train_route,
)
from entroly.proxy_config import ProxyConfig
from entroly.witness import (
    WitnessAnalyzer,
    apply_witness_policy,
    extract_claims,
    select_evidence_windows,
)


def test_extract_claims_skips_questions() -> None:
    claims = extract_claims("Which one is correct? Delhi is the headquarters.")
    assert [claim.text for claim in claims] == ["Delhi is the headquarters."]


def test_grounded_short_answer_is_retained() -> None:
    analyzer = WitnessAnalyzer()
    result = analyzer.analyze(
        "The Oberoi Group is a hotel company with its head office in Delhi.",
        "Delhi",
    )
    assert result.certificates[0].label == "grounded"
    assert result.flagged() == []


def test_invented_claim_is_actionable_unknown_not_grounded() -> None:
    analyzer = WitnessAnalyzer()
    result = analyzer.analyze(
        "The module contains fetch_user and delete_user functions.",
        "Call update_user_profile to update the email address.",
    )
    assert result.certificates[0].label in {"unknown", "unsupported"}
    assert result.flagged()


def test_comparative_contradiction_overrides_quote_tokens() -> None:
    analyzer = WitnessAnalyzer()
    context = (
        "Arthur's Magazine (1844-1846) was an American literary periodical. "
        "First for Women is a woman's magazine published by Bauer Media Group, "
        "started in 1989.\n\n"
        "Question: Which magazine was started first Arthur's Magazine or First for Women?"
    )
    result = analyzer.analyze(context, "First for Women was started first.")
    assert result.certificates[0].label == "contradicted"


def test_evidence_adequacy_requires_claim_slots() -> None:
    windows, adequacy = select_evidence_windows(
        "Version 0.19.2 was released with 38 agent wrappers.",
        "There are 38 agent wrappers in the release.",
    )
    assert windows
    assert adequacy > 0.7


def test_strict_policy_suppresses_unsupported_claim() -> None:
    analyzer = WitnessAnalyzer()
    output = "The Oberoi Group has its head office in Delhi. Paris is the capital of Germany."
    result = analyzer.analyze(
        "The Oberoi Group has its head office in Delhi.",
        output,
    )
    rewrite = apply_witness_policy(output, result, mode="strict")
    assert rewrite.changed
    assert "Paris is the capital of Germany" not in rewrite.output
    assert "Delhi" in rewrite.output


def test_extract_claims_handles_bullets_tables_and_code_refs() -> None:
    claims = extract_claims(
        "- Alpha launched in 2024.\n"
        "| Name | Value |\n"
        "| Beta | 42 |\n"
        "`load_config()` reads config.yaml."
    )
    assert any("Alpha launched" in claim.text for claim in claims)
    assert any("Beta" in claim.text and "42" in claim.text for claim in claims)
    assert any(claim.kind == "code_ref" for claim in claims)


def test_summary_profile_warns_unknown_instead_of_suppressing() -> None:
    analyzer = WitnessAnalyzer(profile="summary")
    result, rewrite = analyzer.analyze_and_rewrite(
        "The report says revenue increased in 2024.",
        "Revenue increased in 2024. The report also confirms a new Tokyo office.",
        mode="strict",
    )
    assert result.flagged()
    assert "Tokyo office" in rewrite.output
    assert rewrite.suppressed_count == 0
    assert rewrite.warned_count >= 1


def test_code_profile_suppresses_unknown_claims() -> None:
    analyzer = WitnessAnalyzer(profile="code")
    _, rewrite = analyzer.analyze_and_rewrite(
        "The module contains fetch_user().",
        "Call update_user_profile() to update the email address.",
        mode="strict",
    )
    assert rewrite.changed
    assert "update_user_profile" not in rewrite.output
    assert rewrite.suppressed_count >= 1


def test_proxy_witness_gateway_rewrites_openai_response() -> None:
    proxy = PromptCompilerProxy(object(), ProxyConfig(witness_mode="strict"))
    content = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "The Oberoi Group has its head office in Delhi. Paris is the capital of Germany.",
                }
            }
        ]
    }

    updated, headers = proxy._apply_witness_gateway(
        content,
        "The Oberoi Group has its head office in Delhi.",
    )

    rewritten = updated["choices"][0]["message"]["content"]
    assert headers["X-Entroly-Witness"] == "flagged"
    assert headers["X-Entroly-Witness-Id"] in proxy._witness_certificates
    assert headers["X-Entroly-Witness-Rewritten"] == "true"
    assert "Paris is the capital of Germany" not in rewritten
    assert "entroly_witness" not in updated


def test_proxy_witness_preserves_structured_json_output() -> None:
    proxy = PromptCompilerProxy(object(), ProxyConfig(witness_mode="strict"))
    content = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"answer": "Paris is the capital of Germany."}),
                }
            }
        ]
    }

    updated, headers = proxy._apply_witness_gateway(content, "The context mentions Berlin.")

    assert json.loads(updated["choices"][0]["message"]["content"]) == {
        "answer": "Paris is the capital of Germany."
    }
    assert headers.get("X-Entroly-Witness-Rewrite-Skipped") == "structured-output"


def test_proxy_witness_can_embed_when_explicitly_enabled() -> None:
    proxy = PromptCompilerProxy(object(), ProxyConfig(witness_mode="audit", witness_embed=True))
    content = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Paris is the capital of Germany.",
                }
            }
        ]
    }

    updated, headers = proxy._apply_witness_gateway(content, "The context mentions Berlin.")

    assert headers["X-Entroly-Witness"] == "flagged"
    assert "entroly_witness" in updated
    assert updated["entroly_witness"]["id"] == headers["X-Entroly-Witness-Id"]


def test_proxy_witness_strict_stream_sse_is_provider_parseable() -> None:
    proxy = PromptCompilerProxy(object(), ProxyConfig(witness_mode="strict"))
    sse = proxy._format_witness_sse(
        "openai",
        {"model": "gpt-4o-mini"},
        "The Oberoi Group has its head office in Delhi.",
    )

    assert b"data: " in sse
    assert b"[DONE]" in sse
    assert _extract_text_from_sse(sse) == "The Oberoi Group has its head office in Delhi."


def test_rust_witness_core_when_available() -> None:
    try:
        import entroly_core  # type: ignore
    except Exception:
        pytest.skip("entroly_core not installed")
    if not hasattr(entroly_core, "py_witness_analyze"):
        pytest.skip("installed entroly_core does not include Rust WITNESS")

    payload = json.loads(entroly_core.py_witness_analyze(
        "The Oberoi Group has its head office in Delhi.",
        "The Oberoi Group has its head office in Delhi. Paris is the capital of Germany.",
        "strict",
    ))

    assert payload["witness"]["n_grounded"] == 1
    assert payload["policy"]["changed"] is True
    assert "Paris is the capital of Germany" not in payload["output"]


def test_witness_certificate_feedback_endpoint() -> None:
    proxy = PromptCompilerProxy(object(), ProxyConfig(witness_mode="strict"))
    app = Starlette(routes=[
        Route("/witness", _witness_list),
        Route("/witness/{witness_id}/feedback", _witness_feedback_route, methods=["POST"]),
    ])
    app.state.proxy = proxy
    client = TestClient(app)
    _, headers = proxy._apply_witness_gateway(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Paris is the capital of Germany.",
                    }
                }
            ]
        },
        "The context mentions Berlin.",
    )
    witness_id = headers["X-Entroly-Witness-Id"]

    response = client.post(f"/witness/{witness_id}/feedback", json={"verdict": "false_positive"})

    assert response.status_code == 200
    assert response.json()["verdict"] == "false_positive"
    assert client.get("/witness").json()["feedback"]["false_positive"] == 1


def test_witness_training_endpoint_records_honest_label(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ENTROLY_WITNESS_MODEL_PATH", str(tmp_path / "risk_model.json"))
    proxy = PromptCompilerProxy(object(), ProxyConfig(witness_mode="audit"))
    app = Starlette(routes=[
        Route("/witness/train", _witness_train_route, methods=["POST"]),
    ])
    app.state.proxy = proxy
    client = TestClient(app)

    response = client.post("/witness/train", json={
        "context": "Einstein developed relativity in 1915.",
        "output": "Einstein developed relativity in 1915.",
        "label": "ci_passed",
        "profile": "benchmark_qa",
    })

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["training"]["y"] == 0
    assert (tmp_path / "risk_model.json").exists()
