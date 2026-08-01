from __future__ import annotations

import json
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from entroly.proxy_assurance import (
    ProxyAssuranceController,
    load_calibration_profile,
)
from entroly.sufficiency_calibration import CalibrationProfile


def _body():
    return {
        "model": "example",
        "messages": [
            {"role": "system", "content": "policy" * 200},
            {"role": "user", "content": "old evidence" * 200},
            {"role": "assistant", "content": "recent answer"},
            {"role": "user", "content": "current question"},
        ],
    }


def test_off_mode_is_exact_noop() -> None:
    body = _body()
    result = ProxyAssuranceController().apply(
        body, query="question", context_window=1000
    )
    assert result.body == body
    assert result.headers == {}
    assert not result.enabled


def test_multimodal_shape_fails_closed_without_mutation() -> None:
    body = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        ]
    }
    result = ProxyAssuranceController(mode="candidate_units").apply(
        body, query="hello", context_window=1000
    )
    assert result.body == body
    assert result.headers["X-Entroly-Assurance-Decision"] == (
        "BYPASS_MULTIMODAL_OR_STRUCTURED"
    )
    assert result.headers["X-Entroly-Assurance-Changed"] == "false"


def test_candidate_mode_emits_bounded_receipt_headers(monkeypatch) -> None:
    from entroly import proxy_assurance

    fake = SimpleNamespace(
        messages=tuple(_body()["messages"][-2:]),
        receipt={
            "decision": "COMPRESSED_CERTIFIED",
            "attempts": [
                {
                    "certificate_scope": "candidate_units",
                    "certificate_verdict": "sufficient",
                }
            ],
        },
        original_tokens=1000,
        delivered_tokens=200,
        budget_compliant=True,
        changed=True,
    )
    monkeypatch.setattr(
        proxy_assurance,
        "compress_messages_assured",
        lambda *_args, **_kwargs: fake,
    )
    result = ProxyAssuranceController(
        mode="candidate_units", budget_tokens=300
    ).apply(_body(), query="question", context_window=1000)
    assert result.changed
    assert result.headers["X-Entroly-Assurance-Certificate"] == (
        "sufficient/candidate_units"
    )
    assert result.headers["X-Entroly-Assurance-Tokens"] == "1000->200"
    assert len(result.body["messages"]) == 2


def test_runtime_error_returns_original_and_bounded_error(monkeypatch) -> None:
    from entroly import proxy_assurance

    monkeypatch.setattr(
        proxy_assurance,
        "compress_messages_assured",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("secret raw data")),
    )
    body = _body()
    result = ProxyAssuranceController(mode="semantic").apply(
        body, query="question", context_window=1000
    )
    assert result.body == body
    assert result.headers["X-Entroly-Assurance-Decision"] == "BYPASS_RUNTIME_ERROR"
    assert result.headers["X-Entroly-Assurance-Error"] == "ValueError"
    assert "secret" not in str(result.headers)


def test_from_config_records_invalid_profile_without_raising(tmp_path) -> None:
    path = tmp_path / "profile.json"
    path.write_text("not-json", encoding="utf-8")
    controller = ProxyAssuranceController.from_config(
        SimpleNamespace(
            assurance_mode="semantic",
            assurance_profile_path=str(path),
            assurance_ledger_path="",
            assurance_budget_tokens=0,
            assurance_budget_fraction=0.2,
            assurance_preserve_last_n=4,
            assurance_fallback="original",
            assurance_max_expansions=2,
        )
    )
    result = controller.apply(_body(), query="question", context_window=1000)
    assert result.body == _body()
    assert result.headers["X-Entroly-Assurance-Decision"] == "BYPASS_INIT_ERROR"
    assert "profile:ValueError" in result.headers["X-Entroly-Assurance-Error"]


def test_profile_loader_rejects_unknown_fields(tmp_path) -> None:
    profile = CalibrationProfile(
        version="v",
        threshold=0.1,
        target_failure_rate=0.01,
        accepted_samples=100,
        accepted_failures=0,
        failure_upper_bound=0.01,
        total_samples=200,
        dataset_count=2,
        model_count=2,
        accepted_dataset_count=2,
        accepted_model_count=2,
        dataset_fingerprint="a" * 64,
        calibration_membership=("b" * 64,),
      calibration_ready=True,
    )
    payload = asdict(profile)
    payload["unknown"] = True
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        load_calibration_profile(path)


def test_proxy_config_from_env_is_opt_in(monkeypatch) -> None:
    from entroly.proxy_config import ProxyConfig

    monkeypatch.delenv("ENTROLY_ASSURANCE_MODE", raising=False)
    assert ProxyConfig.from_env().assurance_mode == "off"

    monkeypatch.setenv("ENTROLY_ASSURANCE_MODE", "candidate_units")
    monkeypatch.setenv("ENTROLY_ASSURANCE_BUDGET_TOKENS", "2048")
    monkeypatch.setenv("ENTROLY_ASSURANCE_PRESERVE_LAST_N", "6")
    config = ProxyConfig.from_env()
    assert config.assurance_mode == "candidate_units"
    assert config.assurance_budget_tokens == 2048
    assert config.assurance_preserve_last_n == 6


def test_proxy_initializes_assurance_without_enabling_by_default(monkeypatch) -> None:
    from entroly.proxy import PromptCompilerProxy
    from entroly.proxy_config import ProxyConfig

    monkeypatch.setenv("ENTROLY_SESSION_RESCUE", "0")
    monkeypatch.setenv("ENTROLY_RATE_LIMIT", "0")

    class Engine:
        def stats(self):
            return {}

    proxy = PromptCompilerProxy(Engine(), ProxyConfig())
    assert not proxy._assurance_controller.enabled

    proxy_enabled = PromptCompilerProxy(
        Engine(),
        ProxyConfig(assurance_mode="candidate_units"),
    )
    assert proxy_enabled._assurance_controller.enabled


def test_live_proxy_assurance_skips_legacy_lossy_compressors(monkeypatch) -> None:
    import asyncio
    import json as _json

    from httpx import ASGITransport, AsyncClient
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    import entroly.proxy as proxy_module
    import entroly.proxy_assurance as assurance_module
    import entroly.proxy_transform as transform_module
    from entroly.proxy import PromptCompilerProxy
    from entroly.proxy_config import ProxyConfig

    monkeypatch.setenv("ENTROLY_SESSION_RESCUE", "0")
    monkeypatch.setenv("ENTROLY_RATE_LIMIT", "0")

    fake = SimpleNamespace(
        messages=(
            {"role": "assistant", "content": "recent answer"},
            {"role": "user", "content": "current question"},
        ),
        receipt={
            "decision": "COMPRESSED_CERTIFIED",
            "attempts": [
                {
                    "certificate_scope": "candidate_units",
                    "certificate_verdict": "sufficient",
                }
            ],
        },
        original_tokens=1000,
        delivered_tokens=20,
        budget_compliant=True,
        changed=True,
    )
    monkeypatch.setattr(
        assurance_module,
        "compress_messages_assured",
        lambda *_args, **_kwargs: fake,
    )
    monkeypatch.setattr(
        transform_module,
        "compress_tool_messages",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy tool compressor must not run")
        ),
    )
    monkeypatch.setattr(
        proxy_module,
        "compress_conversation_messages",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy conversation compressor must not run")
        ),
    )

    class Engine:
        def stats(self):
            return {}

        def advance_turn(self):
            return None

        def optimize_context(self, token_budget, query):
            return {"selected_fragments": [], "query_analysis": {}}

    async def run():
        config = ProxyConfig(
            assurance_mode="candidate_units",
            witness_mode="off",
        )
        config.enable_adaptive_budget = False
        config.enable_dynamic_budget = False
        config.enable_hierarchical_compression = False
        config.enable_passive_feedback = False
        config.enable_context_scaffold = False
        proxy = PromptCompilerProxy(Engine(), config)
        proxy._confidence_threshold = 1.0
        captured = {}

        async def capture(_url, _headers, body, *_args, **kwargs):
            captured["body"] = _json.loads(_json.dumps(body))
            return JSONResponse(
                {"ok": True},
                headers=kwargs.get("extra_headers") or {},
            )

        proxy._forward_response = capture
        app = Starlette(
            routes=[
                Route(
                    "/v1/chat/completions",
                    proxy.handle_proxy,
                    methods=["POST"],
                )
            ]
        )
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"authorization": "Bearer test"},
                json=_body(),
            )
        return response, captured

    response, captured = asyncio.run(run())
    assert response.status_code == 200
    assert response.headers["x-entroly-assurance-decision"] == (
        "COMPRESSED_CERTIFIED"
    )
    assert response.headers["x-entroly-assurance-certificate"] == (
        "sufficient/candidate_units"
    )
    assert captured["body"]["messages"] == list(fake.messages)


def test_proxy_assurance_operator_contract_is_documented() -> None:
    docs = (Path(__file__).resolve().parents[1] / "docs" / "ASSURED_CONTEXT.md").read_text(encoding="utf-8")
    assert "ENTROLY_ASSURANCE_MODE=candidate_units" in docs
    assert "ENTROLY_ASSURANCE_PROFILE" in docs
    assert "ENTROLY_TRUST_PROXY_ENV=1" in docs
    assert "original request unchanged" in docs
