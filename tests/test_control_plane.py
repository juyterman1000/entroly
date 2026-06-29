from __future__ import annotations

import base64
import struct
from copy import deepcopy

from entroly.control_plane import (
    EntrolyCompressionDecision,
    audit_request_transform,
    canonical_json_dumps,
    plan_request,
    stable_request_fingerprint,
)
from entroly.proxy_transform import (
    inject_context_anthropic,
    inject_context_gemini,
    inject_context_openai,
)


def test_plan_request_respects_bypass_without_mutating_body():
    body = {
        "model": "gpt-4o",
        "temperature": 0.2,
        "messages": [{"role": "user", "content": "hello"}],
    }
    original = deepcopy(body)

    decision = plan_request(
        body,
        headers={"x-entroly-bypass": "true"},
        path="/v1/chat/completions",
        token_budget=1,
    )

    assert decision.action == "passthrough"
    assert decision.should_compress is False
    assert "bypass_header" in decision.reasons
    assert decision.cache_policy == "preserve_bytes"
    assert body == original
    assert isinstance(decision, EntrolyCompressionDecision)


def test_plan_request_compresses_tool_output_under_budget_pressure():
    tool_output = "[ERROR] " + ("database retry failed\n" * 300)
    body = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "find the failure"},
            {"role": "tool", "content": tool_output},
        ],
    }

    decision = plan_request(body, path="/v1/chat/completions", token_budget=100)

    assert decision.action == "compress"
    assert "tool_output_present" in decision.reasons
    assert "budget_pressure" in decision.reasons
    assert "cache_alignment_candidate" in decision.reasons
    assert any(s.kind == "tool_output" and s.modality == "log" for s in decision.surfaces)
    assert decision.budget_pressure is not None
    assert decision.budget_pressure > 1.0


def test_plan_request_observes_when_context_already_fits_budget():
    body = {"messages": [{"role": "user", "content": "short prompt"}]}

    decision = plan_request(body, token_budget=1_000)

    assert decision.action == "observe"
    assert "under_budget" in decision.reasons
    assert decision.should_compress is False
    assert decision.cache_policy == "align_context_prefix"


def test_plan_request_detects_provider_controls_and_image_surface():
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "What changed in this screenshot?"},
                    {"inlineData": {"mimeType": "image/png", "data": "abc"}},
                ],
            }
        ],
        "generationConfig": {"temperature": 0.3},
    }

    decision = plan_request(body, token_budget=10_000)

    assert decision.provider == "gemini"
    assert "provider_controls_present" in decision.reasons
    assert "image_present" in decision.reasons
    assert any(s.modality == "image" for s in decision.surfaces)


def test_plan_request_estimates_inline_image_tokens_when_dimensions_are_available():
    png = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 384, 384) + b"\x08\x02\x00\x00\x00"
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(png).decode("ascii")}},
                ],
            }
        ],
    }

    decision = plan_request(body, path="/v1beta/models/gemini-2.0-flash:generateContent")

    image_surface = next(s for s in decision.surfaces if s.modality == "image")
    assert image_surface.estimated_tokens == 258


def test_audit_accepts_openai_context_injection_without_control_changes():
    before = {
        "model": "gpt-4o",
        "temperature": 0.2,
        "top_p": 0.9,
        "messages": [{"role": "user", "content": "hello"}],
    }

    after = inject_context_openai(before, "CTX")
    audit = audit_request_transform(before, after, path="/v1/chat/completions")

    assert audit.compliant is True
    assert audit.provider_control_violations == ()
    assert audit.tool_contract_violations == ()


def test_audit_accepts_anthropic_context_injection_without_control_changes():
    before = {
        "model": "claude-sonnet-4-5-20250929",
        "temperature": 1,
        "thinking": {"type": "enabled", "budget_tokens": 8192},
        "messages": [{"role": "user", "content": "hello"}],
    }

    after = inject_context_anthropic(before, "CTX")
    audit = audit_request_transform(before, after, path="/v1/messages")

    assert audit.compliant is True
    assert audit.provider_control_violations == ()
    assert audit.tool_contract_violations == ()


def test_audit_accepts_gemini_context_injection_without_control_changes():
    before = {
        "contents": [],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1024},
    }

    after = inject_context_gemini(before, "CTX")
    audit = audit_request_transform(before, after, path="/v1beta/models/gemini:generateContent")

    assert audit.compliant is True
    assert audit.provider_control_violations == ()
    assert audit.tool_contract_violations == ()


def test_audit_flags_added_provider_control():
    before = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}
    after = deepcopy(before)
    after["temperature"] = 0.1

    audit = audit_request_transform(before, after, path="/v1/chat/completions")

    assert audit.compliant is False
    assert len(audit.provider_control_violations) == 1
    violation = audit.provider_control_violations[0]
    assert violation.path == "temperature"
    assert violation.kind == "added"
    assert violation.before_fingerprint == "missing"


def test_audit_flags_nested_gemini_generation_config_change():
    before = {
        "contents": [],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1024},
    }
    after = deepcopy(before)
    after["generationConfig"]["temperature"] = 0.9

    audit = audit_request_transform(before, after, path="/v1beta/models/gemini:generateContent")

    assert audit.compliant is False
    assert len(audit.provider_control_violations) == 1
    violation = audit.provider_control_violations[0]
    assert violation.path == "generationConfig"
    assert violation.kind == "changed"
    assert violation.before_fingerprint != violation.after_fingerprint


def test_audit_flags_tool_contract_change_unless_explicitly_allowed():
    before = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}],
    }
    after = deepcopy(before)
    after["tools"] = [{"type": "function", "function": {"name": "retrieve", "parameters": {}}}]

    blocked = audit_request_transform(before, after, path="/v1/chat/completions")
    allowed = audit_request_transform(
        before,
        after,
        path="/v1/chat/completions",
        allow_tool_contract_changes=True,
    )

    assert blocked.compliant is False
    assert blocked.tool_contract_violations[0].path == "tools"
    assert allowed.compliant is True


def test_canonical_json_dumps_is_dict_order_stable_but_list_order_sensitive():
    left = {"b": 2, "a": {"y": 1, "x": 0}, "tools": [{"name": "a"}, {"name": "b"}]}
    right = {"tools": [{"name": "a"}, {"name": "b"}], "a": {"x": 0, "y": 1}, "b": 2}
    reordered_tools = {"b": 2, "a": {"y": 1, "x": 0}, "tools": [{"name": "b"}, {"name": "a"}]}

    assert canonical_json_dumps(left) == canonical_json_dumps(right)
    assert canonical_json_dumps(left) != canonical_json_dumps(reordered_tools)


def test_stable_request_fingerprint_tracks_sticky_headers_and_tool_contracts():
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "search", "parameters": {}}}],
    }

    base = stable_request_fingerprint(
        body,
        headers={"Authorization": "Bearer secret-a", "OpenAI-Beta": "assistants=v2"},
        path="/v1/chat/completions",
    )
    same = stable_request_fingerprint(
        {"tools": body["tools"], "messages": body["messages"], "model": "gpt-4o"},
        headers={"openai-beta": "assistants=v2", "authorization": "Bearer secret-a"},
        path="/v1/chat/completions",
    )
    header_changed = stable_request_fingerprint(
        body,
        headers={"Authorization": "Bearer secret-a", "OpenAI-Beta": "responses=v1"},
        path="/v1/chat/completions",
    )
    tool_changed = deepcopy(body)
    tool_changed["tools"][0]["function"]["name"] = "lookup"
    tool_fingerprint_changed = stable_request_fingerprint(
        tool_changed,
        headers={"Authorization": "Bearer secret-a", "OpenAI-Beta": "assistants=v2"},
        path="/v1/chat/completions",
    )

    assert base == same
    assert base["entroly_header_fingerprint"] != header_changed["entroly_header_fingerprint"]
    assert base["entroly_protocol_fingerprint"] != header_changed["entroly_protocol_fingerprint"]
    assert base["entroly_tool_fingerprint"] != tool_fingerprint_changed["entroly_tool_fingerprint"]


def test_audit_allows_only_explicit_model_routing_change():
    before = {
        "model": "gpt-4o",
        "temperature": 0.2,
        "messages": [{"role": "user", "content": "hello"}],
    }
    after = deepcopy(before)
    after["model"] = "gpt-4o-mini"
    after["temperature"] = 0.8

    audit = audit_request_transform(
        before,
        after,
        path="/v1/chat/completions",
        allow_model_change=True,
    )

    assert audit.compliant is False
    assert len(audit.provider_control_violations) == 1
    assert audit.provider_control_violations[0].path == "temperature"
