"""
Multi-Provider Proxy Support — Comprehensive Test Suite
=========================================================

Tests for zero-friction multi-provider proxy support across all major IDEs.
Covers the critical detection, extraction, injection, and routing logic.

  T-01  DETECT PROVIDER PATH         path-based provider detection
  T-02  DETECT PROVIDER HEADERS      header-based provider detection
  T-03  DETECT PROVIDER FORMAT       body-format-based provider detection (contents vs messages)
  T-04  DETECT PROVIDER OPENAI       OpenAI-compatible providers default to "openai"
  T-05  EXTRACT USER GEMINI          Gemini contents/parts extraction
  T-06  EXTRACT MODEL GEMINI URL     model name from Gemini URL path
  T-07  INJECT CONTEXT GEMINI        systemInstruction creation
  T-08  INJECT CONTEXT GEMINI EX     prepend to existing systemInstruction
  T-09  CONTEXT WINDOW GEMINI        Gemini model context windows
  T-10  PROVIDER CONTROLS PASS      request-side generation controls pass through
  T-12  PROVIDER FALLBACK            edge cases and priority
  T-13  PROXY CONFIG GEMINI          gemini_base_url in ProxyConfig
  T-14  IDE REALISTIC SCENARIOS      real-world IDE request patterns (Cursor, VS Code, Claude Code, etc.)
  T-15  OPENROUTER MULTI-PROVIDER    Gemini/Claude models via OpenRouter stay "openai"
  T-16  STREAMING DETECTION          Gemini streamGenerateContent path detection
  T-17  END-TO-END FORMAT CHAIN      detect → extract → inject full pipeline
"""

import sys
from pathlib import Path

import pytest

# Ensure the entroly package is importable
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from entroly.proxy_transform import (  # noqa: E402
    detect_provider,
    extract_model,
    extract_user_message,
    inject_context_gemini,
    inject_context_openai,
    inject_context_anthropic,
    is_legacy_claude_3_model,
    strip_anthropic_unsupported_params,
)
from entroly.proxy_config import ProxyConfig, context_window_for_model  # noqa: E402
from entroly.proxy import (  # noqa: E402
    _content_to_text,
    _estimate_message_tokens,
    compress_conversation_messages,
    PromptCompilerProxy,
)


class TestProviderForwardingPolicy:
    def test_http_client_prefers_explicit_ca_bundle_env(self, monkeypatch, tmp_path):
        from entroly.proxy import _http_client_kwargs

        ca_file = tmp_path / "ca.pem"
        ca_file.write_text("test-ca", encoding="utf-8")
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(ca_file))

        kwargs = _http_client_kwargs()

        assert kwargs["trust_env"] is True
        assert kwargs["verify"] == str(ca_file)

    def test_anthropic_headers_preserve_provider_metadata_without_hop_by_hop(self):
        proxy = PromptCompilerProxy(object(), ProxyConfig())

        out = proxy._build_headers(
            {
                "authorization": "Bearer test",
                "x-api-key": "sk-ant-test",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "prompt-caching-2024-07-31",
                "x-stainless-lang": "python",
                "host": "127.0.0.1:9377",
                "content-length": "123",
            },
            "anthropic",
        )

        assert out["Content-Type"] == "application/json"
        assert out["authorization"] == "Bearer test"
        assert out["x-api-key"] == "sk-ant-test"
        assert out["anthropic-version"] == "2023-06-01"
        assert out["anthropic-beta"] == "prompt-caching-2024-07-31"
        assert out["x-stainless-lang"] == "python"
        assert "host" not in out
        assert "content-length" not in out

    def test_openai_compatible_prefix_models_use_registry_context_window(self):
        assert context_window_for_model("deepseek-reasoner") == 128_000


class TestAnthropicCompatibilitySanitizer:
    """Provider-level cleanup for native Anthropic request bodies."""

    def test_legacy_claude_3_models_are_identified(self):
        assert is_legacy_claude_3_model("claude-3-haiku-20240307")
        assert is_legacy_claude_3_model("claude-3-5-sonnet-20241022")

    def test_newer_and_unknown_models_are_preserved(self):
        assert not is_legacy_claude_3_model("claude-sonnet-4-5-20250929")
        assert not is_legacy_claude_3_model("some-future-model")

    def test_strips_context_management_for_legacy_anthropic_target(self):
        # Architectural rule: generation params (`thinking`, `temperature`, …)
        # are NEVER touched, even on legacy Claude 3.x targets. Only the
        # transport-layer `context_management` field, which the public
        # Messages API actively rejects, gets removed.
        body = {
            "model": "claude-3-5-haiku-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "context_management": {"strategy": "auto"},
            "thinking": {"type": "enabled"},
            "max_tokens": 1024,
        }

        cleaned = strip_anthropic_unsupported_params(body)

        assert "context_management" not in cleaned
        assert "thinking" in cleaned, "generation params must not be stripped"
        assert cleaned["max_tokens"] == 1024
        assert body["context_management"] == {"strategy": "auto"}

    def test_strips_context_management_for_newer_anthropic_target(self):
        body = {
            "model": "claude-sonnet-4-5-20250929",
            "context_management": {"strategy": "auto"},
            "thinking": {"type": "enabled"},
        }

        cleaned = strip_anthropic_unsupported_params(body)
        assert "context_management" not in cleaned
        assert "thinking" in cleaned

    def test_strips_context_management_for_unknown_anthropic_target(self):
        body = {
            "model": "custom-anthropic-compatible-model",
            "context_management": {"strategy": "auto"},
            "thinking": {"type": "enabled"},
        }

        cleaned = strip_anthropic_unsupported_params(body)
        assert "context_management" not in cleaned
        assert "thinking" in cleaned

    def test_claude_code_sonnet_46_context_management_is_removed(self):
        body = {
            "model": "claude-sonnet-4-6-20260501",
            "messages": [{"role": "user", "content": "what is nix"}],
            "context_management": {"edits": "auto"},
            "max_tokens": 1024,
        }

        cleaned = strip_anthropic_unsupported_params(body)

        assert "context_management" not in cleaned
        assert cleaned["messages"] == body["messages"]
        assert cleaned["max_tokens"] == 1024


class TestProxyContentBlockAccounting:
    """Regression coverage for Claude Code block-list message content."""

    def test_content_to_text_handles_anthropic_blocks(self):
        content = [
            {"type": "text", "text": "review this PR"},
            {"type": "image", "source": {"type": "base64", "data": "..."}},
        ]

        assert _content_to_text(content) == "review this PR"

    def test_token_estimate_handles_anthropic_blocks(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "what is nix"}],
            }
        ]

        assert _estimate_message_tokens(messages) > 0

    def test_conversation_compression_preserves_block_content_shape(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "what is nix"}],
            }
        ]

        result = compress_conversation_messages(messages, context_window=1)

        assert result[0]["content"] == messages[0]["content"]

    def test_full_proxy_optimizes_claude_code_blocks_without_split_error(self):
        import asyncio
        import json

        from httpx import ASGITransport, AsyncClient
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        class FakeEngine:
            def __init__(self):
                self._turn_counter = 0

            def advance_turn(self):
                return None

            def optimize_context(self, token_budget, query):
                return {
                    "selected_fragments": [
                        {
                            "id": "auth-1",
                            "source": "auth.py",
                            "content": "def nix_package(): return 'nix'",
                            "preview": "def nix_package(): return 'nix'",
                            "token_count": 16,
                            "entropy_score": 1.0,
                            "variant": "full",
                        }
                    ],
                    "query_analysis": {},
                }

        async def run():
            cfg = ProxyConfig()
            cfg.enable_adaptive_budget = False
            cfg.enable_dynamic_budget = False
            cfg.enable_hierarchical_compression = False
            cfg.enable_passive_feedback = False
            cfg.enable_context_scaffold = False
            proxy = PromptCompilerProxy(FakeEngine(), cfg)
            proxy._confidence_threshold = 0.0
            app = Starlette(
                routes=[Route("/v1/messages", proxy.handle_proxy, methods=["POST"])]
            )
            captured = {}

            async def capture(_url, _headers, body, *_args, **_kwargs):
                captured["body"] = json.loads(json.dumps(body))
                return JSONResponse(
                    {
                        "id": "msg-test",
                        "type": "message",
                        "model": "claude-sonnet-4-6-20260501",
                        "content": [{"type": "text", "text": "Nix is a package manager."}],
                    }
                )

            proxy._forward_response = capture
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/v1/messages",
                    json={
                        "model": "claude-sonnet-4-6-20260501",
                        "max_tokens": 128,
                        "context_management": {"edits": "auto"},
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": "what is nix"}],
                            }
                        ],
                    },
                    headers={
                        "x-api-key": "sk-ant-test",
                        "anthropic-version": "2023-06-01",
                    },
                )
            return response, captured, proxy

        response, captured, proxy = asyncio.run(run())

        assert response.status_code == 200
        assert proxy._requests_optimized == 1
        assert "context_management" not in captured["body"]
        assert "auth.py" in json.dumps(captured["body"])


# ═══════════════════════════════════════════════════════════════════════
# T-01: detect_provider — Path-Based Detection
# ═══════════════════════════════════════════════════════════════════════

class TestDetectProviderPath:
    """Provider detection from request path."""

    def test_anthropic_messages_path(self):
        assert detect_provider("/v1/messages", {}) == "anthropic"

    def test_anthropic_messages_path_with_params(self):
        assert detect_provider("/v1/messages?stream=true", {}) == "anthropic"

    def test_gemini_generate_content(self):
        path = "/v1beta/models/gemini-2.5-pro:generateContent"
        assert detect_provider(path, {}) == "gemini"

    def test_gemini_stream_generate_content(self):
        path = "/v1beta/models/gemini-2.0-flash:streamGenerateContent"
        assert detect_provider(path, {}) == "gemini"

    def test_openai_chat_completions(self):
        assert detect_provider("/v1/chat/completions", {}) == "openai"

    def test_openai_completions(self):
        assert detect_provider("/v1/completions", {}) == "openai"

    def test_path_priority_over_body_format(self):
        """Path-based detection beats body-format detection."""
        body = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
        assert detect_provider("/v1/messages", {}, body) == "anthropic"

    def test_gemini_path_with_messages_body(self):
        """Gemini path wins even if body has messages (shouldn't happen, but safe)."""
        body = {"messages": [{"role": "user", "content": "hi"}]}
        path = "/v1beta/models/gemini-2.0-flash:generateContent"
        assert detect_provider(path, {}, body) == "gemini"


# ═══════════════════════════════════════════════════════════════════════
# T-02: detect_provider — Header-Based Detection
# ═══════════════════════════════════════════════════════════════════════

class TestDetectProviderHeaders:
    """Provider detection from request headers."""

    def test_gemini_goog_api_key(self):
        headers = {"x-goog-api-key": "AIza..."}
        assert detect_provider("/some/path", headers) == "gemini"

    def test_anthropic_x_api_key_only(self):
        headers = {"x-api-key": "sk-ant-..."}
        assert detect_provider("/some/path", headers) == "anthropic"

    def test_anthropic_x_api_key_with_authorization_is_openai(self):
        """When both x-api-key and authorization are present, default to openai."""
        headers = {"x-api-key": "key", "authorization": "Bearer sk-..."}
        assert detect_provider("/some/path", headers) == "openai"

    def test_openai_authorization_only(self):
        headers = {"authorization": "Bearer sk-..."}
        assert detect_provider("/some/path", headers) == "openai"

    def test_gemini_header_priority_over_body(self):
        """x-goog-api-key header wins even with OpenAI-format body."""
        headers = {"x-goog-api-key": "AIza..."}
        body = {"messages": [{"role": "user", "content": "hi"}]}
        assert detect_provider("/some/path", headers, body) == "gemini"


# ═══════════════════════════════════════════════════════════════════════
# T-03: detect_provider — Body-Format Detection
# ═══════════════════════════════════════════════════════════════════════

class TestDetectProviderBodyFormat:
    """Provider detection from body format (contents vs messages)."""

    def test_native_gemini_contents_format(self):
        """Body with 'contents' and no 'messages' → gemini."""
        body = {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]}
        assert detect_provider("/api/generate", {}, body) == "gemini"

    def test_openai_messages_format(self):
        """Body with 'messages' → openai, even if model is gemini."""
        body = {
            "model": "gemini-2.5-pro",
            "messages": [{"role": "user", "content": "hello"}],
        }
        assert detect_provider("/v1/chat/completions", {}, body) == "openai"

    def test_openai_messages_with_claude_model(self):
        """Body with 'messages' and claude model → openai (e.g., OpenRouter)."""
        body = {
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "hello"}],
        }
        assert detect_provider("/v1/chat/completions", {}, body) == "openai"

    def test_body_with_both_contents_and_messages(self):
        """If body has both 'contents' and 'messages', prefer openai (messages)."""
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "contents": [{"parts": [{"text": "hi"}]}],
        }
        assert detect_provider("/v1/chat/completions", {}, body) == "openai"

    def test_empty_body_defaults_openai(self):
        assert detect_provider("/v1/chat/completions", {}, {}) == "openai"

    def test_no_body_defaults_openai(self):
        assert detect_provider("/v1/chat/completions", {}) == "openai"


# ═══════════════════════════════════════════════════════════════════════
# T-04: detect_provider — OpenAI-Compatible Providers
# ═══════════════════════════════════════════════════════════════════════

class TestDetectProviderOpenAICompat:
    """OpenAI-compatible providers (OpenRouter, Ollama, etc.) → 'openai'."""

    def test_deepseek_model(self):
        body = {"model": "deepseek-chat", "messages": []}
        assert detect_provider("/v1/chat/completions", {}, body) == "openai"

    def test_mistral_model(self):
        body = {"model": "mistral-large-latest", "messages": []}
        assert detect_provider("/v1/chat/completions", {}, body) == "openai"

    def test_ollama_model(self):
        body = {"model": "llama3:latest", "messages": []}
        assert detect_provider("/v1/chat/completions", {}, body) == "openai"

    def test_openrouter_slashed_model(self):
        body = {"model": "anthropic/claude-3-opus", "messages": []}
        assert detect_provider("/v1/chat/completions", {}, body) == "openai"

    def test_unknown_model(self):
        body = {"model": "my-custom-model-v2", "messages": []}
        assert detect_provider("/v1/chat/completions", {}, body) == "openai"


# ═══════════════════════════════════════════════════════════════════════
# T-05: extract_user_message — Gemini Format
# ═══════════════════════════════════════════════════════════════════════

class TestExtractUserMessageGemini:
    """Gemini uses contents/parts instead of messages."""

    def test_simple_user_message(self):
        body = {
            "contents": [
                {"role": "user", "parts": [{"text": "Hello world"}]}
            ]
        }
        assert extract_user_message(body, "gemini") == "Hello world"

    def test_multi_part_message(self):
        body = {
            "contents": [
                {"role": "user", "parts": [
                    {"text": "First part"},
                    {"text": "Second part"},
                ]}
            ]
        }
        assert extract_user_message(body, "gemini") == "First part Second part"

    def test_last_user_message(self):
        body = {
            "contents": [
                {"role": "user", "parts": [{"text": "First question"}]},
                {"role": "model", "parts": [{"text": "Answer"}]},
                {"role": "user", "parts": [{"text": "Follow up"}]},
            ]
        }
        assert extract_user_message(body, "gemini") == "Follow up"

    def test_empty_contents(self):
        body = {"contents": []}
        assert extract_user_message(body, "gemini") == ""

    def test_no_contents(self):
        body = {}
        assert extract_user_message(body, "gemini") == ""

    def test_mixed_parts_with_non_text(self):
        """Non-text parts (images, etc.) are skipped."""
        body = {
            "contents": [
                {"role": "user", "parts": [
                    {"inlineData": {"mimeType": "image/png", "data": "..."}},
                    {"text": "What is this?"},
                ]}
            ]
        }
        assert extract_user_message(body, "gemini") == "What is this?"

    def test_implicit_user_role(self):
        """Gemini defaults role to 'user' when absent."""
        body = {
            "contents": [
                {"parts": [{"text": "Implicit user"}]}
            ]
        }
        assert extract_user_message(body, "gemini") == "Implicit user"


# ═══════════════════════════════════════════════════════════════════════
# T-06: extract_model — Gemini URL Path
# ═══════════════════════════════════════════════════════════════════════

class TestExtractModelGeminiURL:
    """Gemini embeds model name in URL path."""

    def test_generate_content_url(self):
        path = "/v1beta/models/gemini-2.5-pro:generateContent"
        assert extract_model({}, path) == "gemini-2.5-pro"

    def test_stream_generate_content_url(self):
        path = "/v1beta/models/gemini-2.0-flash:streamGenerateContent"
        assert extract_model({}, path) == "gemini-2.0-flash"

    def test_body_model_takes_precedence(self):
        body = {"model": "gemini-2.5-flash"}
        path = "/v1beta/models/gemini-2.0-flash:generateContent"
        assert extract_model(body, path) == "gemini-2.5-flash"

    def test_no_model_no_path(self):
        assert extract_model({}, "/v1/chat/completions") == ""

    def test_standard_body_model(self):
        body = {"model": "gpt-4o"}
        assert extract_model(body) == "gpt-4o"


# ═══════════════════════════════════════════════════════════════════════
# T-07: inject_context_gemini — New systemInstruction
# ═══════════════════════════════════════════════════════════════════════

class TestInjectContextGeminiNew:
    """inject_context_gemini creates systemInstruction when absent."""

    def test_creates_system_instruction(self):
        body = {"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]}
        result = inject_context_gemini(body, "Context here")
        assert "systemInstruction" in result
        assert result["systemInstruction"] == {
            "parts": [{"text": "Context here"}]
        }

    def test_preserves_contents(self):
        body = {"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]}
        result = inject_context_gemini(body, "Context")
        assert result["contents"] == body["contents"]

    def test_does_not_mutate_original(self):
        body = {"contents": []}
        original_keys = set(body.keys())
        inject_context_gemini(body, "Context")
        assert set(body.keys()) == original_keys


# ═══════════════════════════════════════════════════════════════════════
# T-08: inject_context_gemini — Existing systemInstruction
# ═══════════════════════════════════════════════════════════════════════

class TestInjectContextGeminiExisting:
    """inject_context_gemini prepends to existing systemInstruction."""

    def test_prepends_to_existing(self):
        body = {
            "systemInstruction": {
                "parts": [{"text": "You are a helpful assistant."}]
            },
            "contents": [],
        }
        result = inject_context_gemini(body, "Injected context")
        parts = result["systemInstruction"]["parts"]
        assert len(parts) == 2
        assert parts[0]["text"] == "Injected context"
        assert parts[1]["text"] == "You are a helpful assistant."

    def test_replaces_non_dict_system_instruction(self):
        body = {
            "systemInstruction": "just a string",
            "contents": [],
        }
        result = inject_context_gemini(body, "Context")
        assert result["systemInstruction"] == {
            "parts": [{"text": "Context"}]
        }


# ═══════════════════════════════════════════════════════════════════════
# T-09: context_window_for_model — Gemini Models
# ═══════════════════════════════════════════════════════════════════════

class TestContextWindowGemini:
    """Gemini model context window lookup."""

    def test_gemini_25_pro(self):
        assert context_window_for_model("gemini-2.5-pro") == 1_048_576

    def test_gemini_25_flash(self):
        assert context_window_for_model("gemini-2.5-flash") == 1_048_576

    def test_gemini_20_flash(self):
        assert context_window_for_model("gemini-2.0-flash") == 1_048_576

    def test_gemini_15_pro(self):
        assert context_window_for_model("gemini-1.5-pro") == 2_097_152

    def test_gemini_15_flash(self):
        assert context_window_for_model("gemini-1.5-flash") == 1_048_576

    def test_gemini_prefix_match(self):
        """Fuzzy prefix matching for dated variants."""
        assert context_window_for_model("gemini-2.5-pro-preview-0325") == 1_048_576

    def test_unknown_model_default(self):
        assert context_window_for_model("totally-unknown-model") == 128_000


# ═══════════════════════════════════════════════════════════════════════
# T-10/T-11: Provider request controls pass through unchanged
# ========================================================================

class TestProviderRequestControlsPassthrough:
    """Entroly must not write provider-owned generation controls."""

    def test_openai_temperature_is_preserved(self):
        body = {
            "model": "gpt-4o",
            "temperature": 0.2,
            "messages": [{"role": "user", "content": "hello"}],
        }

        result = inject_context_openai(body, "CTX")

        assert result["temperature"] == 0.2
        assert body["messages"] == [{"role": "user", "content": "hello"}]

    def test_openai_temperature_is_not_injected(self):
        body = {"model": "gpt-4o", "messages": []}

        result = inject_context_openai(body, "CTX")

        assert "temperature" not in result

    def test_anthropic_thinking_is_preserved(self):
        body = {
            "model": "claude-sonnet-4-5-20250929",
            "thinking": {"type": "enabled", "budget_tokens": 8192},
            "messages": [{"role": "user", "content": "hello"}],
        }

        result = inject_context_anthropic(body, "CTX")

        assert result["thinking"] == {"type": "enabled", "budget_tokens": 8192}
        assert "temperature" not in result

    def test_anthropic_temperature_is_preserved_when_user_sent_it(self):
        body = {
            "model": "claude-sonnet-4-5-20250929",
            "temperature": 1,
            "thinking": {"type": "enabled", "budget_tokens": 8192},
            "messages": [{"role": "user", "content": "hello"}],
        }

        result = inject_context_anthropic(body, "CTX")

        assert result["temperature"] == 1
        assert result["thinking"] == body["thinking"]

    def test_gemini_generation_config_is_preserved(self):
        body = {
            "contents": [],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1024},
        }

        result = inject_context_gemini(body, "CTX")

        assert result["generationConfig"] == {
            "temperature": 0.3,
            "maxOutputTokens": 1024,
        }

    def test_gemini_generation_config_is_not_injected(self):
        body = {"contents": []}

        result = inject_context_gemini(body, "CTX")

        assert "generationConfig" not in result

# T-12: Provider Fallback & Priority
# ═══════════════════════════════════════════════════════════════════════

class TestProviderFallback:
    """Edge cases and detection priority."""

    def test_empty_path_and_headers(self):
        assert detect_provider("", {}) == "openai"

    def test_generic_path(self):
        assert detect_provider("/api/v1/generate", {}) == "openai"

    def test_path_priority_over_format(self):
        """Path detection takes priority over body format."""
        body = {"contents": [{"parts": [{"text": "hi"}]}]}
        assert detect_provider("/v1/messages", {}, body) == "anthropic"

    def test_header_priority_over_format(self):
        """Header detection takes priority over body format."""
        body = {"messages": [{"role": "user", "content": "hi"}]}
        headers = {"x-goog-api-key": "AIza..."}
        assert detect_provider("/some/path", headers, body) == "gemini"


# ═══════════════════════════════════════════════════════════════════════
# T-13: ProxyConfig — gemini_base_url
# ═══════════════════════════════════════════════════════════════════════

class TestProxyConfigGemini:
    """ProxyConfig includes gemini_base_url with correct default."""

    def test_default_gemini_base_url(self):
        config = ProxyConfig()
        assert config.gemini_base_url == "https://generativelanguage.googleapis.com"

    def test_custom_gemini_base_url(self):
        config = ProxyConfig(gemini_base_url="http://localhost:8080")
        assert config.gemini_base_url == "http://localhost:8080"

    def test_from_env_gemini_base(self, monkeypatch):
        monkeypatch.setenv("ENTROLY_GEMINI_BASE", "https://custom.gemini.api")
        config = ProxyConfig.from_env()
        assert config.gemini_base_url == "https://custom.gemini.api"


# ═══════════════════════════════════════════════════════════════════════
# T-14: IDE Realistic Scenarios
# ═══════════════════════════════════════════════════════════════════════

class TestProxyConfigEnvParsing:
    """ProxyConfig.from_env tolerates malformed optional numeric knobs."""

    def test_from_env_numeric_values_fall_back_when_invalid(self, monkeypatch):
        monkeypatch.setenv("ENTROLY_PROXY_PORT", "not-an-int")
        monkeypatch.setenv("ENTROLY_CONTEXT_FRACTION", "not-a-float")
        monkeypatch.setenv("ENTROLY_QUALITY", "not-a-float")
        monkeypatch.setenv("ENTROLY_FISHER_SCALE", "not-a-float")
        monkeypatch.setenv("ENTROLY_TRAJECTORY_CMIN", "not-a-float")
        monkeypatch.setenv("ENTROLY_TRAJECTORY_LAMBDA", "not-a-float")

        config = ProxyConfig.from_env()

        assert config.port == 9377
        assert config.context_fraction == 0.15
        assert config.quality is None
        assert config.fisher_scale == 0.55
        assert config.trajectory_c_min == 0.6
        assert config.trajectory_lambda == 0.07

    def test_from_env_numeric_values_keep_valid_overrides(self, monkeypatch):
        monkeypatch.setenv("ENTROLY_PROXY_PORT", "9444")
        monkeypatch.setenv("ENTROLY_CONTEXT_FRACTION", "0.2")
        monkeypatch.setenv("ENTROLY_FISHER_SCALE", "0.7")
        monkeypatch.setenv("ENTROLY_TRAJECTORY_CMIN", "0.8")
        monkeypatch.setenv("ENTROLY_TRAJECTORY_LAMBDA", "0.03")

        config = ProxyConfig.from_env()

        assert config.port == 9444
        assert config.context_fraction == 0.2
        assert config.fisher_scale == 0.7
        assert config.trajectory_c_min == 0.8
        assert config.trajectory_lambda == 0.03


class TestIDERealisticScenarios:
    """Real-world request patterns from major IDEs and tools.

    Each test simulates the EXACT request shape an IDE sends, verifying
    that detect_provider + extract_user_message + injection all work
    correctly for that IDE's specific format.
    """

    # ── Cursor ──

    def test_cursor_openai_gpt4o(self):
        """Cursor → GPT-4o: standard OpenAI format."""
        path = "/v1/chat/completions"
        headers = {"authorization": "Bearer sk-..."}
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "fix the login bug"},
            ],
            "stream": True,
        }
        assert detect_provider(path, headers, body) == "openai"
        assert extract_user_message(body, "openai") == "fix the login bug"
        result = inject_context_openai(body, "CONTEXT")
        assert result["messages"][0]["content"].startswith("CONTEXT")

    def test_cursor_anthropic_claude(self):
        """Cursor → Claude: Anthropic messages format."""
        path = "/v1/messages"
        headers = {"x-api-key": "sk-ant-...", "anthropic-version": "2023-06-01"}
        body = {
            "model": "claude-sonnet-4-5-20250929",
            "system": "You are a coding assistant.",
            "messages": [{"role": "user", "content": "refactor this function"}],
            "max_tokens": 4096,
        }
        assert detect_provider(path, headers, body) == "anthropic"
        assert extract_user_message(body, "anthropic") == "refactor this function"
        result = inject_context_anthropic(body, "CONTEXT")
        assert result["system"].startswith("CONTEXT")

    def test_cursor_gemini_via_openrouter(self):
        """Cursor → OpenRouter → Gemini: OpenAI format with gemini model name.

        THIS IS THE CRITICAL BUG FIX TEST. Before the fix, this returned
        "gemini" which caused routing to Google's API instead of OpenRouter.
        """
        path = "/v1/chat/completions"
        headers = {"authorization": "Bearer sk-or-..."}
        body = {
            "model": "gemini-2.5-pro",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "write unit tests"},
            ],
            "stream": True,
        }
        # MUST be "openai" — the body uses messages format, goes to OpenRouter
        assert detect_provider(path, headers, body) == "openai"
        assert extract_user_message(body, "openai") == "write unit tests"
        result = inject_context_openai(body, "CONTEXT")
        assert result["messages"][0]["content"].startswith("CONTEXT")

    def test_cursor_claude_via_openrouter(self):
        """Cursor → OpenRouter → Claude: OpenAI format with claude model name.

        Another critical multi-provider scenario.
        """
        path = "/v1/chat/completions"
        headers = {"authorization": "Bearer sk-or-..."}
        body = {
            "model": "claude-opus-4-6",
            "messages": [
                {"role": "user", "content": "explain this code"},
            ],
        }
        # MUST be "openai" — body uses messages format for OpenRouter
        assert detect_provider(path, headers, body) == "openai"
        assert extract_user_message(body, "openai") == "explain this code"

    # ── VS Code with Copilot ──

    def test_vscode_copilot_gpt4o(self):
        """VS Code Copilot → GPT-4o: standard OpenAI format."""
        path = "/v1/chat/completions"
        headers = {"authorization": "Bearer ghu_..."}
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are GitHub Copilot."},
                {"role": "user", "content": "complete this function"},
            ],
        }
        assert detect_provider(path, headers, body) == "openai"
        assert extract_user_message(body, "openai") == "complete this function"

    # ── Claude Code ──

    def test_claude_code_anthropic(self):
        """Claude Code → Anthropic: native Anthropic format."""
        path = "/v1/messages"
        headers = {"x-api-key": "sk-ant-api03-..."}
        body = {
            "model": "claude-opus-4-6",
            "system": [{"type": "text", "text": "You are Claude."}],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "review this PR"},
                ]},
            ],
            "max_tokens": 8192,
        }
        assert detect_provider(path, headers, body) == "anthropic"
        assert extract_user_message(body, "anthropic") == "review this PR"
        result = inject_context_anthropic(body, "CONTEXT")
        assert isinstance(result["system"], list)
        assert result["system"][0]["text"] == "CONTEXT"

    # ── Native Gemini (Google AI Studio / SDK) ──

    def test_native_gemini_generate_content(self):
        """Google AI Studio → Gemini: native generateContent format."""
        path = "/v1beta/models/gemini-2.5-pro:generateContent"
        headers = {"x-goog-api-key": "AIzaSy..."}
        body = {
            "contents": [
                {"role": "user", "parts": [{"text": "explain this error"}]},
            ],
            "generationConfig": {"maxOutputTokens": 2048},
        }
        assert detect_provider(path, headers, body) == "gemini"
        assert extract_user_message(body, "gemini") == "explain this error"
        assert extract_model(body, path) == "gemini-2.5-pro"
        result = inject_context_gemini(body, "CONTEXT")
        assert result["systemInstruction"]["parts"][0]["text"] == "CONTEXT"
        assert result["generationConfig"]["maxOutputTokens"] == 2048

    def test_native_gemini_stream(self):
        """Google AI Studio → Gemini: streaming via streamGenerateContent."""
        path = "/v1beta/models/gemini-2.0-flash:streamGenerateContent"
        headers = {"x-goog-api-key": "AIzaSy..."}
        body = {
            "contents": [
                {"role": "user", "parts": [{"text": "write a function"}]},
            ],
        }
        assert detect_provider(path, headers, body) == "gemini"
        assert extract_model(body, path) == "gemini-2.0-flash"
        # Verify streaming detection from URL path
        assert "streamGenerateContent" in path

    def test_native_gemini_with_system_instruction(self):
        """Gemini request with existing systemInstruction — context prepended."""
        path = "/v1beta/models/gemini-2.5-pro:generateContent"
        headers = {"x-goog-api-key": "AIzaSy..."}
        body = {
            "systemInstruction": {
                "parts": [{"text": "You are a Python expert."}],
            },
            "contents": [
                {"role": "user", "parts": [{"text": "optimize this loop"}]},
            ],
        }
        provider = detect_provider(path, headers, body)
        assert provider == "gemini"
        result = inject_context_gemini(body, "CONTEXT")
        parts = result["systemInstruction"]["parts"]
        assert len(parts) == 2
        assert parts[0]["text"] == "CONTEXT"
        assert parts[1]["text"] == "You are a Python expert."

    # ── JetBrains AI ──

    def test_jetbrains_openai_format(self):
        """JetBrains AI Assistant → OpenAI-compatible format."""
        path = "/v1/chat/completions"
        headers = {"authorization": "Bearer jb-..."}
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "generate a test for UserService"},
            ],
        }
        assert detect_provider(path, headers, body) == "openai"

    # ── Ollama (local models) ──

    def test_ollama_local_model(self):
        """Ollama → local model: OpenAI-compatible format."""
        path = "/v1/chat/completions"
        headers = {}  # Ollama typically has no auth
        body = {
            "model": "codellama:13b",
            "messages": [
                {"role": "user", "content": "explain this regex"},
            ],
        }
        assert detect_provider(path, headers, body) == "openai"
        assert extract_user_message(body, "openai") == "explain this regex"

    # ── DeepSeek / Mistral ──

    def test_deepseek_via_direct_api(self):
        """DeepSeek → direct API: OpenAI-compatible format."""
        path = "/v1/chat/completions"
        headers = {"authorization": "Bearer sk-..."}
        body = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "debug this"}],
        }
        assert detect_provider(path, headers, body) == "openai"

    def test_mistral_via_direct_api(self):
        """Mistral → direct API: OpenAI-compatible format."""
        path = "/v1/chat/completions"
        headers = {"authorization": "Bearer ..."}
        body = {
            "model": "mistral-large-latest",
            "messages": [{"role": "user", "content": "refactor"}],
        }
        assert detect_provider(path, headers, body) == "openai"


# ═══════════════════════════════════════════════════════════════════════
# T-15: OpenRouter Multi-Provider (The Critical Bug Fix)
# ═══════════════════════════════════════════════════════════════════════

class TestOpenRouterMultiProvider:
    """OpenRouter sends ALL models via OpenAI format (/v1/chat/completions).

    Before the bug fix, model-name detection would route these to the
    wrong API (e.g., Google's API for gemini models, Anthropic's API for
    claude models). Now we use body-format detection instead.
    """

    def test_openrouter_gemini_model(self):
        body = {
            "model": "google/gemini-2.5-pro",
            "messages": [{"role": "user", "content": "hello"}],
        }
        assert detect_provider("/v1/chat/completions", {}, body) == "openai"

    def test_openrouter_gemini_bare_model(self):
        """Some configs use bare model names without vendor prefix."""
        body = {
            "model": "gemini-2.5-pro",
            "messages": [{"role": "user", "content": "hello"}],
        }
        assert detect_provider("/v1/chat/completions", {}, body) == "openai"

    def test_openrouter_claude_model(self):
        body = {
            "model": "anthropic/claude-3-opus",
            "messages": [{"role": "user", "content": "hello"}],
        }
        assert detect_provider("/v1/chat/completions", {}, body) == "openai"

    def test_openrouter_claude_bare_model(self):
        body = {
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "hello"}],
        }
        assert detect_provider("/v1/chat/completions", {}, body) == "openai"

    def test_openrouter_deepseek_model(self):
        body = {
            "model": "deepseek/deepseek-chat",
            "messages": [{"role": "user", "content": "hello"}],
        }
        assert detect_provider("/v1/chat/completions", {}, body) == "openai"

    def test_openrouter_full_pipeline(self):
        """Full pipeline: detect → extract → inject → context injection for OpenRouter."""
        path = "/v1/chat/completions"
        headers = {"authorization": "Bearer sk-or-..."}
        body = {
            "model": "gemini-2.5-pro",
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "fix the auth bug"},
            ],
        }
        provider = detect_provider(path, headers, body)
        assert provider == "openai"

        msg = extract_user_message(body, provider)
        assert msg == "fix the auth bug"

        injected = inject_context_openai(body, "CONTEXT")
        assert injected["messages"][0]["content"].startswith("CONTEXT")
        assert injected["model"] == "gemini-2.5-pro"  # model name preserved

        assert "temperature" not in injected
        assert "generationConfig" not in injected


# ═══════════════════════════════════════════════════════════════════════
# T-16: Streaming Detection
# ═══════════════════════════════════════════════════════════════════════

class TestStreamingDetection:
    """Verify streaming is detected correctly for all providers."""

    def test_openai_stream_field(self):
        body = {"model": "gpt-4o", "messages": [], "stream": True}
        assert body.get("stream", False) is True

    def test_openai_no_stream(self):
        body = {"model": "gpt-4o", "messages": []}
        assert body.get("stream", False) is False

    def test_gemini_streaming_from_url_path(self):
        """Gemini streaming is detected from URL path, not body field."""
        path = "/v1beta/models/gemini-2.0-flash:streamGenerateContent"
        body = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
        # Body has no "stream" field
        is_streaming = body.get("stream", False)
        assert is_streaming is False
        # But URL path indicates streaming
        if not is_streaming and "streamGenerateContent" in path:
            is_streaming = True
        assert is_streaming is True

    def test_gemini_non_streaming_url(self):
        """generateContent (not stream) should not trigger streaming."""
        path = "/v1beta/models/gemini-2.0-flash:generateContent"
        body = {"contents": []}
        is_streaming = body.get("stream", False)
        if not is_streaming and "streamGenerateContent" in path:
            is_streaming = True
        assert is_streaming is False

    def test_anthropic_stream_field(self):
        body = {"model": "claude-3", "messages": [], "stream": True}
        assert body.get("stream", False) is True


# ═══════════════════════════════════════════════════════════════════════
# T-17: End-to-End Format Chain
# ═══════════════════════════════════════════════════════════════════════

class TestEndToEndFormatChain:
    """Full pipeline: detect → extract → inject → context injection for each provider."""

    def test_openai_full_chain(self):
        path = "/v1/chat/completions"
        headers = {"authorization": "Bearer sk-..."}
        body = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
        }
        provider = detect_provider(path, headers, body)
        assert provider == "openai"
        msg = extract_user_message(body, provider)
        assert msg == "hello"
        model = extract_model(body, path)
        assert model == "gpt-4o"
        injected = inject_context_openai(body, "CTX")
        assert injected["messages"][0]["role"] == "system"
        assert "temperature" not in injected

    def test_anthropic_full_chain(self):
        path = "/v1/messages"
        headers = {"x-api-key": "sk-ant-..."}
        body = {
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "hello"}],
        }
        provider = detect_provider(path, headers, body)
        assert provider == "anthropic"
        msg = extract_user_message(body, provider)
        assert msg == "hello"
        model = extract_model(body, path)
        assert model == "claude-sonnet-4-5-20250929"
        injected = inject_context_anthropic(body, "CTX")
        assert injected["system"] == "CTX"
        assert "temperature" not in injected

    def test_gemini_full_chain(self):
        path = "/v1beta/models/gemini-2.5-pro:generateContent"
        headers = {"x-goog-api-key": "AIza..."}
        body = {
            "contents": [
                {"role": "user", "parts": [{"text": "hello"}]},
            ],
        }
        provider = detect_provider(path, headers, body)
        assert provider == "gemini"
        msg = extract_user_message(body, provider)
        assert msg == "hello"
        model = extract_model(body, path)
        assert model == "gemini-2.5-pro"
        injected = inject_context_gemini(body, "CTX")
        assert injected["systemInstruction"]["parts"][0]["text"] == "CTX"
        assert "generationConfig" not in injected

    def test_openrouter_gemini_full_chain(self):
        """OpenRouter with Gemini model — must use OpenAI format throughout."""
        path = "/v1/chat/completions"
        headers = {"authorization": "Bearer sk-or-..."}
        body = {
            "model": "gemini-2.5-pro",
            "messages": [{"role": "user", "content": "hello"}],
        }
        provider = detect_provider(path, headers, body)
        assert provider == "openai"  # NOT "gemini"!
        msg = extract_user_message(body, provider)
        assert msg == "hello"  # extracted from messages, not contents
        model = extract_model(body, path)
        assert model == "gemini-2.5-pro"
        injected = inject_context_openai(body, "CTX")
        assert injected["messages"][0]["role"] == "system"
        assert "temperature" not in injected
        assert "generationConfig" not in injected


class TestLiveGatewayRedaction:
    def test_bypass_still_enforces_explicit_outbound_policy(self):
        import asyncio

        from httpx import ASGITransport, AsyncClient
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        from entroly.provider_policy import GatewayRedactionPolicy

        async def run():
            proxy = PromptCompilerProxy(object(), ProxyConfig())
            proxy._bypass = True
            proxy._gateway_redaction = GatewayRedactionPolicy(enabled=True)
            captured = {}

            async def capture(_url, _headers, body, *_args, **kwargs):
                captured["body"] = body
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
                    json={
                        "model": "gpt-4o",
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    "contact dev@example.com using "
                                    "sk-abcdefghijklmnopqrstuvwxyz"
                                ),
                            }
                        ],
                    },
                )
            return response, captured["body"]

        response, forwarded = asyncio.run(run())

        rendered = str(forwarded)
        assert "dev@example.com" not in rendered
        assert "sk-abcdefghijklmnopqrstuvwxyz" not in rendered
        assert response.headers["x-entroly-redaction"] == "changed"
        assert int(response.headers["x-entroly-redaction-count"]) == 2

    def test_enabled_policy_fails_closed_for_uninspectable_raw_body(self):
        import asyncio

        from httpx import ASGITransport, AsyncClient
        from starlette.applications import Starlette
        from starlette.routing import Route

        from entroly.provider_policy import GatewayRedactionPolicy

        async def run():
            proxy = PromptCompilerProxy(object(), ProxyConfig())
            proxy._gateway_redaction = GatewayRedactionPolicy(enabled=True)
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
                return await client.post(
                    "/v1/chat/completions",
                    content=b"not-json secret",
                    headers={"content-type": "text/plain"},
                )

        response = asyncio.run(run())

        assert response.status_code == 415
        assert response.json()["error"] == "outbound_redaction_requires_json"
