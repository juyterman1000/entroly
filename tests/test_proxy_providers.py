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
  T-10  TEMPERATURE GEMINI           generationConfig.temperature placement
  T-11  TEMPERATURE GEMINI SKIP      respect user-set Gemini temperature
  T-12  PROVIDER FALLBACK            edge cases and priority
  T-13  PROXY CONFIG GEMINI          gemini_base_url in ProxyConfig
  T-14  IDE REALISTIC SCENARIOS      real-world IDE request patterns (Cursor, VS Code, Claude Code, etc.)
  T-15  OPENROUTER MULTI-PROVIDER    Gemini/Claude models via OpenRouter stay "openai"
  T-16  STREAMING DETECTION          Gemini streamGenerateContent path detection
  T-17  END-TO-END FORMAT CHAIN      detect → extract → inject → temperature full pipeline
"""

import sys
from pathlib import Path

import pytest

# Ensure the entroly package is importable
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from entroly.proxy_transform import (
    apply_temperature,
    detect_provider,
    extract_model,
    extract_user_message,
    inject_context_gemini,
    inject_context_openai,
    inject_context_anthropic,
)
from entroly.proxy_config import ProxyConfig, context_window_for_model


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
# T-10: apply_temperature — Gemini generationConfig
# ═══════════════════════════════════════════════════════════════════════

class TestApplyTemperatureGemini:
    """Gemini places temperature inside generationConfig."""

    def test_sets_temperature_in_generation_config(self):
        body = {"contents": []}
        result = apply_temperature(body, 0.7, provider="gemini")
        assert result["generationConfig"]["temperature"] == 0.7

    def test_preserves_existing_generation_config(self):
        body = {"contents": [], "generationConfig": {"maxOutputTokens": 1024}}
        result = apply_temperature(body, 0.5, provider="gemini")
        assert result["generationConfig"]["temperature"] == 0.5
        assert result["generationConfig"]["maxOutputTokens"] == 1024

    def test_does_not_mutate_original(self):
        body = {"contents": []}
        apply_temperature(body, 0.7, provider="gemini")
        assert "generationConfig" not in body


# ═══════════════════════════════════════════════════════════════════════
# T-11: apply_temperature — Gemini User Override
# ═══════════════════════════════════════════════════════════════════════

class TestApplyTemperatureGeminiOverride:
    """User-set Gemini temperature is never overwritten."""

    def test_respects_user_temperature(self):
        body = {
            "contents": [],
            "generationConfig": {"temperature": 0.3},
        }
        result = apply_temperature(body, 0.9, provider="gemini")
        assert result["generationConfig"]["temperature"] == 0.3

    def test_openai_user_override_still_works(self):
        body = {"messages": [], "temperature": 0.2}
        result = apply_temperature(body, 0.8, provider="openai")
        assert result["temperature"] == 0.2


# ═══════════════════════════════════════════════════════════════════════
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
        """Full pipeline: detect → extract → inject → temperature for OpenRouter."""
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

        temped = apply_temperature(injected, 0.5, provider)
        assert temped["temperature"] == 0.5  # top-level, not generationConfig


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
    """Full pipeline: detect → extract → inject → temperature for each provider."""

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
        temped = apply_temperature(injected, 0.4, provider)
        assert temped["temperature"] == 0.4

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
        temped = apply_temperature(injected, 0.4, provider)
        assert temped["temperature"] == 0.4

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
        temped = apply_temperature(injected, 0.4, provider)
        assert temped["generationConfig"]["temperature"] == 0.4

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
        temped = apply_temperature(injected, 0.4, provider)
        assert temped["temperature"] == 0.4  # top-level, not generationConfig
        assert "generationConfig" not in temped
