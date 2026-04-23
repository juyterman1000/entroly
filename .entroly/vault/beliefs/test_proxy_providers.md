---
claim_id: c6d9a4d0-737c-43b9-ad30-496374d333f3
entity: test_proxy_providers
status: inferred
confidence: 0.75
sources:
  - tests\test_proxy_providers.py:52
  - tests\test_proxy_providers.py:91
  - tests\test_proxy_providers.py:122
  - tests\test_proxy_providers.py:165
  - tests\test_proxy_providers.py:193
  - tests\test_proxy_providers.py:259
  - tests\test_proxy_providers.py:287
  - tests\test_proxy_providers.py:314
  - tests\test_proxy_providers.py:345
  - tests\test_proxy_providers.py:375
last_checked: 2026-04-23T03:07:07.940720+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: test_proxy_providers

**Language:** python
**Lines of code:** 885

## Types
- `class TestDetectProviderPath()` — Provider detection from request path.
- `class TestDetectProviderHeaders()` — Provider detection from request headers.
- `class TestDetectProviderBodyFormat()` — Provider detection from body format (contents vs messages).
- `class TestDetectProviderOpenAICompat()` — OpenAI-compatible providers (OpenRouter, Ollama, etc.) → 'openai'.
- `class TestExtractUserMessageGemini()` — Gemini uses contents/parts instead of messages.
- `class TestExtractModelGeminiURL()` — Gemini embeds model name in URL path.
- `class TestInjectContextGeminiNew()` — inject_context_gemini creates systemInstruction when absent.
- `class TestInjectContextGeminiExisting()` — inject_context_gemini prepends to existing systemInstruction.
- `class TestContextWindowGemini()` — Gemini model context window lookup.
- `class TestApplyTemperatureGemini()` — Gemini places temperature inside generationConfig.
- `class TestApplyTemperatureGeminiOverride()` — User-set Gemini temperature is never overwritten.
- `class TestProviderFallback()` — Edge cases and detection priority.
- `class TestProxyConfigGemini()` — ProxyConfig includes gemini_base_url with correct default.
- `class TestIDERealisticScenarios()` — Real-world request patterns from major IDEs and tools. Each test simulates the EXACT request shape an IDE sends, verifying that detect_provider + extract_user_message + injection all work correctly fo
- `class TestOpenRouterMultiProvider()` — OpenRouter sends ALL models via OpenAI format (/v1/chat/completions). Before the bug fix, model-name detection would route these to the wrong API (e.g., Google's API for gemini models, Anthropic's API
- `class TestStreamingDetection()` — Verify streaming is detected correctly for all providers.
- `class TestEndToEndFormatChain()` — Full pipeline: detect → extract → inject → temperature for each provider.

## Functions
- `def test_anthropic_messages_path(self)`
- `def test_anthropic_messages_path_with_params(self)`
- `def test_gemini_generate_content(self)`
- `def test_gemini_stream_generate_content(self)`
- `def test_openai_chat_completions(self)`
- `def test_openai_completions(self)`
- `def test_path_priority_over_body_format(self)` — Path-based detection beats body-format detection.
- `def test_gemini_path_with_messages_body(self)` — Gemini path wins even if body has messages (shouldn't happen, but safe).
- `def test_gemini_goog_api_key(self)`
- `def test_anthropic_x_api_key_only(self)`
- `def test_anthropic_x_api_key_with_authorization_is_openai(self)` — When both x-api-key and authorization are present, default to openai.
- `def test_openai_authorization_only(self)`
- `def test_gemini_header_priority_over_body(self)` — x-goog-api-key header wins even with OpenAI-format body.
- `def test_native_gemini_contents_format(self)` — Body with 'contents' and no 'messages' → gemini.
- `def test_openai_messages_format(self)` — Body with 'messages' → openai, even if model is gemini.
- `def test_openai_messages_with_claude_model(self)` — Body with 'messages' and claude model → openai (e.g., OpenRouter).
- `def test_body_with_both_contents_and_messages(self)` — If body has both 'contents' and 'messages', prefer openai (messages).
- `def test_empty_body_defaults_openai(self)`
- `def test_no_body_defaults_openai(self)`
- `def test_deepseek_model(self)`
- `def test_mistral_model(self)`
- `def test_ollama_model(self)`
- `def test_openrouter_slashed_model(self)`
- `def test_unknown_model(self)`
- `def test_simple_user_message(self)`
- `def test_multi_part_message(self)`
- `def test_last_user_message(self)`
- `def test_empty_contents(self)`
- `def test_no_contents(self)`
- `def test_mixed_parts_with_non_text(self)` — Non-text parts (images, etc.) are skipped.
- `def test_implicit_user_role(self)` — Gemini defaults role to 'user' when absent.
- `def test_generate_content_url(self)`
- `def test_stream_generate_content_url(self)`
- `def test_body_model_takes_precedence(self)`
- `def test_no_model_no_path(self)`
- `def test_standard_body_model(self)`
- `def test_creates_system_instruction(self)`
- `def test_preserves_contents(self)`
- `def test_does_not_mutate_original(self)`
- `def test_prepends_to_existing(self)`
- `def test_replaces_non_dict_system_instruction(self)`
- `def test_gemini_25_pro(self)`
- `def test_gemini_25_flash(self)`
- `def test_gemini_20_flash(self)`
- `def test_gemini_15_pro(self)`
- `def test_gemini_15_flash(self)`
- `def test_gemini_prefix_match(self)` — Fuzzy prefix matching for dated variants.
- `def test_unknown_model_default(self)`
- `def test_sets_temperature_in_generation_config(self)`
- `def test_preserves_existing_generation_config(self)`
- `def test_does_not_mutate_original(self)`
- `def test_respects_user_temperature(self)`
- `def test_openai_user_override_still_works(self)`
- `def test_empty_path_and_headers(self)`
- `def test_generic_path(self)`
- `def test_path_priority_over_format(self)` — Path detection takes priority over body format.
- `def test_header_priority_over_format(self)` — Header detection takes priority over body format.
- `def test_default_gemini_base_url(self)`
- `def test_custom_gemini_base_url(self)`
- `def test_from_env_gemini_base(self, monkeypatch)`
- `def test_cursor_openai_gpt4o(self)` — Cursor → GPT-4o: standard OpenAI format.
- `def test_cursor_anthropic_claude(self)` — Cursor → Claude: Anthropic messages format.
- `def test_cursor_gemini_via_openrouter(self)` — Cursor → OpenRouter → Gemini: OpenAI format with gemini model name. THIS IS THE CRITICAL BUG FIX TEST. Before the fix, this returned "gemini" which caused routing to Google's API instead of OpenRouter
- `def test_cursor_claude_via_openrouter(self)` — Cursor → OpenRouter → Claude: OpenAI format with claude model name. Another critical multi-provider scenario.
- `def test_vscode_copilot_gpt4o(self)` — VS Code Copilot → GPT-4o: standard OpenAI format.
- `def test_claude_code_anthropic(self)` — Claude Code → Anthropic: native Anthropic format.
- `def test_native_gemini_generate_content(self)` — Google AI Studio → Gemini: native generateContent format.
- `def test_native_gemini_stream(self)` — Google AI Studio → Gemini: streaming via streamGenerateContent.
- `def test_native_gemini_with_system_instruction(self)` — Gemini request with existing systemInstruction — context prepended.
- `def test_jetbrains_openai_format(self)` — JetBrains AI Assistant → OpenAI-compatible format.
- `def test_ollama_local_model(self)` — Ollama → local model: OpenAI-compatible format.
- `def test_deepseek_via_direct_api(self)` — DeepSeek → direct API: OpenAI-compatible format.
- `def test_mistral_via_direct_api(self)` — Mistral → direct API: OpenAI-compatible format.
- `def test_openrouter_gemini_model(self)`
- `def test_openrouter_gemini_bare_model(self)` — Some configs use bare model names without vendor prefix.
- `def test_openrouter_claude_model(self)`
- `def test_openrouter_claude_bare_model(self)`
- `def test_openrouter_deepseek_model(self)`
- `def test_openrouter_full_pipeline(self)` — Full pipeline: detect → extract → inject → temperature for OpenRouter.
- `def test_openai_stream_field(self)`
- `def test_openai_no_stream(self)`
- `def test_gemini_streaming_from_url_path(self)` — Gemini streaming is detected from URL path, not body field.
- `def test_gemini_non_streaming_url(self)` — generateContent (not stream) should not trigger streaming.
- `def test_anthropic_stream_field(self)`
- `def test_openai_full_chain(self)`
- `def test_anthropic_full_chain(self)`
- `def test_gemini_full_chain(self)`
- `def test_openrouter_gemini_full_chain(self)` — OpenRouter with Gemini model — must use OpenAI format throughout.

## Dependencies
- `entroly.proxy_config`
- `entroly.proxy_transform`
- `pathlib`
- `pytest`
- `sys`

## Key Invariants
- test_openrouter_gemini_full_chain: OpenRouter with Gemini model — must use OpenAI format throughout.
