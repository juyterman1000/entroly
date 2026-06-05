# Provider Compliance Notes

This is an engineering checklist, not legal advice. Entroly is local middleware:
it can reduce and annotate the prompt content sent to a configured provider, but
it does not override that provider's terms, data policy, rate limits, model
availability, or safety policy.

## Default Stance

- Use documented provider configuration only: custom base URLs, BYOK settings,
  MCP settings, or native SDK/API endpoints.
- Do not claim data is fully local when the user has configured a cloud LLM API.
  In cloud proxy mode, the selected prompt content is sent to that provider.
- Do not silently substitute models by default. `ENTROLY_RAVS_ROUTER=1` is
  required before RAVS can route a request to a cheaper model.
- Do not enable federation, cloud NLI, escalation, or external webhooks without
  explicit user configuration.
- Keep provider-owned generation controls pass-through. The proxy must not
  rewrite `temperature`, `thinking`, `reasoning_effort`, `top_p`, or equivalent
  controls unless the user explicitly asks for that behavior.

## Provider References Checked

| Provider/tool | Official reference | Entroly implication |
|---|---|---|
| OpenAI API | https://developers.openai.com/api/docs/guides/your-data and https://developers.openai.com/api/reference/overview | Use bearer API keys securely. Cloud API calls are subject to OpenAI data controls and usage policies. |
| Anthropic Claude API | https://platform.claude.com/docs/en/build-with-claude/extended-thinking and https://platform.claude.com/docs/en/api/openai-sdk | Preserve thinking/adaptive-thinking fields. Native API-compatible forwarding should remove only transport fields the public API rejects. |
| Claude Code | https://code.claude.com/docs/en/settings | Use documented settings and environment variables. Prefer MCP mode when a proxy override is not documented or supported by the installed version. |
| GitHub Copilot CLI | https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/use-byok-models | BYOK custom provider configuration is documented through `COPILOT_PROVIDER_BASE_URL`, `COPILOT_PROVIDER_TYPE`, `COPILOT_PROVIDER_API_KEY`, and `COPILOT_MODEL`. |
| Google Gemini API | https://ai.google.dev/gemini-api/docs/openai and https://ai.google.dev/gemini-api/terms | OpenAI-compatible base URL is documented. Paid and unpaid Gemini API data use differs; users must choose the right billing/project mode. |
| Google Vertex AI | https://cloud.google.com/vertex-ai/generative-ai/docs/data-governance | Vertex AI has separate enterprise data-governance terms from AI Studio/Gemini API. |
| OpenRouter | https://openrouter.ai/docs/api/reference/overview and https://openrouter.ai/terms | OpenRouter forwards to model providers and requires users to comply with the selected model provider terms. |
| Groq | https://console.groq.com/docs/openai | OpenAI-compatible base URL is documented; unsupported parameters can produce provider errors. |
| Together AI | https://docs.together.ai/docs/openai-api-compatibility and https://docs.together.ai/docs/privacy-and-security | OpenAI-compatible base URL is documented; data handling is governed by Together's platform settings and terms. |
| Amazon Bedrock | https://docs.aws.amazon.com/bedrock/latest/userguide/data-protection.html | Bedrock has AWS-specific data-protection and IAM requirements; treat it as a separate provider path, not generic Anthropic/OpenAI. |

## Current Product Controls

- Local deterministic paths: indexing, selection, dashboard metrics, WITNESS/EICV
  without NLI, PRISM learning, autotune, and MCP context serving.
- Cloud provider paths: proxy forwarding to configured LLM APIs; optional
  `--witness-nli` OpenAI entailment checks; user-configured integration
  webhooks; optional federation transport when enabled.
- Privacy guardrails: no Entroly-owned backend, no analytics SDKs, local storage
  in `.entroly/`, secret redaction for internal logs.

## Image Inputs

Image optimization must stay opt-in. Entroly may estimate provider image tokens
from published formulas and recommend safer dimensions, but provider usage
metadata or token-count APIs remain the billing source of truth. Do not claim
exact image billing unless it comes from the provider response or official
token-count endpoint. See `docs/limitations.md`.

## Release Checklist

1. Run `entroly doctor --privacy`.
2. Run provider-shape tests for OpenAI, Anthropic, Gemini, OpenRouter, and any
   documented BYOK wrapper touched by the change.
3. Confirm `ENTROLY_RAVS_ROUTER`, `ENTROLY_FEDERATION`, and
   `ENTROLY_WITNESS_NLI` remain opt-in.
4. Confirm privacy/docs copy says "no Entroly phone-home" rather than "all data
   never leaves your machine" when cloud proxy mode is supported.
5. Re-check official provider docs before adding a new wrapper or base-URL path.
