# Entroly agent compatibility

Entroly integrates through four distinct paths. A client name in this document does **not** imply that every subscription, OAuth flow, model, or provider route is intercepted.

1. **Native attachment or plugin** — the host registers Entroly as a scoped MCP or context engine.
2. **One-command proxy launch** — Entroly starts a local proxy, sets the client's supported endpoint variables, and launches the client.
3. **Automatic MCP configuration** — Entroly writes or merges the client's project/user MCP configuration.
4. **Guided custom endpoint** — Entroly starts the proxy and prints a client-specific configuration block rather than mutating a versioned third-party schema.

Provider-bound token and cost measurements exist only when the request actually traverses the Entroly proxy. MCP-only integrations expose context selection, receipts, exact recovery, and verification tools, but do not automatically intercept every inference request made by the host.

## Compatibility matrix

| Agent or platform | Best Entroly path | Status | Authentication and routing boundary |
|---|---|---|---|
| Claude Code | Scoped MCP attachment; API-key proxy wrap | Native | Claude Pro/Max subscription sessions should use MCP. Public-API proxying requires `ANTHROPIC_API_KEY`. |
| Codex CLI | Scoped MCP attachment; API-key proxy wrap | Native | ChatGPT-account mode can bypass `OPENAI_BASE_URL`; provider-bound measurement requires API-key or custom-provider routing. |
| GitHub Copilot CLI | MCP for subscription sessions; custom-provider proxy for BYOK | Supported with mode boundary | MCP works with the signed-in CLI. Entroly does not claim interception of GitHub-hosted subscription inference. |
| OpenClaw | ContextEngine plugin and scoped MCP attachment | Native | OpenClaw retains provider authentication; Entroly assembles context and emits receipts. |
| Cursor | Automatic project MCP config; optional custom proxy endpoint | Automatic MCP | Restart Cursor after configuration. Proxy accounting exists only when the model route points through Entroly. |
| Aider | Session-scoped OpenAI-compatible proxy | One command | Requires an API/provider route that accepts a custom endpoint. |
| OpenCode | Session-scoped OpenAI-compatible proxy | One command | Provider authentication remains owned by OpenCode and its upstream. |
| Gemini CLI | Session-scoped Gemini-compatible proxy | One command | Requires `GEMINI_API_KEY` for the provider-bound proxy path. |
| Qwen Code | Session-scoped OpenAI-compatible proxy | One command | Confirm the selected provider honors `OPENAI_BASE_URL`. |
| Cline | Printed OpenAI-compatible endpoint settings | Guided setup | Entroly does not silently mutate extension settings whose schema may change by version. |
| Continue | Generated provider snippet | Guided setup | Confirm the active model entry uses the Entroly base URL. |
| Grok CLI | Custom model entry pointed at Entroly | Guided BYOK | Grok custom endpoints use API-key auth. Default signed-in inference is not claimed as intercepted. Entroly's OpenAI-compatible upstream must be set to the intended xAI endpoint. |
| Goose | OpenAI-compatible endpoint | Compatible; end-to-end validation pending | Do not label one-command support until a watchdog-backed request test passes. |
| OpenHands | Local CLI `LLM_BASE_URL` route | Compatible; end-to-end validation pending | A remote/cloud sandbox cannot reach a laptop-local proxy. |
| Mistral Vibe | Generic OpenAI-style provider in `config.toml` | Guided setup | Use a project or user provider entry with the Entroly endpoint. |
| Oh My Pi | Custom provider in `~/.omp/agent/models.yml` | Guided setup | Stored OAuth credentials are not assumed to be valid for an arbitrary proxy. |
| Kimi CLI | Native MCP registration; custom provider where configured | MCP-compatible | OAuth inference passthrough is not claimed until independently tested. |
| ZCode | Custom OpenAI/Anthropic-compatible base URL | Guided setup | Confirm the chosen provider and authentication mode permit a custom URL. |
| Cortex Code | SDK/library boundary only | Not validated as a wrap target | No official, tested endpoint contract is currently documented by Entroly. |

## GitHub Copilot CLI

Entroly supports two different Copilot CLI use cases and keeps them separate.

### Signed-in subscription session: MCP attachment

Use Entroly as an MCP server while Copilot continues to own its hosted model session:

```bash
entroly attach create --client copilot --project . --ttl 4h --install
```

This path can provide scoped context selection, Context Receipts, exact recovery, and verification tools. It does not claim that Copilot's hosted inference request passed through the Entroly proxy.

### BYOK/custom provider: proxy route

Current Copilot CLI custom-provider settings use `COPILOT_PROVIDER_BASE_URL`, `COPILOT_PROVIDER_TYPE`, `COPILOT_PROVIDER_API_KEY`, and `COPILOT_MODEL`. The Entroly wrapper must use those variables and the current `copilot` executable. This route is separately billed by the configured provider; it is not GitHub-hosted subscription inference.

## Grok CLI

Grok supports custom OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages endpoints in `~/.grok/config.toml`. A custom model can point at Entroly, but the upstream must also be configured explicitly. Grok's `GROK_MODELS_BASE_URL` mode fetches `{base_url}/models`; Entroly should not advertise that environment-variable path as one-command support until its model-list contract is tested end to end.

## Generic OpenAI-compatible clients

A client is proxy-compatible when all of the following are true:

- it accepts a custom API base URL;
- it sends a request shape supported by Entroly;
- its authentication or subscription terms permit local routing;
- Entroly is configured with the correct upstream provider;
- a watchdog, receipt, or proxy statistic confirms the request actually reached Entroly.

Compatibility is not inferred merely because a client uses the word “OpenAI-compatible.”

## Status policy

A green status requires a code-backed setup path and a regression or end-to-end contract. Clients with documented custom endpoints but no Entroly request-flow test remain **guided** or **validation pending**. Hosted subscription interception is never inferred from MCP compatibility.