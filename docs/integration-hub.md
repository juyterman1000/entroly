# Entroly integration and operations hub

This page distinguishes direct, tested adapters from generic compatibility and
guided setup. **Provider-bound token and cost evidence exists only when the
model request actually traverses an Entroly proxy or another measured
Entroly-controlled request path.** MCP attachment alone does not imply hosted
subscription inference was intercepted.

## Status definitions

| Status | Contract |
|---|---|
| **Direct and tested** | Entroly ships a named adapter, plugin or wrapper and exercises its request contract in the repository test suite. |
| **Supported with boundary** | A tested Entroly path exists, but authentication, subscription or provider routing limits what can be intercepted. |
| **Guided compatibility** | The framework can use Entroly's SDK, middleware or custom endpoint, but no dedicated framework adapter is claimed. |
| **Validation pending** | A plausible route exists, but Entroly does not advertise end-to-end support until a watchdog-backed request test passes. |

## SDK and framework integrations

### Vercel AI SDK

**Direct and tested middleware shape.** The `entroly-wasm` package exports
`createEntrolyMiddleware`, provider-neutral request optimization, and typed
OpenAI, Anthropic and Gemini helpers. The middleware transforms request params
before generation; streaming response behavior remains owned by the AI SDK.

Code evidence: `entroly-wasm/js/app_sdk.js`. Contract test:
`entroly-wasm/test_app_sdk.js`.

### OpenAI SDK

**Direct and tested JavaScript wrapper plus proxy route.** `wrapOpenAI(client)`
intercepts chat-completion and Responses-style request parameters while leaving
the caller's client responsible for authentication, transport and responses.
Python and other SDKs can point a custom `base_url` at the loopback Entroly
proxy.

Code evidence: `entroly-wasm/js/app_sdk.js`; provider/proxy regressions:
`tests/test_proxy_providers.py` and `tests/test_control_plane.py`.

### Anthropic SDK

**Direct and tested JavaScript wrapper plus native Anthropic proxy route.**
`wrapAnthropic(client)` preserves Anthropic content-block and tool contracts
while optimizing message content before the request. Claude subscription OAuth
sessions are not treated as public-API keys; use scoped MCP for subscription
sessions and the proxy only with a supported API or enterprise route.

Code evidence: `entroly-wasm/js/app_sdk.js`; boundary tests:
`tests/test_subscription_guard.py` and `tests/test_proxy_providers.py`.

### LangChain

**Direct and tested Python adapters.** `EntrolyCompressor` preserves concrete
message types, tool calls, IDs and provider metadata across sync, async, batch
and stream methods. `EntrolyDocumentCompressor` implements the retriever
compressor protocol while preserving document metadata and identity.

Code evidence: `entroly/integrations/langchain.py`. Contract tests:
`tests/test_langchain_deep_integration.py`.

### LiteLLM

**Direct and tested pre-call hook.** `EntrolyLiteLLMCallback` implements the
LiteLLM proxy `async_pre_call_hook` shape and preserves controls and tools while
compressing supported message/input arrays.

Code evidence: `entroly/integrations/litellm.py`. Contract tests:
`tests/test_framework_request_adapters.py`.

### Agno

**Guided compatibility; no dedicated Agno adapter is claimed.** Compress the
messages before handing them to an Agno model, or route a BYOK/custom-endpoint
model through the Entroly proxy. Verify the route with Entroly's watchdog or
value receipt before claiming provider-bound savings.

### Strands Agents

**Guided compatibility; no dedicated Strands hook is claimed.** Use the Python
SDK at the message-assembly boundary or a model provider that permits a custom
base URL. Native Bedrock/enterprise routing remains outside Entroly unless the
request is explicitly routed through a measured Entroly path.

### CrewAI

**Guided compatibility; no dedicated CrewAI adapter is claimed.** Use
`compress_messages` before task execution or a CrewAI/LiteLLM provider route
configured to traverse Entroly. CrewAI callbacks alone do not prove inference
interception.

### AutoGen

**Guided compatibility; no dedicated AutoGen model client is claimed.** Use the
framework-neutral SDK before model invocation or configure an OpenAI-compatible
client endpoint through Entroly. Hosted services that do not permit a custom
endpoint are not intercepted.

## Agent, IDE and provider integrations

### MCP

**Direct and tested.** Entroly ships Python, npm and focused compression MCP
entry points with context selection, receipts, exact recovery, verification and
repository-intelligence tools. MCP gives the host tools; it does not silently
rewrite every model request made by that host.

Start with `entroly attach create --project . --ttl 4h --install` or the
client-specific setup in [agent compatibility](agent-compatibility.md).

### OpenClaw

**Direct and tested first-class plugin.** The published `entroly-openclaw`
package provides a ContextEngine path, local bridge, receipts and optional
proof-guided recovery. OpenClaw retains its agent loop and provider credentials.

See [OpenClaw context engine](openclaw-context-engine.html).

### OpenCode

**Direct and tested package integration.** Entroly provides local MCP setup and
a compaction hook that preserves commands, errors, paths, symbols, verification
status and exact-recovery handles.

See [OpenCode context assurance](opencode-context-assurance.html).

Entroly does not publish a separate model-specific OpenCode + DeepSeek claim in
this matrix. A custom model route must satisfy the same endpoint,
authentication, request-shape and watchdog requirements as every other
OpenAI-compatible provider.

### Claude Code on Vertex AI

**Supported MCP context path; provider interception not validated.** Scoped MCP
attachment works independently of whether Claude Code uses Anthropic or Vertex
AI. Entroly does not claim that Vertex inference traversed its proxy unless an
enterprise route is explicitly configured and observed.

### Claude Code on Azure AI Foundry

**Supported MCP context path; provider interception not validated.** Claude
Code can use Entroly's scoped MCP tools while Azure retains identity, routing
and inference. A separate tested custom endpoint is required before claiming
provider-bound token or dollar evidence.

### Claude Code in VS Code

**Supported with boundary.** Install Entroly as a scoped MCP server for the
workspace. The extension and Claude Code retain their own subscription and
provider behavior; MCP availability is not presented as transparent inference
compression.

### VS Code Copilot

**Supported with mode boundary.** Signed-in Copilot sessions can use scoped MCP
tools. Provider-bound proxy measurement is available only for a separately
configured BYOK/custom-provider route; Entroly does not claim interception of
GitHub-hosted subscription inference.

See [GitHub Copilot CLI boundaries](agent-compatibility.md#github-copilot-cli).

### Grok

**Guided BYOK; broader validation pending.** A custom OpenAI/Anthropic-compatible
model entry can point to Entroly when the selected Grok client exposes a custom
endpoint. Default signed-in inference is not claimed as intercepted, and the
configured upstream must be explicit.

See [Grok CLI boundary](agent-compatibility.md#grok-cli).

## Configuration, observability and operations

| Need | Entroly documentation |
|---|---|
| Installation and configuration | [First-run trust](first-run-trust.md) and [engine/install options](DETAILS.md#engine--install-options) |
| Context and filesystem contract | [Context control plane](context-control-plane.md) and [reliable event delivery](reliable-event-delivery.md) |
| Savings tracking | [Live tokenomics](live-tokenomics.md), `entroly value`, and `entroly dashboard` |
| Metrics and monitoring | [Grafana/Prometheus dashboard](grafana/README.md) and [gateway accounting](gateway-accounting.md) |
| Simulation | `entroly simulate`, `entroly perf`, and [benchmark protocols](BENCHMARKS.md) |
| API and product surface | [Product surface](product-surface.md) and [command reference](DETAILS.md#command-reference) |
| Architecture | [Architecture](architecture.md) and [verified system map](architecture/ENTROLY_VERIFIED_MAP.md) |
| Releases and CI/CD | [Release notes](releases/v1.0.77.md) and repository workflows under `.github/workflows/` |
| Limitations | [Explicit limitations](limitations.md) |
| Errors and troubleshooting | `entroly doctor`, [first-run diagnostics](first-run-trust.md), and [MCP troubleshooting](mcp-server-guide.html#troubleshooting) |

## Verification policy

Compatibility is not inferred from a framework name. Before promoting a guided
row to direct support, Entroly requires:

1. a code-backed setup or adapter;
2. preservation of tools, controls and provider message shapes;
3. a regression or watchdog-backed end-to-end request test;
4. an explicit authentication and subscription boundary;
5. visible failure behavior and a way to prove the request reached Entroly.
