# Framework and SDK context compression integrations

Entroly provides two different application SDK paths. They must not be described as equivalent.

## Receipt-first gateway path

The recommended assurance path calls the local `/v1/compress` endpoint. The endpoint changes a payload only after exact omitted content is persisted and returns Entroly recovery headers. Python exports `CompressionGatewayClient`, `wrap_openai`, `wrap_anthropic`, and `EntrolyLiteLLMHook`. Node/TypeScript exports `EntrolyGatewayClient`, `createGatewayMiddleware`, and provider wrappers ending in `WithGateway`.

```python
from entroly.integrations import CompressionGatewayClient, wrap_openai

gateway = CompressionGatewayClient(budget_tokens=12_000)
client = wrap_openai(openai_client, gateway=gateway)
response = client.chat.completions.create(model="gpt-5", messages=messages)
```

```javascript
const { EntrolyGatewayClient, wrapOpenAIWithGateway } = require("entroly-wasm");
const gateway = new EntrolyGatewayClient({ budgetTokens: 12000 });
const client = wrapOpenAIWithGateway(openai, { gatewayClient: gateway });
```

Non-loopback gateway URLs require explicit `allow_remote` / `allowRemote` authorization. The proxy must also be started in its explicit authenticated remote mode with a trusted TLS/tunnel transport. The SDK `access_token` / `accessToken` option sends the proxy-wide `X-Entroly-Access-Token` capability; `sidecar_token` / `sidecarToken` sends `X-Entroly-Sidecar-Token` when the sidecar guard is configured. Remote deployments need both configured capabilities. This prevents an integration from silently adding a new prompt destination.

## LiteLLM Proxy hook

`EntrolyLiteLLMHook.async_pre_call_hook(...)` follows LiteLLM's proxy pre-call contract and transforms only completion/text-completion request types. Configure the exported instance through LiteLLM's `litellm_settings.callbacks`. The hook exposes the last Entroly receipt headers through LiteLLM's response-header hook.

## Dependency-free local helper

The older Node/WASM `optimize*Params` helpers remain dependency-free and local, but use deterministic middle compaction rather than the proxy recovery store. They are suitable only where that limitation is acceptable. Do not describe them as exact-recovery middleware.
