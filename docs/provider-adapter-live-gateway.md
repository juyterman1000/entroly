# Production provider adapter seam

The cache-aware gateway control plane is provider-neutral, while the live proxy receives provider-specific JSON payloads and forwards them to provider-specific endpoints. `entroly/provider_adapters.py` is the boundary between those worlds.

## Adapter responsibilities

The adapter converts supported request bodies into `CanonicalGatewayRequest` for:

- OpenAI Chat Completions
- OpenAI Responses API
- Anthropic Messages API
- Gemini generate and stream-generate endpoints

The canonical view keeps only fields the control plane can reason about safely:

- model
- messages
- tools
- streaming requirement
- response schema requirement
- portable attribution metadata
- vision requirement
- reasoning-control requirement

It also computes the estimates used by cache-aware routing:

- cacheable prefix tokens
- new input tokens
- expected output tokens

## Same-provider routing

If the target provider is unchanged, the live proxy can preserve the original body and replace only the model. `apply_target_same_provider(...)` validates the target and performs that rewrite. For Gemini, the model is validated and rewritten in the `/models/{model}` path segment.

## Cross-provider failover

A cross-provider failover target must not reuse the original provider body. Provider-specific generation controls and tool formats are not interchangeable.

For that reason, cross-provider execution must render from the canonical request using `render_canonical_request(...)`. That renderer emits only portable fields with equivalent semantics. Provider-specific fields are deliberately dropped unless a future explicit adapter implements a verified translation.

## Production invariant

```text
same provider  -> preserve body + validated model rewrite
cross provider -> render from canonical request only
```

This prevents silent semantic drift while still allowing a future enterprise transport layer to execute capability-safe cross-provider failover.
