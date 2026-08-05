# Production provider adapter seam

The cache-aware gateway control plane reasons over a canonical request, while
the live proxy receives provider-specific JSON and forwards it to a
provider-specific endpoint. `entroly/provider_adapters.py` is the semantic
boundary between those representations.

## Adapter responsibilities

The adapter converts supported request bodies into `CanonicalGatewayRequest`
for:

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

When the target provider is unchanged, the live proxy can preserve the original
body and replace only the model. `apply_target_same_provider(...)` validates the
target and performs that rewrite. For Gemini, the model is validated and
rewritten in the `/models/{model}` path segment.

## Cross-provider canonical rendering

`render_canonical_request(...)` is a conservative conformance utility. It can
render a limited text-only canonical request for adapter tests and explicitly
fails closed for tools, tool-call history, vision, reasoning controls, response
schemas, and provider-specific controls.

That renderer is **not** permission to route a live request to another provider.
It does not establish operator consent, target credential ownership, region,
retention, contract, billing, or equivalent provider semantics.

`GatewayControlPlane` therefore removes every target whose provider differs
from the original provider and records `cross_provider_disabled` in the
failover receipt. A source-provider failure fails closed rather than invoking
this renderer as an automatic fallback.

## Production invariant

```text
same provider  -> preserve provider body + validated model rewrite
cross provider -> non-executable target (`cross_provider_disabled`)
```

Any future cross-provider product would require a separate operator-authorized
credential, data-policy, contract, region, conformance, and execution boundary.
It must not be introduced by weakening the current gateway invariant.

See [`gateway-provider-boundary.md`](gateway-provider-boundary.md) for the
executable evidence and supported claim boundary.
