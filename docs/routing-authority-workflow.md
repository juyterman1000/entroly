# Routing Authority Workflow

Entroly already contains the components needed for model-cost routing. This
workflow coordinates them instead of introducing another router.

## Authority order

For each live proxy request:

1. The bounded transport handler admits and caches the request body.
2. The gateway shadow observer may produce a non-authoritative plan.
3. RAVS remains the only model-change proposer.
4. Existing cache economics may reject the proposal before execution.
5. `RoutingAuthorityCoordinator` validates the final proposal:
   - the target remains on the detected provider transport;
   - source and target resolve to the same registry provider;
   - target model metadata is exact and sufficiently trusted;
   - every required capability is proven for the target;
   - the target input/output budget can admit the request;
   - explicit pricing exists for both models by default;
   - the target has a lower estimated uncached cost;
   - active or shadow escalation is not simultaneously authoritative;
   - the merged `GatewayControlPlane` produces the same executable target;
   - no previous model mutation occurred for the request.
6. The existing `apply_target_same_provider` adapter performs the sole rewrite.
7. Existing provider transport, streaming, usage parsing, and spend accounting
   continue unchanged.
8. A bounded receipt records the decision without prompt or credential content.

Any missing evidence denies the model change and lets the existing RAVS guard
forward the original request.

## Activation

The coordinator is compatibility-preserving and disabled by default.

Observe validated proposals without executing them:

```bash
ENTROLY_ROUTING_AUTHORITY=1
ENTROLY_ROUTING_AUTHORITY_MODE=observe
ENTROLY_RAVS_ROUTER=1
```

Execute validated same-provider proposals:

```bash
ENTROLY_ROUTING_AUTHORITY=1
ENTROLY_ROUTING_AUTHORITY_MODE=execute
ENTROLY_RAVS_ROUTER=1
ENTROLY_PRICING_CATALOG=/absolute/path/to/pricing.json
ENTROLY_ESCALATION_MODE=observe
```

Pricing is required by default. It can be disabled only through the explicit
`ENTROLY_ROUTING_AUTHORITY_REQUIRE_PRICING=0` override; production deployments
should keep the requirement enabled.

Response correlation headers are enabled by default and can be disabled with:

```bash
ENTROLY_ROUTING_AUTHORITY_HEADERS=0
```

A protected `GET /routing-authority` sidecar reports bounded receipts and
counters. It never stores prompt text, authorization values, API keys, or raw
request IDs.

## Conflict rules

The workflow permits exactly one model mutation per request.

- Cross-provider targets are rejected.
- `ENTROLY_ESCALATION_MODE=active` and `shadow` conflict with cost-routing
  execution and therefore block the RAVS rewrite.
- JSON-schema and cache-control requests remain on the original model until the
  model registry contains explicit model-level parity fields.
- Announced-only model records may authorize basic text chat when exact model
  resolution and explicit pricing exist. Advanced capabilities require
  verified, discovered, or user-supplied registry metadata.
- Unknown source or target models fail to the original request.

## Evidence semantics

A routing-authority receipt records:

- opaque run and request-correlation identifiers;
- source, proposed, and executed provider/model identifiers;
- required capability names;
- pricing provenance and estimated source/target cost;
- source and target model-decision receipt digests;
- gateway-plan digest and route reason;
- execution, denial, or conflict outcome;
- response status and bounded latency.

The receipt does not claim provider success, answer quality, realized invoice
savings, universal compatibility, or legal compliance. Provider-reported usage
remains the source of truth for realized spend.
