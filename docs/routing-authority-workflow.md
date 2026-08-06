# Safe Model Routing Authority

Entroly coordinates existing routing components instead of introducing a second
router. RAVS remains the sole model-change proposer, cache economics may reject
uneconomical proposals, and the routing authority performs the final bounded
same-provider authorization.

The feature is disabled by default. The public workflow is:

1. **Observe** validated proposals without changing the model.
2. **Inspect** bounded routing receipts and denial reasons.
3. **Execute** only after the operator supplies exact provider, origin, model,
   pricing, and credential-authorization controls.

## Authority order

For each live proxy request:

1. The bounded transport handler admits and caches the request body.
2. Gateway shadow mode may record a non-authoritative plan.
3. RAVS proposes at most one model change.
4. Existing cache economics may reject the proposal.
5. The deployment safety boundary verifies:
   - the proxy is loopback-only in execute mode;
   - exactly authorized providers and model IDs are used;
   - the upstream origin is pinned;
   - execute mode uses the official provider API origin;
   - the pricing catalog is absolute, present, valid JSON, bounded in size,
     not world-writable, and complete for every allowlisted model;
   - the operator acknowledged use of an authorized official API credential;
   - RAVS is enabled and no competing escalation authority is active.
6. `RoutingAuthorityCoordinator` verifies:
   - source and target remain on the detected provider transport;
   - source and target resolve to the same registry provider;
   - target metadata is exact and sufficiently trusted;
   - every required capability is proven for the target;
   - the target context budget admits the request;
   - explicit pricing exists for source and target;
   - the target has lower estimated uncached cost;
   - `GatewayControlPlane` produces the same executable target;
   - no previous model mutation occurred for the request.
7. The existing provider adapter performs the sole model rewrite.
8. Existing provider transport, streaming, usage parsing, and spend accounting
   continue unchanged.
9. A bounded receipt records execution or denial without prompt or credential
   content.

Any missing or conflicting evidence keeps the original model.

## Recommended activation

### 1. Observe first

OpenAI:

```bash
entroly proxy \
  --routing observe \
  --provider openai
```

Anthropic:

```bash
entroly proxy \
  --routing observe \
  --provider anthropic
```

Gemini:

```bash
entroly proxy \
  --routing observe \
  --provider gemini
```

Observe mode never changes the requested model. It pins the provider origin and
records why a proposal would have been executed or denied.

### 2. Inspect evidence

```bash
entroly routing status
```

The protected local sidecar is also available at:

```text
http://127.0.0.1:9377/routing-authority
```

Receipts include provider/model identifiers, pricing provenance and digest,
estimated source/target cost, capability requirements, gateway-plan evidence,
and the final execution or denial reason. They exclude prompts, authorization
values, API keys, and raw request IDs.

### 3. Prepare pricing

Execution requires an absolute JSON path and exact entries for every allowlisted
model:

```json
{
  "source": "operator-verified-2026-08-06",
  "models": {
    "openai:gpt-4o": {
      "input_per_million": 10.0,
      "output_per_million": 30.0,
      "cache_read_per_million": 5.0
    },
    "openai:gpt-4o-mini": {
      "input_per_million": 0.15,
      "output_per_million": 0.60,
      "cache_read_per_million": 0.075
    }
  }
}
```

The values above are an illustrative schema, not a current pricing claim. The
operator must use provider-published prices applicable to their account.

### 4. Enable execution explicitly

```bash
entroly proxy \
  --routing execute \
  --provider openai \
  --allow-model gpt-4o \
  --allow-model gpt-4o-mini \
  --pricing-catalog /absolute/path/to/pricing.json \
  --ack-authorized-api
```

The acknowledgement means the operator is using an official API credential they
or their organization is authorized to use. Entroly does not inspect, persist,
or copy the credential into receipts.

Before starting the proxy, validate an environment-based deployment with:

```bash
entroly routing check
```

## Environment equivalent

Observe mode:

```bash
export ENTROLY_ROUTING_AUTHORITY=1
export ENTROLY_ROUTING_AUTHORITY_MODE=observe
export ENTROLY_ROUTING_AUTHORITY_ALLOWED_PROVIDERS=openai
export ENTROLY_ROUTING_AUTHORITY_ALLOWED_ORIGINS=openai=https://api.openai.com
export ENTROLY_RAVS_ROUTER=1
entroly serve --proxy
```

Execute mode additionally requires:

```bash
export ENTROLY_ROUTING_AUTHORITY_MODE=execute
export ENTROLY_ROUTING_AUTHORITY_ALLOWED_MODELS=openai:gpt-4o,openai:gpt-4o-mini
export ENTROLY_PRICING_CATALOG=/absolute/path/to/pricing.json
export ENTROLY_ROUTING_AUTHORITY_REQUIRE_PRICING=1
export ENTROLY_ROUTING_AUTHORITY_ACK=authorized-official-api
export ENTROLY_ESCALATION_MODE=observe
entroly serve --proxy
```

## Startup refusal conditions

Execute mode refuses to start when any of these conditions is true:

- the bind host is not loopback;
- no provider or upstream-origin pin is configured;
- the configured proxy base origin differs from the pin;
- the pinned execute origin is not the official provider API origin;
- fewer than two exact models are allowlisted;
- a model belongs to a provider outside the provider allowlist;
- pricing is disabled, missing, malformed, incomplete, non-numeric, negative,
  too large, or world-writable;
- the API-authorization acknowledgement is absent;
- RAVS is disabled;
- active or shadow escalation would create competing routing authority.

## Request-time refusal conditions

Even after startup, the original model remains unchanged when:

- the source, target, or proxy provider is not allowlisted;
- the source or target model is not allowlisted;
- the request upstream does not match the pinned origin;
- provider, registry, capability, context-budget, pricing, or gateway evidence
  fails;
- a second mutation is attempted;
- the adapter would change the URL origin.

## Explicit non-goals

This workflow does not:

- route across providers;
- switch credentials, accounts, organizations, or billing owners;
- use consumer subscription cookies or browser sessions;
- enable remote or multi-tenant execution;
- add automatic retries;
- infer unsupported JSON-schema or cache-control parity;
- claim provider approval, legal compliance, answer-quality retention, or
  realized invoice savings.

Provider-reported usage remains the source of truth for realized spend.
