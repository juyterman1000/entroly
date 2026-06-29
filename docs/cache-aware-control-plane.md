# Cache-aware gateway control plane

Entroly treats a warm provider prefix cache as a priced, expiring asset. Model
routing therefore minimizes projected billed cost rather than nominal token
price.

## Decision model

For candidate model (m):

[
C_m =
T_p R_{p,m} +
T_n R_{i,m} +
T_o R_{o,m}
]

Where:

- (T_p) is stable-prefix tokens,
- (T_n) is new uncached input,
- (T_o) is expected output,
- (R_p) is cache-read price when the lease is warm and cache-write/input
  price otherwise.

The router projects this across a bounded turn horizon. It switches only when
the projected saving exceeds hysteresis and the candidate remains inside the
risk-specific quality tolerance.

Provider failure and explicit escalation override cache stickiness.

## Stable prefix

`CanonicalPrefixBuilder` creates a deterministic cacheable prefix. Sections
are ordered by priority and name, mappings use canonical JSON, tools are sorted
by semantic identity, and dynamic request content is kept outside the prefix.

```python
from entroly import CanonicalPrefixBuilder

prompt = (
    CanonicalPrefixBuilder(version="1")
    .add("policy", {"mode": "coding"}, priority=10)
    .add_tools(tool_schemas)
    .add("repository", repo_map, priority=40)
    .build(dynamic_tail=current_user_message)
)
```

Changing the dynamic tail does not change `prompt.prefix_hash`.

## Cache-aware routing

```python
from entroly import CacheAwareRouter, CachePrice, ModelCandidate

router = CacheAwareRouter()
router.observe(
    conversation_id,
    model="primary",
    provider="openai",
    prefix_hash=prompt.prefix_hash,
    cached_prefix_tokens=80_000,
    cache_hit=True,
)

decision = router.decide(
    conversation_id,
    current_model="primary",
    candidates=[
        ModelCandidate(
            "primary",
            "openai",
            CachePrice(10, 1, 30),
            quality=1.0,
        ),
        ModelCandidate(
            "economy",
            "anthropic",
            CachePrice(3, 0.3, 15),
            quality=0.99,
        ),
    ],
    prefix_hash=prompt.prefix_hash,
    prefix_tokens=80_000,
    new_input_tokens=1_000,
    expected_output_tokens=2_000,
)
```

Cache hits are observations from provider usage; they are never fabricated from
similarity alone.

## Capability-safe failover

`ProviderFailoverPlanner` excludes unhealthy targets, open circuits, and
models that cannot satisfy required streaming, tools, JSON schema, vision, or
reasoning capabilities. Failover order is deterministic.

Outbound redaction is disabled by default. When explicitly enabled,
`GatewayRedactionPolicy` replaces configured secret/PII patterns and emits a
receipt containing only salted digests and counts, never the matched values.

## Durable spend ledger

`UsageLedger` stores idempotent request records in SQLite WAL mode. It parses:

- OpenAI/Azure cached-token details,
- Anthropic cache-read and cache-creation tokens,
- Gemini cached-content tokens.

Costs use integer microdollars and an explicit `UsagePricing` source. The
ledger reports actual provider token categories separately from estimated ELC
savings.

## Coding harness budgets

`CodingHarnessBudgetController` reserves the stable prefix, dynamic request,
and retrieval capacity before allocating the remainder to subagents.

Subagent utility is concave:

[
U_i(x) = w_i log(1 + x/s_i)
]

A deterministic discrete water-filling algorithm repeatedly assigns the next
token quantum to the highest feasible marginal utility while respecting each
agent's minimum, maximum, token budget, and optional spend ceiling.

## Integrated use

`GatewayControlPlane` composes all gateway decisions without performing
network I/O:

```text
canonical request
  -> opt-in redaction
  -> capability-safe failover plan
  -> cache-aware model decision
  -> transport execution
  -> provider usage observation
  -> cache lease + durable spend ledger
```

The existing proxy can adopt the control plane incrementally because planning
and accounting are transport-independent.
