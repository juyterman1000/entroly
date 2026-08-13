# Traffic Receipts

Entroly's live proxy exposes a content-blind, per-request **Traffic Receipt** that
joins the context, cache, routing, provider-usage and verification signals the
proxy already produces.

Open the live view while the proxy is running:

```text
http://127.0.0.1:9377/traffic
```

The bounded JSON surface is:

```text
GET /traffic-receipts?limit=20
```

A receipt can show:

```text
Claude Code request

Original context       73,440 tokens
Entroly context         18,206 tokens
──────────────────────────────────
Tokens avoided          55,234

Evidence retained       —      (withheld unless request-local evidence exists)
Recovery evidence       YES    (only when recovery evidence is present)
Warm prefix protected   14,820 tokens
Cache hit               YES    (provider-reported cache usage)

Requested model         Sonnet
Executed model          Sonnet
Routing decision        STAY
Reason                   ...

Input cost              $...   (provider usage + explicit pricing required)
Cache benefit           $...   (provider cache usage + explicit pricing required)
Net measured saving     —      (requires a linked measured counterfactual)

Context risk            UNKNOWN
Verification            PASS

Receipt integrity       ✓ SHA-256 OK
```

The numbers above are an illustrative shape, not bundled benchmark or savings
claims. The live page itself contains no hard-coded demonstration values.

## Truth contract

Traffic Receipts intentionally distinguish measurement from modeling:

- **Original / Entroly context tokens** are deterministic local estimates from
  the provider canonicalization layer.
- **Tokens avoided** is the difference between the admitted request estimate and
  the final outbound request estimate.
- **Evidence retained** is emitted only when a request-local evidence source is
  available. Shared mutable proxy `last_*` coverage state is deliberately
  withheld to prevent cross-request attribution under concurrency.
- **Recovery evidence** becomes true only when per-request recovery evidence is
  present in the proxy's recovery headers. It does not claim that exact recovery
  was available before an intervention.
- **Warm prefix protected** comes from the existing hash-only
  `PrefixContinuityGuard` intervention for that request.
- **Cache hit** prefers provider-reported cache-read usage. For streaming calls,
  Entroly can also observe the provider-updated cache lease after the stream
  completes.
- **Input cost / cache benefit** require provider-reported usage and an explicit,
  auditable pricing catalog.
- **Net measured saving** is deliberately unavailable until Entroly has a
  request-correlated measured counterfactual. Modeled token reduction is not
  relabeled as realized invoice savings.
- **Verification** comes from WITNESS/EICV response evidence when reported.
- **Receipt integrity** is a SHA-256 canonical-payload consistency check. It is
  not presented as a cryptographic signature.

## Correlation and recovery ownership

When the client does not provide `X-Request-ID`, Entroly establishes one bounded
request identifier before the core proxy handles the request. The receipt and
provider usage accounting therefore observe the same request id.

Recursive exact-recovery attempts contribute to the final outer response but do
not finalize a separate buyer-visible receipt. Depth zero owns the Traffic
Receipt lifecycle.

## Privacy contract

The receipt ledger is bounded and in-memory. Receipts do **not** contain:

- prompt or message content;
- tool output or code;
- authorization header values or API keys;
- the raw request ID;
- the raw user-agent.

The raw request ID is reduced to a salted correlation digest. The user-agent is
used only to classify a product label such as `Claude Code` or `Codex` and is not
stored.

Every receipt has a SHA-256 digest over a canonical JSON payload and that digest
is checked before the receipt is admitted to the in-memory ledger.

## Scope

Traffic Receipts are an observability/product surface. They do not add a new
router, retry loop, provider policy, cache engine, optimizer or execution path.
They observe the existing hardened proxy and expose its evidence in one place.
