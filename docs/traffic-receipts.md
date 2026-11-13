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
- **Net saving (conservative)** (`net_saving_conservative_micro_usd`, with
  `net_saving_basis`) is a provider-calibrated *estimate*, not a measured
  counterfactual. It is the **headline dollar figure** on Traffic Value, on
  `/stats` and in `entroly value`: summed, signed, over the requests that have
  it, reported with how many requests that was, and never combined with the
  list-price model. It prices only what Entroly changed, with every unknown
  resolved against Entroly:
  - removed content at the cache-read rate if any recent request carried it,
    otherwise at the uncached input rate;
  - added content (injected context, rewritten blocks) and any cached prefix
    Entroly's edits broke, drawn from the tokens the provider did not serve
    from cache, dearest first: reported 1-hour writes, then 5-minute writes,
    then uncached input. When the provider reports no 5-minute/1-hour split,
    every write is assumed to be a 1-hour write. Anything beyond that pool is
    charged at the cache-read rate;
  - bytes converted to tokens with the provider's own count of the forwarded
    request, using the ratio that under-counts removals and over-counts
    additions.
  An unchanged request is exactly zero, the provider's own cache discount is
  never credited, and the value can be negative. It needs provider usage, a
  prefix-continuity observation and a priced model; otherwise it is absent,
  not zero. Prices come from `ENTROLY_PRICING_CATALOG` when set, else the
  bundled tables in `entroly/data/` (disable with `ENTROLY_BUNDLED_PRICING=0`).
  The bundled tables are used for this estimate only and never satisfy
  routing's explicit-catalog rule.

  | Table | Provider key | Source |
  |---|---|---|
  | `anthropic_pricing.json` | `anthropic` | platform.claude.com pricing page |
  | `openai_pricing.json` | `openai` | developers.openai.com pricing and model pages |
  | `google_pricing.json` | `gemini` | ai.google.dev Gemini API pricing page |

  Each table carries `source_url` and `verified_at`, and the receipt's
  `net_saving_basis.pricing_source` names the table and its date. A table
  lists only models whose standard text price was read from the provider's
  own page on that date, under their exact API ids (a dated snapshot only
  where the provider documents it at the same price). Anything else has no
  price and the request makes no dollar claim:
  - a model the table does not list (for example `gpt-5.6-sol`, whose page
    price was marked promotional);
  - a request larger than the size the table was verified for
    (`priced_up_to_input_tokens`): OpenAI's page has a long-context switch
    with no stated threshold, so those models are priced up to 200,000 input
    tokens and not above;
  - batch, flex, priority, regional-uplift, audio and image pricing.

  Two provider-specific rules, both against Entroly:
  - OpenAI lists 30-minute and 12-hour cache-write prices for its newest
    models without saying which usage field reports written tokens
    (`cache_writes_unreported`). Every uncached token Entroly is charged for
    on those models is priced as a 12-hour write; removed content is still
    credited at the plain input rate or the cached-input rate.
  - Gemini prompts above 200,000 tokens bill at the long-context rate, which
    sets the price of what Entroly added; removed content is credited at the
    base rate. The hourly storage price of an explicit Gemini cache is not
    modelled, since an explicit cache is addressed by name and Entroly's
    edits neither create nor break one.

  Measured by replaying a 25-turn agent session per provider through the
  proxy against a provider stub that bills prefix-cache reads and writes
  (64 requests, all priced): the stub billed $0.253 less on the Anthropic
  session and $0.507 less on the OpenAI one; the ledger reported $0.159 and
  $0.396. The headline was $0.554 against $0.760 measured; the list-price
  model for the same traffic was $1.62. One workload, one stub, published
  prices: evidence that the estimate lands under the measured saving there,
  not a bound for every workload.
- **Net-saving governor** (`net_saving_governor`) uses the same estimate to
  restrain Entroly. Per conversation it adds up the conservative net saving of
  sealed receipts; once a conversation has at least 3 priced requests, losses
  on at least 2 of them, and is below -$0.02 in total, the proxy forwards the client's own request instead
  of Entroly's optional rewrites for the rest of that conversation. Emergency
  session rescue and required redaction still apply. Each response carries
  `X-Entroly-Net-Saving-Governor: hold|observe`; the receipt records the
  decision applied (`applied`) and the standing after it (priced and losing
  requests, cumulative microdollars, floor), and a held request carries a
  `net_saving_governor_hold` value row (`protected`, never headline).
  Replaying a conversation's receipts in order reproduces every hold.
  Requests without a priced estimate do not count, the hold never crosses
  conversations, and state is a bounded table of conversation-id digests and
  integers. Tune with `ENTROLY_NET_SAVING_MIN_REQUESTS` and
  `ENTROLY_NET_SAVING_FLOOR_USD`; `ENTROLY_NET_SAVING_GOVERNOR=0` turns it off.
  One expensive request never holds a conversation on its own: a history
  rewrite is charged in full on its turn and earns its saving back later. The
  governor covers Entroly's changes to the prompt; opt-in model routing
  (RAVS) is governed by its own confidence gate.
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
