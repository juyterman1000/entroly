# Live tokenomics and the public-counter contract

Entroly measures value locally without turning a benchmark or package download
into a savings claim. The canonical command is:

```bash
entroly value
entroly value --json
```

The JSON form is an `entroly.value-receipt.v1` document with four evidence
classes:

| Evidence class | Included | Excluded |
|---|---|---|
| `provider_path` | Requests observed by the Entroly proxy, input tokens reduced, and modeled input cost for explicit price-catalog matches | Provider invoices, output-token savings, negotiated prices and requests that bypassed Entroly |
| `local_operations` | Token reduction measured by SDK, MCP, npm and other local operations, plus `modeled_value_at_list_usd`: those tokens priced at the default catalog input rate | Invoice verification; Entroly cannot prove the result reached a paid provider, so this figure is replacement cost and is never added to `provider_path` |
| `legacy_unclassified` | Historical reduction retained for continuity | Public savings totals and current provider-path claims |
| `trust_signals` | Modeled routing value and locally observed verification interventions | Claims that every intervention prevented a real-world failure |

`entroly dashboard` reads the same shared ledger and refreshes the local value
panel automatically. The ledger stores operational counts and aggregate token
measurements, not prompts or source content.

The dashboard also exposes a **Banked Future Value** calculator for
`local_operations.tokens_reduced`. It multiplies locally reduced tokens by a
user-selected USD-per-million-input-tokens rate (default `$1.00/M`). The rate
is stored only in the browser's local storage. This value is a forward-looking
scenario for context that may later replace provider input; it is not added to
provider-bound savings, the realized cost total, or the public community
counter. The machine-readable snapshot labels it
`modeled_future_value_not_realized_savings`.

The proxy also publishes a reconciled all-layer counter and two components:

| Surface | Aggregate | Compression component | Tool-schema component |
|---|---|---|---|
| OTEL | `entroly.proxy.tokens.saved` | `entroly.proxy.tokens.compression_saved` | `entroly.proxy.tokens.tool_schema_saved` |
| Prometheus | `entroly_proxy_tokens_saved_total` | `entroly_proxy_compression_tokens_saved_total` | `entroly_proxy_tool_schema_tokens_saved_total` |

The aggregate is the canonical before/after request delta. Components partition
that total and are not added to it again. Tool-schema deferral is measured only
when the caller explicitly supplies `X-Entroly-Active-Tools`; Entroly does not
guess which tools are safe to hide. Forced choices and unnamed provider tools
are retained, and a non-matching allowlist leaves the request unchanged.

## Community counter: conservative, opt-in, and cumulative

The optional product-health collector accepts a separate savings-contribution
event for provider-bound proxy reductions. Before the event leaves a consenting
machine, tokens are rounded down to whole 1,000-token units and modeled input
cost is rounded down to whole cents. Sub-unit remainders stay local. The event
contains no model, price, prompt, code, path, request body, or exact per-request
value.

The unauthenticated `GET /v1/public-savings` endpoint can therefore report:

- a cumulative lower bound of opted-in, provider-bound tokens saved;
- cumulative modeled input cost avoided for explicitly priced models;
- the aggregate update time and fixed measurement/privacy labels.

It is not a census, provider invoice, unique-user metric, or exact worldwide
total. It starts when the community counter is deployed and includes only
consenting proxy installations. Downloads are package fetches, not users, and
are never multiplied by a benchmark ratio to create savings.

## Public-site contract

The website may switch from its checked-in reproducible proof to the cumulative
counter only when all of these are true:

1. A public aggregate endpoint is deployed and its source is linked.
2. Participation is separately opt-in and previewable before consent.
3. The displayed number is labelled by evidence class and time window.
4. No prompt, content, path, model identifier, price, IP address or exact
   per-installation workload is exposed.
5. Withdrawal and retention behavior are documented and tested.
6. The README renderer fails closed to `unavailable`; it never estimates from
   downloads, stars, benchmark ratios or missing observations.

The site polls the configured endpoint every 60 seconds and fails closed to the
checked-in proof when the endpoint is absent, malformed, or unavailable. A
user's exact trustworthy total remains the local Context Value Receipt. See
[Privacy-Safe Product Health Telemetry](telemetry-privacy.md) for the complete
event allowlist, quantization, retention, anonymized archival, and deletion
boundaries.

The production deployment target is the source-controlled
[Cloudflare Workers + D1 collector](../deploy/cloudflare-community-savings/README.md).
It keeps Worker observability disabled, stores no request headers or IP
addresses in application tables, rate-limits uploads, and exposes the same
identifier-free public response as the local Python collector. The public site
must keep using its checked-in proof until that Worker is deployed, initialized,
and verified from the GitHub Pages origin.
