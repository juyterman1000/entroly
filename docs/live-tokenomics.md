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
| `local_operations` | Token reduction measured by SDK, MCP, npm and other local operations | Dollar savings; Entroly cannot prove the result reached a paid provider |
| `legacy_unclassified` | Historical reduction retained for continuity | Public savings totals and current provider-path claims |
| `trust_signals` | Modeled routing value and locally observed verification interventions | Claims that every intervention prevented a real-world failure |

`entroly dashboard` reads the same shared ledger and refreshes the local value
panel automatically. The ledger stores operational counts and aggregate token
measurements, not prompts or source content.

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

## Why the README does not show a worldwide total yet

The optional product-health collector intentionally accepts only coarse token
reduction buckets and whether a provider-bound observation had a positive
modeled-cost signal. It rejects exact token counts, exact dollar amounts, model
identifiers, prompts, code, paths and exception details.

Consequently, the collector can honestly report:

- opted-in monthly pseudonyms that observed a positive reduction;
- provider-bound positive-reduction observations;
- coarse reduction and token buckets;
- observations with a positive modeled-cost signal;
- platform-family and bounded reliability aggregates.

It cannot honestly report an exact all-time worldwide token or dollar total.
Downloads are package fetches, not users, and multiplying them by a benchmark
ratio would fabricate adoption and savings.

## Requirements for a future public counter

A public README counter may be enabled only when all of these are true:

1. A public aggregate endpoint is deployed and its source is linked.
2. Participation is separately opt-in and previewable before consent.
3. The displayed number is labelled by evidence class and time window.
4. No prompt, content, path, model identifier, price, IP address or exact
   per-installation workload is exposed.
5. Withdrawal and retention behavior are documented and tested.
6. The README renderer fails closed to `unavailable`; it never estimates from
   downloads, stars, benchmark ratios or missing observations.

Until that contract is met, the live trustworthy counter is the user's local
Context Value Receipt. See [Privacy-Safe Product Health Telemetry](telemetry-privacy.md)
for the complete event allowlist and deletion behavior.
