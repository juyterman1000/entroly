# Entroly for Teams — the business case

**Reduce AI coding input-token spend locally, with reproducible accuracy-retention benchmarks.**

A one-page brief for whoever owns the budget, the architecture, and the risk. Entroly can be evaluated with a local proxy or a standalone Rust binary; this page summarizes the spend, security posture, and deployment story.

---

## The problem, in business terms

Coding agents (Claude Code, Cursor, Codex, Copilot…) re-send large chunks of the repo on every turn. Token spend scales with **usage × context size**, and agentic usage is growing far faster than per-token prices are falling — so the **total** bill keeps climbing. Most of those tokens are duplicated boilerplate and low-signal context the model never needed.

Entroly sits between your agents and the LLM API and selects context under a token budget. Core indexing and selection run locally; committed benchmark artifacts show the measured accuracy-retention results.

## What you get

| Outcome | What it means |
|---|---|
| **Measurable input reduction** | Entroly reports source and selected tokens under an explicit budget; translate that result using your provider pricing. |
| **Measured accuracy retention** | Compression is selection-based and benchmarked for accuracy *retention*, not just ratio — with committed result files and confidence intervals. |
| **Local & private core** | Indexing, selection, and deterministic verification paths run **on-device** — no code or context is sent anywhere for *analysis*. |
| **Hallucination guard** | WITNESS scores model output against supplied evidence without an extra provider call; local compute still applies. |
| **Multiple integration paths** | HTTP proxy, MCP server, Python SDK, or a standalone Rust binary with no Python runtime. Integration helpers cover 30+ wrap targets; compatibility depends on the installed tool and version. |

## ROI — how to size it (no vanity numbers)

Savings are workload-dependent, so here's the honest formula instead of a made-up figure:

```
monthly_savings ≈ monthly_input_token_spend × input_token_reduction
                 + model_routing_savings (RAVS, optional)
                 + provider prefix-cache discount captured (cache alignment)
```

- **Input-token reduction is workload-specific.** Run the local verification smoke and a representative proxy pilot; do not treat repository examples as a billing forecast.
- **Validate the install locally**, on your own code, with no API key:
  ```bash
  pip install entroly && cd /your/repo && entroly verify-claims
  ```
  This bounded install smoke test writes `.entroly_verification.json` with sampled token savings, indexing speed, and file coverage. Use a representative proxy pilot and the dashboard to size ROI for your actual workload.

> Pricing for "$ saved" uses a local, output-aware, **overridable** price table — set your negotiated rates via `ENTROLY_PRICING_FILE`. The dashboard shows cumulative $ saved and a per-lever Cost Intelligence breakdown.

## Security & privacy (the procurement gate)

- **Local-first.** Indexing, ranking, selection, dedup, and deterministic verification run on the developer's machine. No embeddings API, no cloud service required for the core.
- **Your code is not sent anywhere for analysis.** *(When you proxy a cloud LLM, the **compressed** prompt still goes to that provider — exactly as it would without Entroly. Entroly reduces what's sent; it doesn't add a new destination.)*
- **No outbound analytics by default.** Local usage metrics are stored for the dashboard. Optional federation and cloud-backed features must be enabled separately.
- **Auditable local core.** Core selection and deterministic verification paths are testable. Evaluate stateful learning, exploration, routing, and optional cloud-backed modes separately.
- **Apache-2.0** licensed. **Air-gap-capable** when used with local/offline endpoints.
- The Rust binary does not require a Python runtime. Validate operating-system libraries, architecture, and artifact integrity for your deployment target.

## Deployment options

| Mode | Use it when |
|---|---|
| **HTTP proxy** | Point a supported agent or provider base URL at `localhost:9377`. |
| **MCP server** | IDE/agent integration (Cursor, Claude Code, Windsurf, …). |
| **Python SDK** | `from entroly import compress` inside your own pipeline. |
| **Single Rust binary** (`entroly-rs`) | No Python runtime; frictionless / locked-down hosts. |

## Why it clears the bar

- **Spend** — use the emitted source/selected-token counts and your negotiated provider rates. Run a representative pilot before making an ROI claim.
- **Risk** — local-first, provider-neutral, Apache-2.0, with no outbound analytics by default. Review optional modes against your requirements.
- **Adoption** — proxy, MCP, SDK, and standalone Rust-binary paths are available. Integration helpers cover 30+ wrap targets; confirm the path for your tool and version.

## Evaluate it

1. `pip install entroly && cd /your/repo && entroly verify-claims` — validate the local install and inspect the sampled smoke report.
2. Reproduce the public benchmarks and inspect their limits in the [public evidence ledger](public-evidence.md).
3. Run a representative proxy pilot; use the dashboard's Cost Intelligence panel to size workload-specific savings.

---

Full technical detail → [README](../README.md) · [Architecture](DETAILS.md). Apache-2.0.
