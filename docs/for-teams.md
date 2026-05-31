# Entroly for Engineering Leaders

**Cut your team's AI coding bill 70%+ — locally, with proof that accuracy didn't drop.**

A one-page brief for the people who own the budget, the architecture, and the risk. Devs adopt Entroly bottom-up (a 30-second proxy or one Rust binary); this page is the part you take to finance, security, and your VP.

---

## The problem, in business terms

Coding agents (Claude Code, Cursor, Codex, Copilot…) re-send large chunks of the repo on every turn. Token spend scales with **usage × context size**, and agentic usage is growing far faster than per-token prices are falling — so the **total** bill keeps climbing. Most of those tokens are duplicated boilerplate and low-signal context the model never needed.

Entroly sits between your agents and the LLM API and sends only the context that matters, under a token budget — **locally, deterministically, with the accuracy held.**

## What you get

| Outcome | What it means |
|---|---|
| **Lower bill** | 70–95% fewer input tokens on large-repo workloads (measured; workload-dependent). |
| **No accuracy tax** | Compression is selection-based and benchmarked for accuracy *retention*, not just ratio — committed, reproducible result files, not screenshots. |
| **Local & private** | Indexing, selection, and verification run **on-device** — no code or context is sent anywhere for *analysis*. |
| **Hallucination guard** | WITNESS checks model output against supplied evidence at $0 / no extra LLM call (Python product). |
| **Drop-in** | HTTP proxy, MCP server, Python SDK, or a zero-dependency Rust binary. Works with 30+ agents. |

## ROI — how to size it (no vanity numbers)

Savings are workload-dependent, so here's the honest formula instead of a made-up figure:

```
monthly_savings ≈ monthly_input_token_spend × input_token_reduction
                 + model_routing_savings (RAVS, optional)
                 + provider prefix-cache discount captured (cache alignment)
```

- **Input-token reduction** measured 70–95% on large repos (e.g. **97.9% on the DeepSpeed repo at an 8K budget**). Short prompts / tiny repos save less.
- **Measure *your* number in 60 seconds**, on your own code, no API key:
  ```bash
  pip install entroly && cd /your/repo && entroly verify-claims
  ```
  It writes a machine-readable `.entroly_verification.json` (token savings, indexing speed, file coverage) you can hand to finance.

> Pricing for "$ saved" uses a local, output-aware, **overridable** price table — set your negotiated rates via `ENTROLY_PRICING_FILE`. The dashboard shows cumulative $ saved and a per-lever Cost Intelligence breakdown.

## Security & privacy (the procurement gate)

- **Local-first.** Indexing, ranking, selection, dedup, and deterministic verification run on the developer's machine. No embeddings API, no cloud service required for the core.
- **Your code is not sent anywhere for analysis.** *(When you proxy a cloud LLM, the **compressed** prompt still goes to that provider — exactly as it would without Entroly. Entroly reduces what's sent; it doesn't add a new destination.)*
- **No telemetry by default** — usage stats are opt-in and disabled unless you enable them.
- **Deterministic** — same input → bit-identical output (auditable, testable).
- **Apache-2.0** licensed. **Air-gap-capable** when used with local/offline endpoints.
- The Rust single binary is **zero-dependency** (no Python runtime) for locked-down environments.

## Deployment options

| Mode | Use it when |
|---|---|
| **HTTP proxy** | Zero code change — point `*_BASE_URL` at `localhost:9377`. Any language/agent. |
| **MCP server** | IDE/agent integration (Cursor, Claude Code, Windsurf, …). |
| **Python SDK** | `from entroly import compress` inside your own pipeline. |
| **Single Rust binary** (`entroly-rs`) | No Python runtime; frictionless / locked-down hosts. |

## Per-role summary

- **CIO** — recurring AI spend down, measurable and exportable; local-first + Apache-2.0 + no telemetry clears the privacy/procurement bar.
- **CTO** — provider-neutral layer (no single-vendor lock-in), deterministic, local compute; complements your stack rather than replacing it.
- **VP Eng** — drop-in (30-sec proxy / one command), 30+ agent integrations, no workflow change; impact is visible on the dashboard from day one.
- **Devs** — install once, agents get a broader project map in a smaller context; nothing to re-learn.

## Evaluate it

1. `pip install entroly && cd /your/repo && entroly verify-claims` — your real numbers, locally.
2. Reproduce the public accuracy-retention benchmarks (committed JSON): see the [Benchmarks](../README.md#benchmarks) section.
3. Roll out via the proxy for one team; watch the dashboard's Cost Intelligence panel.

---

Full technical detail → [README](../README.md) · [Architecture](DETAILS.md). Apache-2.0.
