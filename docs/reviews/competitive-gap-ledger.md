# Eight-pillar competitive gap ledger

**Status of this document.** Entroly columns cite measured evidence in this
repository. Competitor columns record what Headroom and LeanCTX *document about
themselves*. Neither competitor has been installed or executed here, so no
competitor row is verified behaviour and no comparative claim in this file is a
benchmark result.

Nothing in this ledger supports a statement that Entroly leads, ties or trails.
That requires executing both tools on shared workloads, which has not been done.

## Pinned systems

| system | source | version | date | license | install | pinned? |
|---|---|---|---|---|---|---|
| Entroly | this repo | 1.0.73 | 2026-08-03 | see repo | `pip install entroly` | yes, SHA `0f3d131` |
| Headroom | [headroomlabs-ai/headroom](https://github.com/headroomlabs-ai/headroom) | **not stated on the repo landing page** | — | Apache-2.0 | `pip install "headroom-ai[all]"` | **NO — version unpinned** |
| LeanCTX | [yvgude/lean-ctx](https://github.com/yvgude/lean-ctx) | v3.9.13 | 2026-07-29 | — | Rust binary | tag pinned, SHA not resolved |

§4.1 requires a commit SHA per competitor before final benchmarking. Neither is
resolved. **Benchmarking against either is blocked until they are.**

## Evidence key

```
M  measured in this repo, command committed
T  tested (unit/integration), no product path
D  documented by the vendor, NOT executed here
-  absent / not found
?  unknown
```

## Pillar 1 — Compression intelligence and content breadth

| dimension | Entroly | Headroom | LeanCTX | evidence / gap |
|---|---|---|---|---|
| JSON / structured | T — codec preserves ids, codes, amounts, timestamps; recovery == original byte stream | D — "SmartCrusher", 60–95% fewer tokens on JSON | ? | Entroly measured on 3 fixtures only. Competitor ratio unverified. |
| Logs | T — safe templating; critical numerics never merged | D | D — shell patterns | Entroly gap: no streaming operation. |
| Shell / tool output | T — entropy+role scoring, command-agnostic | D | D — "60+ shell patterns" | LeanCTX claims per-tool breadth Entroly does not have. |
| Code | M — **AST skeleton for Python** (stdlib `ast`, exact node ranges); line-heuristic fallback elsewhere | D — AST for Python, JS/TS, Go, Rust, Java, C/C++, Perl | D — AST, 10 read modes | **Gap narrowed, not closed.** Entroly parses Python only; both competitors document many languages. |
| RAG / documents | T — query-conditioned spans + neighbours + citations | D | ? | untested against a public RAG set |
| Conversation / memory | T — instructions/decisions kept, prefix byte-identical | D | D — sessions/memory | |
| Schema / MCP descriptions | T — contract preserved, prose dropped | ? | D — 62–76 MCP tools | |
| Tabular / CSV | M — header verbatim, per-column type/quantiles/missingness, exact recovery; 3,915→111 tokens | D | ? | Added after the first ledger revision. |
| Images / multimodal | **-** | D — "40–90% reduction via ML router" | ? | **Gap: Entroly has no multimodal path.** |
| Learned/model-based compression | **-** | D — "Kompress-v2-base", HF model on agentic traces | ? | **Gap: Entroly is entirely training-free.** Not necessarily a deficiency; it is a different trade (no model download, no inference cost). |

## Pillar 2 — Code, repository and agent intelligence

| dimension | Entroly | Headroom | LeanCTX | evidence / gap |
|---|---|---|---|---|
| Repo indexing | M — 215 modules / 524 edges mapped by `scripts/codebase_graph.py` | ? | D — local Rust binary, repo graphs | |
| AST / tree-sitter | **-** | D | D | **Gap** |
| Symbol / call / import graph | partial — `depgraph.rs` | ? | D | |
| Task-conditioned policies | **-** | ? | D — 10 read modes | **Gap** |
| Test localization | **-** | ? | D | **Gap** |

## Pillar 3 — Proxy, providers, integrations

| dimension | Entroly | Headroom | LeanCTX | evidence / gap |
|---|---|---|---|---|
| Proxy | M — 268 proxy tests pass (transport trust, cache alignment, access security, providers, session rescue, control plane). Live-provider suite excluded: no credentials | D — `headroom proxy` | ? | Corrects an earlier row in this file that said "not tested"; the suite existed and passes. **Gap: streaming cancellation, backpressure and cache-aware NET accounting are still unmeasured.** |
| MCP | present | D — 3 tools | D — 62–76 tools | |
| Framework adapters | ? | D — Anthropic, OpenAI, Vercel AI, LiteLLM, LangChain, Agno, Strands, ASGI | ? | **Gap: Headroom documents 8 framework adapters.** |
| Agent integrations | OpenClaw + others advertised | D — 15 named agents | D — 30+ agents | **Gap, and Entroly's advertised list is not contract-tested.** |
| Provider conformance | **not tested** | D | ? | **Blocked: no provider credentials available here.** |

## Pillar 4 — Assurance, recovery, security, privacy

| dimension | Entroly | Headroom | LeanCTX | evidence / gap |
|---|---|---|---|---|
| Exact recovery | **M — `recover(ref) == original byte stream`, digest AND length verified, and reachable from a terminal via `entroly compress` / `entroly recover` across a process boundary** | ? | D — "proves what they save" | Entroly's strongest verified position. |
| Content-addressed verification | M — forged digest and forged byte_length both rejected | ? | ? | |
| Receipts / provenance | M — byte offsets + source/fragment SHA-256 | ? | D — "context proofs" | |
| Fail-closed sufficiency | T — named `CalibrationPolicy`; uncalibrated never claims sufficient | ? | ? | No competitor documents an abstention contract. |
| Threat model | partial — `tests/test_codec_abuse.py` covers forged receipts, corrupt stores, compression bombs, malformed input, cross-scope isolation (30 tests). Found and fixed a DoS and a crash | ? | ? | **Gap: prompt injection, path traversal, proxy auth and concurrent-writer durability are NOT covered.** |

## Pillar 5 — Performance, caching, scalability

| dimension | Entroly | Headroom | LeanCTX | evidence / gap |
|---|---|---|---|---|
| Per-stage latency | M — `benchmarks/stage_latency.py`, p50/p95 for digest/route/represent/prune/store/recover at tiny/10KB/100KB/1MB | ? | ? | Found routing a 1MB payload cost more than compressing it; fixed. Log codec at 665 ms/MB is measured and unaddressed. |
| Cache-aware accounting | **-** | D — cache mode | ? | **Gap: Entroly reports gross, not net-of-cache savings.** |
| Scale workloads | **-** | ? | D — monorepo | **Gap: no 1 MB / monorepo / concurrency runs.** |

## Pillar 6 — Cross-surface and packaging parity

| dimension | Entroly | Headroom | LeanCTX | evidence / gap |
|---|---|---|---|---|
| Shared engine across surfaces | M — Python and WASM compile the same `entroly-qccr` crate | ? | single Rust binary | |
| Golden parity fixture | T — `tests/fixtures/qccr_parity.json`, **Python side only** | ? | ? | **Gap: npm side not wired to the fixture.** |
| Pure-Python fallback | M — CI job green | n/a | n/a | Entroly ships a fallback; both competitors are native-only by design. |
| Release surfaces | M — 15 surfaces synchronised at 1.0.73 | ? | ? | |

## Pillar 7 — Operations and developer experience

| dimension | Entroly | Headroom | LeanCTX | evidence / gap |
|---|---|---|---|---|
| One-command install | M | D | D | |
| `doctor` | present | D | D | |
| First-run value | mixed — `entroly compress` -> receipt -> `entroly recover` is a complete journey with 9 tests across a process boundary; but pure-Python still reports 0% on a small project | ? | D — "zero config" | **Known gap, recorded in `test_simulate_small_project.py`.** |
| Observability | partial | D — stats MCP tool | ? | **Gap: no Prometheus/OTel.** |
| Enterprise controls | **-** | ? | ? | **Gap: no SBOM, signing, RBAC, air-gapped mode.** |

## Pillar 8 — Benchmark credibility

| dimension | Entroly | Headroom | LeanCTX | evidence / gap |
|---|---|---|---|---|
| Committed harness | M — `benchmarks/sufficiency_baseline.py`, `benchmarks/codec_ablation.py` | ? | ? | |
| Model-in-the-loop eval | **-** | ? | ? | **Gap: every Entroly number here is substring retention, not answer accuracy.** |
| Held-out validation | **-** | ? | ? | **Gap: sufficiency thresholds are in-sample on 3 fixtures.** |
| Competitor comparison | **-** | — | — | **Blocked: neither competitor installed or executed.** |
| Confidence intervals | **-** | ? | ? | **Gap.** |

## The honest summary

Entroly's verified strength is the **assurance contract** — exact byte-stream
recovery with digest-and-length verification, provenance, and a fail-closed
sufficiency verdict that refuses to claim "sufficient" without a named
calibration policy. No competitor documents an equivalent.

Entroly's clearest documented deficits are **multi-language code understanding**
(Python is now parsed; every other language still uses line heuristics, while both
competitors document many), **multimodal** and
**learned compression** (Headroom documents both, Entroly has neither),
**framework/agent adapter breadth**, and **cache-aware net accounting**.

Every Entroly capability marked `T` is a tested contract with **no production
caller**. Under §18 those score at most 2 ("works on limited fixtures"), not 3
("production-reachable and tested").

## Blockers requiring the user

1. **Competitor SHAs** — Headroom's version is not stated on its landing page.
2. **Provider credentials** — no API keys available here, so provider
   conformance and model-in-the-loop benchmarks cannot run.
3. **Executing competitors** — installing and running Headroom and LeanCTX on
   shared workloads is required before any comparative claim.
