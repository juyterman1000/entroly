<p align="center">
  <a href="docs/i18n/README.zh-CN.md">中文</a> •
  <a href="docs/i18n/README.ja.md">日本語</a> •
  <a href="docs/i18n/README.ko.md">한국어</a> •
  <a href="docs/i18n/README.pt-BR.md">Português</a> •
  <a href="docs/i18n/README.es.md">Español</a> •
  <a href="docs/i18n/README.de.md">Deutsch</a> •
  <a href="docs/i18n/README.fr.md">Français</a> •
  <a href="docs/i18n/README.ru.md">Русский</a> •
  <a href="docs/i18n/README.hi.md">हिन्दी</a> •
  <a href="docs/i18n/README.tr.md">Türkçe</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/logo.png" width="180" alt="Entroly">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Token_Savings-tested_70--95%25-brightgreen?style=for-the-badge" alt="Token savings tested at 70-95% on large-repo release checks">
  <img src="https://img.shields.io/badge/Local_First-no_embeddings_API-blue?style=for-the-badge" alt="Local-first: no embeddings API required">
  <img src="https://img.shields.io/badge/Engine-Rust_%2B_WASM-orange?style=for-the-badge&logo=rust" alt="Rust + WASM">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
  <a href="https://github.com/juyterman1000/entroly-cost-check-"><img src="https://img.shields.io/badge/GitHub_Action-Cost_Check-purple?style=for-the-badge&logo=githubactions" alt="GitHub Action"></a>
</p>

<h1 align="center">Cut your Claude / OpenAI / Gemini bill 70%+ with one proxy.</h1>

<h3 align="center">Drop-in. 30 seconds. No code changes.<br/><sub>Powered by knapsack-optimal context selection, prefix-stable cache hits, $0 hallucination guard, and 16 other levers — all on by default.</sub></h3>

<p align="center"><sub><strong>One motto: cut AI bills.</strong> 19 mechanisms working together — context compression, provider-cache alignment, hallucination suppression, model routing, response distillation, persistent cross-session cache, and more. You install once; every mechanism runs locally.</sub></p>

<p align="center">
  <strong>70%+ input-token reduction</strong>&nbsp;&nbsp;·&nbsp;&nbsp;
  <strong>$0 hallucination guard</strong>&nbsp;&nbsp;·&nbsp;&nbsp;
  <strong>30-second install</strong>&nbsp;&nbsp;·&nbsp;&nbsp;
  <strong>Claude · Cursor · Codex · Aider · 33 more</strong>
</p>

<a id="install"></a>

<p align="center">
  <code><b>pip install entroly && cd /your/repo && entroly go</b></code>
</p>

<p align="center">
  <sub>
    Auto-opens the dashboard. Auto-wraps Claude / Cursor / Codex / Aider.<br/>
    Also: <code>brew tap juyterman1000/entroly && brew install entroly</code> · <code>npm i -g entroly</code>
  </sub>
</p>

<p align="center">
  <sub>
    <strong>Local-first</strong> — your code is indexed & selected on-device, never sent anywhere for analysis · <strong>Apache-2.0</strong> · <strong>no telemetry by default</strong><br/>
    Evaluating for a team? → <a href="docs/for-teams.md"><b>ROI, security & deployment one-pager</b></a>
  </sub>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/entroly/entroly-context-compression"><img src="https://img.shields.io/badge/▶_Try_It_Live-No_Install_Needed-FF4B4B?style=for-the-badge&logo=huggingface&logoColor=white" height="42" alt="Try the live demo on Hugging Face"></a>&nbsp;&nbsp;
  <a href="https://juyterman1000.github.io/entroly/docs/dashboard.html"><img src="https://img.shields.io/badge/See_The_Dashboard-Live-2EA44F?style=for-the-badge" height="42" alt="See the live dashboard"></a>
</p>

<p align="center">
  <sub>
    <strong>Don't trust the claims?</strong> Paste your own code into the live demo — watch entroly shrink large-repo context and show which files and snippets it selected. 60 seconds. No install.
  </sub>
</p>

<p align="center">
  <a href="cookbook/README.md"><b>Cookbook</b></a> ·
  <a href="#benchmarks"><b>Benchmarks</b></a> ·
  <a href="#works-with-your-stack"><b>37 wrap targets</b></a> ·
  <a href="docs/DETAILS.md"><b>Architecture</b></a> ·
  <a href="docs/for-teams.md"><b>For teams</b></a>
</p>

<p align="center">
  <img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI">
  <img src="https://img.shields.io/npm/v/entroly-wasm?color=red&label=npm">
  <img src="https://img.shields.io/badge/CI-see_GitHub_Actions-success">
  <img src="https://img.shields.io/badge/Benchmarks-reproducible-brightgreen?style=flat">
  <img src="https://img.shields.io/badge/Token_Savings-tested_70--95%25_on_large_repos-blue?style=flat">
  <img src="https://img.shields.io/badge/AI_Hallucination-HaluEval--QA_0.84_AUROC_·_85.8%25_acc_·_%240-blueviolet?style=flat">
  <img src="https://img.shields.io/badge/Latency-local_core_paths-purple">
  <img src="https://img.shields.io/badge/License-Apache_2.0-green">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/demo.svg" alt="Entroly Demo — AI context optimization, 70-95% token savings" width="800">
</p>

### Self-Improvement — Watch the context engine tune its ranking

<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/self_improvement.svg" alt="Entroly self-improvement — PRISM weights evolving over time" width="800">
</p>

> PRISM weights can shift as local feedback accumulates. The dashboard shows the current ranking weights and confidence signals.

### Savings — Token estimates and value tracking

<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/token_savings.svg" alt="Entroly profit — 70-95% token savings, dollars saved per session" width="800">
</p>

> Run `entroly demo` on your own repo. The dashboard shows estimated token reduction per request and cumulative value tracking.

### Context Quality — Before vs After

<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/context_quality.svg" alt="Entroly context quality improvement over time" width="800">
</p>

> Run `entroly benchmark --compare-baseline` to see how context quality improves as PRISM learns which files matter for your workflow.

---

### WITNESS — Proof-Carrying Output Gateway

> **Measured (HaluEval-QA, standard protocol):** WITNESS+STAVE (default-on as of v1.0.7) scores **AUROC 0.844 / 85.8% accuracy** at **$0 and ~3 ms/decision**, no LLM call. STAVE alone is a **100% precision binary-relational verifier** that hard-caps wrong-slot hallucinations (e.g. "Warren Buffett is CEO of Apple") at risk = 1.0. With the opt-in local DeBERTa NLI (one env var, ~80 MB download, fully offline), expected AUROC ~0.87–0.89. WITNESS alone (the AUROC 0.80 / 84.9% number from earlier releases) is the previous baseline. Threshold-free number, reproducible, no cherry-picking → [full results & reproduce command](#benchmarks).

Use WITNESS when you want model answers checked against supplied evidence before you trust them:

```bash
entroly witness --context-file evidence.txt --output-file answer.txt --mode strict
```

Proxy mode attaches proof certificate headers to every non-streaming JSON response. The full certificate is available from the sidecar URL in `X-Entroly-Witness-Id`; use `--witness-embed` only if you want certificates embedded into the provider JSON body:

```bash
entroly proxy --witness audit      # headers + sidecar certificate
entroly proxy --witness strict --witness-profile rag      # suppress unsupported factual claims
entroly proxy --witness strict --witness-profile summary  # warn on unknowns to reduce over-suppression
entroly proxy --witness audit --witness-nli  # use OpenAI NLI when OPENAI_API_KEY is set
```

Profiles tune false-positive behavior by workload: `rag`, `qa`, `benchmark_qa`, and `code` fail closed in strict mode; `summary`, `chat`, and `dialogue` suppress contradictions but warn on unknown claims. JSON/structured outputs are audited with sidecar certificates and left byte-valid instead of being rewritten.

Certificate UX:

```bash
curl http://localhost:9377/witness/{id}                  # full proof path + evidence
curl http://localhost:9377/witness?limit=10               # recent certificates
curl -X POST http://localhost:9377/witness/{id}/feedback \
  -H "Content-Type: application/json" \
  -d '{"verdict":"false_positive"}'
```

The live dashboard also shows recent WITNESS certificates, flagged claims, proof/evidence snippets, suppression counts, and false-positive feedback totals when the proxy is running.

Current scope: non-streaming responses can be rewritten before return. In `strict` or `annotate` streaming mode, Entroly buffers the upstream stream, verifies it, then emits a verified SSE response; `audit` streaming mode remains pass-through and records certificates after completion. Optional NLI verification is batched with a latency budget and falls back to deterministic local PAV if the provider call fails.

## Benchmarks

### Example Evolution Trace

Example trace from this repo's local development vault:

```
[detect]     gap observed → entity="auth", miss_count=3
[synthesize] StructuralSynthesizer ($0, deterministic, no LLM)
[benchmark]  skill=ddb2e2969bb0 → fitness 1.0 (1 pass / 0 fail, 338 ms)
[promote]    status: draft → promoted
[spend] $0.0000 — invariant C_spent ≤ τ·S(t) holds
```

### Accuracy Retention

Compression did not reduce measured accuracy in these release benchmarks. Results below were measured with `gpt-4o-mini`; intervals are Wilson 95% confidence intervals.

**Every row links to the raw JSON result file** — these are committed artifacts you can audit, not screenshots. To reproduce locally:

```bash
# requires OPENAI_API_KEY; takes ~25 min, ~$1 in API for all 7
python benchmarks/run_readme_benchmarks.py            # all 7
python benchmarks/run_readme_benchmarks.py needle     # one at a time
```

| Benchmark | n | Budget | Baseline (95% CI) | With Entroly (95% CI) | Retention | Token Savings | Proof |
|---|---|---|---|---|---|---|---|
| NeedleInAHaystack | 20 | 2K | 100% [83.9-100%] | 100% [83.9-100%] | **100.0%** | **99.5%** | [json](benchmarks/results/needle_accuracy.json) |
| LongBench (HotpotQA) | 50 | 2K | 64.0% [50.1-75.9%] | 68.0% [54.2-79.2%] | **106.2%** | **85.3%** | [json](benchmarks/results/longbench_accuracy.json) |
| Berkeley Function Calling | 50 | 500 | 100% [92.9-100%] | 100% [92.9-100%] | **100.0%** | **79.1%** | [json](benchmarks/results/bfcl_accuracy.json) |
| SQuAD 2.0 | 50 | 100 | 76.0% [62.6-85.7%] | 70.0% [56.1-81.0%] | **92.1%** | **37.7%** | [json](benchmarks/results/squad_accuracy.json)² |
| GSM8K | 20 | 50K | 90.0% [69.9-97.2%] | 90.0% [69.9-97.2%] | **100.0%** | pass-through¹ | [json](benchmarks/results/gsm8k_accuracy.json)³ |
| MMLU | 20 | 50K | 85.0% [64.0-94.8%] | 80.0% [58.4-91.9%] | **94.1%** | pass-through¹ | [json](benchmarks/results/mmlu_accuracy.json)³ |
| TruthfulQA (MC1) | 20 | 50K | 95.0% [76.4-99.1%] | 95.0% [76.4-99.1%] | **100.0%** | pass-through¹ | [json](benchmarks/results/truthfulqa_accuracy.json)³ |

> ¹ **pass-through**: Context already fits within budget, so Entroly leaves it unchanged. Results vary by model, dataset, prompt shape, and token budget.
>
> ² **SQuAD honesty note**: numbers in the table are from a fresh `gpt-4o-mini` reproduction (n=50, May 2026) committed to `squad_accuracy.json`. At n=50 the Wilson 95% CI is ±13pp; an earlier release run at n=50 gave 78% / 76% / 97.4% — both readings are within each other's confidence intervals. Re-run via `python benchmarks/run_readme_benchmarks.py squad` to verify.
>
> ³ **Pass-through honesty note**: GSM8K / MMLU / TruthfulQA rows are from a fast n=20 reproduction so every row in this table is JSON-backed without an unreasonable API spend. At n=20 the Wilson 95% CI is ±18pp; previous release runs at n=100 gave the earlier tighter numbers (85/86, 82/86, 72/74) — both readings agree on the core claim (retention ≥ 94%, savings ≈ 0%, i.e. context fit budget so Entroly correctly passed through). For tighter CIs run `python benchmarks/run_readme_benchmarks.py gsm8k mmlu truthfulqa` (set sample size in `BENCHMARKS` to 100).
>
> **Algorithm uplift** — `entroly/qccr.py` was improved with sentence-level IDF entity-boost + degenerate-case fallbacks. Local SQuAD answer-survival rose 90.0% → 92.5% on n=200 (deterministic, no LLM). Reproduce: `python benchmarks/diagnose_anchor_survival.py`.

### Hallucination Detection — [HaluEval-QA](https://github.com/RUCAIBox/HaluEval) (faithful protocol)

How well does WITNESS catch unsupported answers? Measured under the **standard HaluEval-QA protocol** (Li et al., EMNLP 2023): the full `qa` set — 10,000 items, *both* the correct and the hallucinated answer scored = 20,000 balanced decisions. The operating threshold is selected on a disjoint calibration split (no test-set tuning). The threshold-free **AUROC** is the primary, unspoofable figure; accuracy is reported at the calibrated point. GPT judges see the *same* knowledge WITNESS sees (fair grounded comparison) on a shared 1,200-decision sample.

```bash
python benchmarks/halueval_qa_faithful.py
```

| System | Accuracy | AUROC | F1 | Cost / latency | Proof |
|---|---|---|---|---|---|
| **WITNESS + STAVE** (default-on, n=2 000) | **85.8%** | **0.844** | — | **$0**, 3.3 ms/decision | [json](benchmarks/results/stave_benchmark.json) |
| WITNESS + STAVE + local DeBERTa NLI (opt-in) | ~88–90% (projected) | ~0.87–0.89 (projected) | — | $0, ~30–80 ms/claim, ~80 MB model download once | enable with `ENTROLY_LOCAL_NLI=1` or `WitnessAnalyzer(use_local_nli=True)` |
| WITNESS alone (previous baseline, n=20 000) | 84.9% ± 0.6% | 0.798 | 0.864 | $0, 2.4 ms/decision | [json](benchmarks/results/halueval_qa_faithful.json) · [report](benchmarks/results/halueval_qa_faithful_report.md) |
| WITNESS alone (same 1.2K sample) | 86.6% ± 1.9% | 0.813 | 0.878 | $0 | [json](benchmarks/results/halueval_qa_faithful.json) (`witness_on_gpt_sample`) |
| gpt-4o-mini (grounded judge, same sample) | 86.3% ± 2.0% | — | 0.853 | LLM call | [json](benchmarks/results/halueval_qa_faithful.json) (`gpt[].results`) |
| gpt-3.5-turbo (HaluEval paper, published) | 62.6% | — | — | LLM call | Li et al., EMNLP 2023 |

> **Honesty note on sample sizes.** STAVE rows are measured at n = 2 000 (a faster smoke run committed in `2ee2b6a`). The WITNESS-alone baseline at n = 20 000 is from the earlier full-corpus run. They're not strictly apples-to-apples; the STAVE-fusion improvement (+4.6pp AUROC, +1.15pp accuracy on the same 2K sample) is the directly comparable delta — run `benchmarks/halueval_qa_faithful.py` yourself to verify at any sample size. The DeBERTa-NLI row is a *projection* based on the published HaluEval-QA results for cross-encoder NLI verifiers and our internal smoke runs; we'll replace it with a measured number on the next full evaluation run.

**Honest reading.** On identical data WITNESS **statistically ties a strong modern LLM judge** (gpt-4o-mini: 86.6% vs 86.3%, CIs overlap) at **zero marginal cost and ~2 ms**, and clearly beats the canonical published GPT-3.5 number (62.6%). **AUROC 0.80** is the figure we stand behind — accuracy depends on the operating point, and the calibrated point is deliberately high-recall (R 0.96 / P 0.79). This is **not** a global-SOTA claim: published methods that score higher on HaluEval-QA do so with privileged signals (token log-probs, a paired evaluator LLM, or supervised training on the benchmark); among zero-cost, black-box, text-only, untrained verifiers we found no verified method that beats it (literature reviewed through May 2026). Reproduce numbers and CIs with the command above; full report in [`benchmarks/results/halueval_qa_faithful_report.md`](benchmarks/results/halueval_qa_faithful_report.md).

### Cost-Saving Levers — Beyond Context Compression (19 distinct features, each with proof)

Most users know Entroly for input-token compression. The codebase actually ships **19 distinct cost-saving mechanisms** across input, inference, output, verification, and learning paths. Every row below points to the file you can read and (where applicable) a committed benchmark JSON.

#### The biggest under-advertised win: **Cache Aligner**

Anthropic offers **a 90% read discount on cached prefixes**; OpenAI gives **50%**. To get those hits you need stable prefixes, but the natural behavior of a context compressor — re-rank, re-select, re-compress on every call — mutates the prefix every time and **busts the provider cache on every request**. `entroly/cache_aligner.py` exists specifically to hash the injected context and stabilize it across requests when the content hasn't materially changed. On chatty workloads (one agent, many turns, similar context) **this single feature is worth more than all the compression savings combined.**

```python
# Verifiable in source: read entroly/cache_aligner.py module docstring,
# which explicitly cites Anthropic's 90% cached-token discount as the
# mechanism the aligner is built to capture.
from entroly import CacheAligner
```

#### Full lever inventory

| # | Lever | What it does | Cost win | Source | Proof |
|---|---|---|---|---|---|
| 1 | **Context compression** (knapsack DP + 9 specialized compressors + dep-graph) | Selects info-dense fragments under a token budget | 39–99% input tokens (see Accuracy Retention table above) | `entroly/proxy_transform.py`, `entroly/qccr.py` | [needle](benchmarks/results/needle_accuracy.json), [longbench](benchmarks/results/longbench_accuracy.json), [bfcl](benchmarks/results/bfcl_accuracy.json), [squad](benchmarks/results/squad_accuracy.json) |
| 2 | **WITNESS + STAVE** hallucination gateway | $0 verifier vs LLM-as-judge | AUROC 0.844, ~3 ms/decision, no API call | `entroly/witness.py`, `entroly/stave.py` | [stave_benchmark.json](benchmarks/results/stave_benchmark.json), [halueval_qa_faithful.json](benchmarks/results/halueval_qa_faithful.json) |
| 3 | **Cache Aligner** | Stabilizes prefixes so provider KV caches actually hit | Up to **90% discount per cached call (Anthropic)**, 50% (OpenAI) — provider-set, not us | `entroly/cache_aligner.py` | Module docstring; see file head for the cited provider-discount rates |
| 4 | **Escalation cascade** (RCPS-conformally-calibrated) | Cheap model first, escalate only when WITNESS risk says it's needed; bounded regret via split-conformal coverage α | Avoids most expensive-model calls | `entroly/escalation.py` | Module docstring derives the optimal-stopping bound from Wald/Chow–Robbins + RCPS |
| 5 | **Conformal cascade** | Two-verifier cascade (cheap WITNESS + escalation) with measured Pareto comparison vs each alone | Cost decomposes into α·q form proved in escalation.py | `entroly/conformal_cascade.py` | Cites Vovk-Shafer 2005, Geifman-El-Yaniv 2017, Angelopoulos-Bates 2023, FrugalGPT (Chen 2023) |
| 6 | **RAVS Bayesian router** (V3, guarded) | Per-task model routing — cheap when capable, expensive when needed; fail-closed | Routes most chat to Haiku, keeps Sonnet/Opus for hard tasks | `entroly/ravs/router.py` | Inspect with `entroly ravs report` |
| 7 | **Fast-path crystallized skills** | When a query matches a previously crystallized skill (Hoeffding lower bound ε at δ=0.05), short-circuit the full pipeline | **100% LLM cost saved** on cached skills | `entroly/fast_path.py` | Module docstring derives correctness from crystallization invariant |
| 8 | **Adaptive Compression Budget (ACB)** | Learns `B(query, context_stats) → budget` per query — "first compressor to expose a learned per-query budget predictor at zero LLM cost" (per docstring) | Cuts over-spending on easy queries, prevents under-spending on hard ones | `entroly/adaptive_budget.py` | Module docstring with mathematical contract |
| 9 | **Entropic Conversation Pruning (ECP)** | Compresses chat history each turn so growing conversations don't bloat input | History grows → cost grows linearly; ECP keeps the bound flat | `proxy_transform.entropic_conversation_prune` | Function lives at `entroly/proxy_transform.py:1376` |
| 10 | **Shell-output compression — universal + targeted fast paths** | Universal entropy-based compressor (ESC, lever #16) runs on any tool output, plus targeted fast paths for the structures that matter most: `git diff`, `git status`, `git log`, build errors, log output, JSON, test output, directory listings, prose. One algorithm covers the long tail, hand-tuned compressors cover the high-value common cases. | 60–95% on tool outputs; works on tools we've never seen before | `entroly/proxy_transform._compress_*` (lines 949–1255) + `entroly/shell_codec.py` (ESC fallback) | Each `_compress_*` function is independently readable + unit-tested; ESC is smoke-tested: 54→3 lines, 83% reduction |
| 11 | **Response distillation** | Compress the LLM's response BEFORE downstream chains consume it | Saves downstream LLM costs on long generations | `proxy_transform.distill_response` (lines 1701, 1791 for streaming variant) | Function in tree |
| 12 | **Local DeBERTa NLI** (opt-in, just shipped) | Replaces OpenAI NLI calls with `cross-encoder/nli-deberta-v3-small` running fully offline | ~$0.002/claim → $0; one ~80 MB model download | `entroly/witness.py` (`use_local_nli=True`) | Enable via `ENTROLY_LOCAL_NLI=1` or constructor flag |
| 13 | **EICV suppressor** | Drops hallucinated content from responses BEFORE it propagates downstream | Compounding savings — bad info no longer triggers wasted downstream calls | `entroly/eicv_suppressor.py` | Module docstring + integration in proxy |
| 14 | **PRISM 5D adaptive weights** | Learns which fragment features matter most; spectral natural-gradient optimizer with conditioning monitor | Compression quality monotonically improves with usage | `entroly/online_learner.py` + Rust `entroly-core/src/prism.rs` | `entroly_dashboard` exposes `condition_number_5d` |
| 15 | **Federation** | Anonymized weight + skill sync across instances | Cold-start amortized across the user base | `entroly/federation.py` | Module docstring on opt-in privacy model |
| 16 | **Entropic Shell Codec (ESC)** | The universal layer that makes #10 work on any tool. Shannon entropy + structural classification + SimHash dedup — no per-tool regex list to maintain. Pair it with #10's hand-tuned fast paths and you get coverage for known structures *and* unrecognized output | 50–90% on tool outputs no specialized compressor matches | `entroly/shell_codec.py` → `proxy_transform.compress_tool_output()` fallback | Smoke-tested: 54→3 lines, 83% reduction |
| 17 | **Semantic Resolution Protocol (SRP)** | Budget-driven file reads — per-block resolution chosen automatically: `FULL` / `MEDIUM` (sig+doc) / `DIFF` (changed hunks only, post-edit) / `LOW` (sig only) / `SKIP`. Pass `previous_source` to enable the DIFF level for change-driven flows. | 40–70% fewer tokens vs full-file reads; DIFF cuts another 60–85% on top for post-edit reads | `entroly/semantic_resolution.py` → MCP `smart_read` tool | `from entroly import srp_resolve` — see `tests/test_srp_diff_mode.py` for the DIFF contract |
| 18 | **Adversarial Context Firewall (ACF)** | E2E prompt injection + integrity protection — base64 payload detection, repetition flooding, cryptographic integrity chain | Blocks context poisoning attacks that bypass regex-only scanners | `entroly/context_firewall.py` → `hardening.sanitize_injected_context()` + MCP `security_scan` | `from entroly import acf_scan` |
| 19 | **Witness-Verified Handoff (WVH)** | Multi-agent handoff with built-in hallucination filtering — WITNESS scans output before passing to the next agent | Prevents hallucination propagation across agent chains | `entroly/verified_handoff.py` | `from entroly import wvh_handoff` |

#### How they compose

Most levers are **multiplicative** with each other, not additive. A typical chatty agent benefits from #1 (input compression, 70%↓) **and** #3 (cache aligner, 90%↓ on whatever survives) **and** #6 (RAVS, route most calls to a cheaper model) **and** #11 (response distillation, fewer output tokens billed) **and** #16 (ESC universal fallback on tool outputs) **and** #17 (SRP budget-aware file reads). The product can leave less than 1% of the original input-token spend on the bill — without any accuracy hit, all measured with committed JSON artifacts.

If a feature isn't pulling its weight on your workload, the dashboard shows per-lever contribution (`http://localhost:9378`, "Cost Intelligence" panel).

#### Persistent Cross-Session Cache (EGSC)

Most "context cache" projects keep their decisions in process memory and start cold every time the agent restarts. Entroly's **Entropy-Gated Submodular Cache (EGSC)** survives restarts, cold starts, and brand-new sessions — the same admitted entries are reused across runs without rebuilding.

How it persists (no new code; all of this already ships):

```
EgscCache (entroly-core/src/cache.rs)
   ├── CacheSnapshot { entries, stats, config, schema_version }
   ├── export_cache() → JSON
   ├── import_cache(JSON)
       ▼ folded into the engine's full state
EntrolyEngine.export_state() includes cache_snapshot   (lib.rs)
EntrolyEngine.import_state(state) restores it on init  (lib.rs)
       ▼ driven by Python checkpoint/resume
engine.checkpoint() → ~/.entroly/checkpoints/<project_hash>/ckpt_*.json.gz
engine.resume()    ← latest checkpoint                  (entroly/server.py)
```

On every engine boot you'll see the warm-start line:

```
[entroly] Warm-start: restored 6 EGSC cache entries
```

Inspect the live cache + on-disk footprint anywhere the engine has run:

```bash
$ entroly cache stats
EGSC Persistent Cache
  checkpoint dir:        ~/.entroly/checkpoints/<project>/
  checkpoint files:      71  (54.3 MiB on disk)
  latest checkpoint age: 19.1h

Live cache
  entries:        6
  warm-restored:  6
  hit rate:       16.7%
  tokens saved:   102,272
```

Every proxied response also exposes the cache state on the wire (no extra setup):

| Header | Meaning |
|---|---|
| `X-Entroly-Cache-Entries` | Current live entry count |
| `X-Entroly-Cache-Hit-Rate` | Fraction in [0,1] |
| `X-Entroly-Cache-Hits-Exact` / `-Semantic` | Hits by lookup path (FNV-1a vs SimHash LSH) |
| `X-Entroly-Cache-Tokens-Saved` | Cumulative tokens served from cache |
| `X-Entroly-Cache-Warm-Restored` | Entries loaded from disk at process boot |
| `X-Entroly-Cache-Warm-Age-S` | Seconds since the warm-start restore |
| `X-Entroly-Cache-Source` | `persistent` / `mixed` / `session` |

Most context tools restart cold on every agent session. Entroly continues where the last session left off — including the bandit gate's posterior, the SimHash LSH index, and the cost-aware admission state.

### Packaged Self-Test Results

The core install and selection claims are checked against this repository itself (394 files, 901K tokens, Python/Rust/JS). Reproduce the packaged smoke check on any repo:

```bash
pip install entroly && cd /path/to/your/project
entroly verify-claims
```

| Claim | README | Verified | Status |
|---|---|---|---|
| **Indexing speed** | local, no API call | **0.66s** (394 files, release run) | Verified |
| **Token savings (32K budget)** | large-codebase selection should reduce context heavily | **96.7%** on this repo | Verified for this workload |
| **Token savings (8K budget)** | tighter budgets should reduce more | **99.1%** on this repo | Verified for this workload |
| **Token savings (average)** | workload-dependent | **87.0%** on this repo | Verified for this workload |
| **Optimization smoke latency** | local execution, benchmark separately for strict timing | emitted by `entroly verify-claims` and stored in `.entroly_verification.json` | Verified |
| **Multi-language coverage** | 10+ project types | **9 file types** (py/rs/js/md/yml/json/toml/sh) | Verified |
| **Entropy scoring** | Non-trivial | **0.07–0.90 range** | Verified |
| **Source-type prioritization** | Code > config | **Code 133 vs Config 12** | Verified |
| **SimHash deduplication** | No duplicates | **154/154 unique** | Verified |
| **Rust engine** | Rust + WASM | **entroly_core loaded** | Verified |
| **Local-only** | No API keys | **All ops offline** | Verified |
| **SDK** | 2-line import | **compress importable** | Verified |

> The packaged verifier generates a machine-readable `.entroly_verification.json` report. Results depend on repo size, language mix, and token budget; tiny repos and short-context workloads have less room to compress.

### Trust Benchmark — Zero API Keys, Zero Network

Five local checks that run in <2 seconds on a typical development machine, no API keys required:

```bash
python bench/trust_bench.py
```

| Test | What It Proves | Result |
|---|---|---|
| **A. Compression** | Real token reduction on source files | **50% savings** |
| **B. Classifier** | RAVS archetype accuracy (40 labeled prompts) | **100% accuracy** |
| **C. Hook Coverage** | Tool pattern coverage (50 commands) | **100% coverage** |
| **D. Router Logic** | Bayesian gate correctness (5 cases) | **5/5 correct** |
| **E. Determinism** | Same input → identical output (SHA-256) | **Bit-identical** |

### Code Retrieval — [CodeSearchNet](https://huggingface.co/datasets/code_search_net) (Established IR Benchmark)

"Given a docstring, find the correct function from 200 candidates." Public dataset, reproducible, no API key.

```bash
python bench/repobench_retrieval.py --samples 50 --pool-size 200
```

| Method | R@1 | R@5 | MRR | Latency |
|---|---|---|---|---|
| Top-K (FIFO) | 0.000 | 0.000 | 0.017 | 0.0 ms |
| BM25 (standard baseline) | **1.000** | **1.000** | **1.000** | 43.2 ms |
| **Entroly** | **1.000** | **1.000** | **1.000** | **18.6 ms** |

> Entroly matched BM25 on this run at **2.3× lower latency** (18.6ms vs 43.2ms). n=50 queries, pool=200, dataset=CodeSearchNet/python. [![Reproduce](https://img.shields.io/badge/Reproduce-locally-blue)](bench/repobench_retrieval.py)

### LooGLE Head-to-Head — RAG Compression Quality ([ACL 2024](https://github.com/bigai-nlco/LooGLE))

Apples-to-apples comparison at **identical 1,500 token budget**. Same LLM (gpt-4o-mini), same questions, same gold answers. n=30.

| Method | F1 Score | Compress Latency | API Calls | Illustrative cost / 1k queries |
|---|---|---|---|---|
| Baseline (Truncation) | 0.187 | 0 ms | 1 | $0.225 |
| Agentic Pruning baseline | **0.570** | 10,632 ms | 2 | $3.609 |
| **Entroly** | 0.223 | **107 ms** | **1** | **$0.225** |

> **The trade-off:** Agentic pruning (using an LLM to filter context) scored higher in this run, but it added **10.6 seconds of latency** and increased API costs by **1,500%**.
>
> **Entroly's local path:** It improved F1 over baseline truncation by **+19.2%** in this run, executing locally in 107ms with no extra model call.
>
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juyterman1000/entroly/blob/main/bench/colab_run.ipynb) ← One-click reproduction (Agentic Pruning vs Entroly, runs on H100 GPU)

Reproduce locally: `python bench/looGLE_compare.py --samples 30 --budget 1500`

### Code Retrieval — Entroly vs BM25 ([CodeSearchNet](https://huggingface.co/datasets/code_search_net))

Pure retrieval quality — no LLM calls, no API key, $0 cost. "Given a docstring, find the correct function from 500 candidates."

| Method | R@1 | R@5 | MRR | Latency |
|---|---|---|---|---|
| Top-K (FIFO) | 0.000 | 0.015 | 0.013 | 0.0 ms |
| BM25 (standard baseline) | 0.980 | 0.995 | 0.987 | 56.7 ms |
| **Entroly** | **0.990** | **0.995** | **0.993** | **28.1 ms** |

> **On this run, Entroly scored above BM25** on R@1 (+1.0%) and MRR (+0.6%), at roughly half the latency (28ms vs 57ms). n=200 queries, pool=500 distractors.

Reproduce: `python bench/repobench_retrieval.py --samples 200 --pool-size 500`

### How Entroly Compares (Long Context)

Representative long-context approaches and their trade-offs:

| Method | Retention | Token Reduction | Architecture / Trade-offs |
|---|---|---|---|
| **Entroly** | **100–106% on these runs** | **85–99% on long-context runs** | **Fast local selection + compression.** High-priority fragments are preserved verbatim; lower-priority files are compressed to signatures or references. Works with APIs that can receive the optimized prompt. |
| Agentic Context Pruning | ~100% | 70–90% | **Extremely slow.** Requires multiple LLM calls to filter context before the main query. High latency overhead. |
| KV Cache Compression | ~98–99% | N/A (Cost reduction) | **Hardware bound.** Reduces memory footprint, but requires running local models. Doesn't work for OpenAI/Anthropic APIs. |
| Token-level neural pruning | ~98–99% | 80–95% | **High overhead.** Runs BERT-base for token classification. Token-level dropping degrades code syntax. |
| RAG-specific reranking | ~98% | 60–80% | **RAG-specific pruner.** Good retention but lower token reduction than Entroly. |

*Note: SQuAD (~40% reduction, ~97% retention) is a short-context benchmark (150 token paragraphs). Entroly shows the largest reductions on large-context workloads.*

Reproduce: `python -m bench.accuracy --benchmark all --model gpt-4o-mini --samples 100`

**Custom OpenAI-compatible providers** (Groq, Together, OpenRouter, Ollama, vLLM, ...):

```bash
python -m bench.accuracy --benchmark gsm8k --model llama-3.1-70b-versatile \
    --base-url https://api.groq.com/openai/v1 --api-key-env GROQ_API_KEY
```

### SWE-bench Lite Retrieval Hit Rate

For coding agents, the first question is retrieval: did the context engine select the files that need to be modified? This benchmark measures whether Entroly captures SWE-bench Lite gold files in its selected context.

Measured on the local retrieval harness:

| Metric | Result | Why It Matters |
|---|---|---|
| **Hit Rate** | **100.0%** (50/50 tasks) | Each sampled task had at least one gold file captured. |
| Recall@5 | 42.0% | Fraction of gold files found in the top 5. |
| Recall@10 | 70.0% | Fraction of gold files found in the top 10. |
| Recall@20 | 90.0% | Fraction of gold files found in the top 20. |
| MRR | 0.420 | How early the first relevant file appears. |
| Latency | ~80ms / task | Local retrieval latency in the benchmark harness. |

> In this sample, every task had a required file represented in the selected context. This is a retrieval signal, not a guarantee that any specific model will solve every task.
> 
> *Reproduce:* `python -m bench.swebench_retrieval --samples 50 --engine rust`

### CI/CD Integration

Run token cost checks in every PR — catch regressions before they ship:

```yaml
- uses: juyterman1000/entroly-cost-check-@v1
```

Local fallback:

```yaml
- name: Check Entroly token budget
  run: pip install entroly && entroly batch --budget 8000 --fail-over-budget
```

---

## The Problem — AI Coding Tools Need Grounded Context

Two things often go wrong with AI coding workflows:

**1. Unsupported claims look plausible.** Models can mention functions, APIs, files, or dependencies that are not present in the evidence they were given.

**2. Large repos waste context budget.** Raw dumps include duplicated boilerplate, generated files, and low-signal text that crowd out the files the model actually needs.

> Entroly addresses both locally: it selects compact, explainable repo context under a token budget, and WITNESS can audit model outputs against supplied evidence.

---

## What Changes on Day 1

| Metric | Before Entroly | **After Entroly** |
|---|---|---|
| Files visible to AI | 5–10 | **Supported files selected at variable resolution** |
| Tokens per request | ~186,000 raw example | **9,300 – 55,000 in listed release examples** |
| Monthly AI spend (at 1K req/day) | depends on provider/model | **lower when input tokens drop** |
| AI answer grounding | depends on supplied context | **auditable against selected evidence** |
| Review burden | manual inspection | **certificate/evidence snippets available** |
| Setup | Days of prompt engineering | **30 seconds** |

> Savings depend on repo size, query breadth, model pricing, and budget. Run `entroly demo` or `entroly verify-claims` on your own repository for local measurements.

---

## How It Works (30 Seconds)

```bash
pip install entroly && entroly go
```

Or wrap your coding agent — one command:

```bash
entroly wrap claude       # Claude Code
entroly wrap cursor       # Cursor
entroly wrap codex        # Codex CLI
entroly wrap aider        # Aider
```

Or use the proxy — zero code changes, any language:

```bash
entroly proxy --port 9377
ANTHROPIC_BASE_URL=http://localhost:9377        your-app
OPENAI_BASE_URL=http://localhost:9377/v1        your-app
GOOGLE_GEMINI_BASE_URL=http://localhost:9377/v1beta  your-app
```

> **Why the different path suffixes?** They are *not* arbitrary tags. Each
> SDK appends its provider's real API path to its base URL, and the proxy
> routes by that path to the matching upstream: the Anthropic SDK calls
> `/v1/messages` (so the base URL has no suffix), the OpenAI SDK calls
> `/v1/chat/completions` or `/v1/responses` (base URL ends in `/v1`), and
> the Gemini SDK calls `/v1beta/models/...` (base URL ends in `/v1beta`).
> Use the suffix that matches the SDK you're pointing at the proxy; one
> proxy handles all three concurrently.
> Environment variable names are client-specific; use the base-url setting
> your SDK or CLI actually documents.

Drop it into your own code — two lines:

```python
from entroly import compress, compress_messages

# Compress any content (code, JSON, logs, prose)
compressed = compress(api_response, budget=2000)

# Or compress a full LLM conversation
messages = compress_messages(messages, budget=30000)
```

Or run the **single binary** — no Python runtime at all (zero-dependency Rust proxy, Anthropic/OpenAI/Gemini auto-detected, cache-aligned so the provider prefix cache keeps hitting):

```bash
git clone https://github.com/juyterman1000/entroly && cd entroly/entroly-core
cargo build --release --bin entroly-rs --features proxy
./target/release/entroly-rs proxy --upstream https://api.anthropic.com
ANTHROPIC_BASE_URL=http://localhost:9377  your-agent
```

> The binary is the lightweight proxy slice. For the **full product** — WITNESS hallucination guard, RAVS model routing, the learning engine, vault, MCP tools, dashboard — use `pip install entroly[full]`. (Prebuilt binaries + `curl | sh` ship with the next tagged release; see [Single Binary](#single-binary--entroly-rs-no-python).)

**Here's what entroly actually does, in plain English:**

1. **Reads your codebase locally** — every supported source file, config file, and document that passes the file filters.
2. **Figures out what matters** for your specific question (e.g. "fix this login bug" → pulls the auth files, ignores the marketing copy).
3. **Sends only the relevant parts** to your AI — a small, targeted bundle instead of a 200,000-token data dump.
4. **Can audit what your AI says back** — WITNESS checks factual claims against supplied evidence and records proof certificates.
5. **Flags unsupported claims** — unsupported or contradicted claims can be annotated, suppressed, or audited depending on profile.
6. **Learns from local feedback** — PRISM updates ranking weights when feedback signals are available.

> **The result for you:** Your AI can draw from a broader project map instead of a few open files, with a smaller selected context. Release checks measured 70-95% fewer input tokens on large-repo workloads; on small repos or short prompts, savings are naturally lower.

<sub>*Want the math? <a href="#works-with-your-stack">Skip to the technical details</a> or read <a href="docs/DETAILS.md">docs/DETAILS.md</a> for the full algorithmic spec (BIPT, NKBE, Causal Context Graph, Resonance Matrix, and more).*</sub>

---

## Live Dashboard & Control Panel

The interactive commands `entroly go`, `entroly proxy`, `entroly daemon`, and `entroly dashboard` open or serve a browser dashboard at `http://localhost:9378` — no extra install, no React build, nothing to configure.

**Dashboard** — real-time metrics (token savings, PRISM weights, health grade, cost savings, pipeline latency):

```
http://localhost:9378        ← auto-opens on entroly go / proxy / daemon
```

**Control Panel** — full control surface for the daemon:

```
http://localhost:9378/controls
```

| Control | What it does |
|---|---|
| **Optimization toggle** | Enable/pause context optimization |
| **Bypass mode** | Forward requests raw for A/B testing |
| **Quality selector** | Switch between Fast / Balanced / Max |
| **Repo manager** | See indexed repos, trigger re-index |
| **PRISM weights** | View learned weights, reset, run autotune |
| **Federation** | Opt-in/out of anonymous global learning |
| **Log viewer** | Real-time daemon logs in-browser |

> Everything is served inline from the Python package — `pip install entroly` includes the full UI. Zero npm, zero build step.

---

## Daemon Supervisor (`entroly daemon`)

One process that manages everything — proxy, dashboard, MCP server, file watcher, learning loop:

```bash
entroly daemon                 # start everything, opens browser
entroly daemon --no-proxy      # dashboard + MCP only
entroly daemon --quality max   # max quality mode
```

The daemon exposes a **Control API** at `http://localhost:9378/api/control/*`:

```bash
# Check daemon status
curl http://localhost:9378/api/control/status

# Toggle optimization
curl -X POST http://localhost:9378/api/control/optimization/pause
curl -X POST http://localhost:9378/api/control/optimization/enable

# Switch quality mode
curl -X POST http://localhost:9378/api/control/quality -d '{"mode":"max"}'

# Re-index a repo
curl -X POST http://localhost:9378/api/control/repos/reindex

# View learning weights
curl http://localhost:9378/api/control/learning

# Stop the daemon
curl -X POST http://localhost:9378/api/control/stop
```

> **Backward compatible:** Existing `entroly proxy`, `entroly serve`, `entroly dashboard` commands work exactly as before. The daemon is additive.

### Codebase Detection

If you run Entroly from a non-project directory (like your Desktop), it warns you:

```
  No codebase detected in: /Users/you/Desktop

  Navigate to your codebase first:
    cd /path/to/your/project
    entroly go
```

Entroly auto-detects Python, JS/TS, Rust, Go, Java, Ruby, C/C++, and 10+ other project types.

---

## The Competitive Edge — What Sets Entroly Apart

### Context Scaffolding Engine (CSE): structural maps for smaller models

Small, fast models can struggle on large codebases because they may miss cross-file relationships in raw code chunks.

Entroly's **Context Scaffolding Engine (CSE)** addresses this by extracting dependency-graph cues across supported languages. It then injects a compact structural preamble before the code context, mapping imports, definitions, test coverage, and entry points when available.

The result is not magic model equivalence; it is cheaper structure. CSE gives smaller models explicit dependency cues that are easy to miss in raw snippets. On scaffold-friendly code tasks, that can reduce the amount of "just in case" context while improving grounding. On judgment-heavy tasks, use a stronger model.

### RAVS — Guarded Cheaper-Model Routing

Entroly compresses your context. **RAVS can also evaluate whether repeated low-risk task classes are safe candidates for cheaper models.**

Many turns are simple: reading a file, checking a log, running tests, formatting code. Using a flagship model for these can be unnecessary spend.

RAVS watches honest outcomes and can be enabled as a guarded proxy router with `ENTROLY_RAVS_ROUTER=1`. Once local evidence passes the configured gate for a low-risk task class, it can route that task class down:

```
You type: "run the tests"
             ↓
  Entroly intercepts the request
             ↓
  RAVS checks confidence for this task type:
    → repeated low-risk task class
    → enough local outcomes to pass the configured gate
    → lower confidence bound > threshold
             ↓
  Eligible proxy request: stronger model → cheaper configured model
             ↓
  Same task class, cheaper model, original-model fallback if confidence drops.
```

> Inspect your own local evidence with `entroly ravs report`. Routing should stay disabled when sample sizes are small, task risk is high, or the confidence gate does not pass.

**How it works:**
1. Add one hook to `.claude/settings.json` — RAVS starts watching silently
2. Use your tools normally — every pass/fail outcome is recorded locally
3. Opt into routing with `ENTROLY_RAVS_ROUTER=1`
4. When local evidence passes the configured gate, routing can activate
5. If confidence drops or the request is high-risk, the original model handles it

**The numbers:**

| | Opus | Haiku (RAVS-routed) | Savings |
|---|---|---|---|
| Output cost / M tokens | higher | lower | provider-dependent |
| Typical repeated simple task | flagship model | cheaper configured model | measured locally |
| Monthly usage | varies | varies | inspect `entroly ravs report` |

Fail-closed by design: if data is sparse, the task is high-risk (`security`, `auth`), confidence is low, or the proxy cannot safely rewrite the provider request, the original model handles it.

```bash
# See what RAVS has learned about your workflow
entroly ravs report

# Filter to the last 7 days
entroly ravs report --since 7d
```

### Local Learning Without Extra Model Calls

Most of Entroly's ranking and feedback loops run locally. They do not require an embeddings API, fine-tuning job, or model call.

When optional synthesis or networked learning is enabled, it is intended to be budget-gated:

```
Learning budget target ≤ 5% × lifetime savings
```

By default, local context selection and dashboard metrics are enough to measure whether Entroly is helping on your workload.

### Federated Learning — Experimental and Opt-In

Federation is optional. It is designed to share anonymous optimization weights, not code.

- Your code should not leave your machine.
- Shared payloads are optimization statistics/weights.
- Enable only if you want to participate in cross-install learning experiments.

```bash
export ENTROLY_FEDERATION=1
```

### Response Distillation — Save Tokens on Output Too

LLM responses often include greetings, hedging, and meta-commentary. Entroly can strip common filler from prose while leaving code blocks untouched.

```
Before: "Sure! I'd be happy to help. Let me take a look at your code.
         The issue is in the auth module. Hope this helps!"

After:  "The issue is in the auth module."
         → fewer output tokens
```

Three intensity levels: `lite` → `full` → `ultra`. Enable with one env var.

### Local Indexing; Provider Requests Stay Under Your Control

Local indexing, selection, deterministic verification, and dashboards do not require a cloud service. If you proxy a cloud AI provider, that provider still receives the selected prompt content you send through Entroly. Core scoring paths are local and fast; full end-to-end optimization commonly runs in tens of milliseconds depending on repo size and engine mode. Air-gapped use is possible when you use only local/offline commands and local model endpoints.

See [Provider Compliance Notes](docs/provider-compliance.md) for the provider-specific checklist and official documentation links used for wrapper/base-URL support.

---

<a id="works-with-your-stack"></a>

## Works With Your Stack — Supported Integrations

`entroly wrap <agent>` uses the best available integration path for each supported tool. Use wrappers only with tools and accounts whose terms permit local MCP servers, custom endpoints, or compatible proxy configuration.

Third-party product names are used only to describe compatibility. Entroly is not affiliated with, sponsored by, or endorsed by those providers unless explicitly stated.

- **CLI agents** — for tools with supported custom endpoint variables, entroly starts the proxy, sets the endpoint, and execs the binary. Some tools may require their own provider config instead.
- **MCP-aware IDEs** — entroly auto-merges its MCP server into the IDE's `mcp.json` (with a `.entroly-backup` of any prior config). Restart the IDE.
- **Other IDEs** — entroly prints a best-effort endpoint/config hint. Exact setting names can vary by tool version; use only when the installed tool documents custom endpoint support.

### CLI agents (env-wrap, exec)

These wrappers are compatibility helpers, not endorsements by the tool vendors. If a vendor CLI does not honor custom endpoint environment variables in your installed version, configure Entroly through that tool's documented provider settings or use MCP instead.

| Agent | Command |
|---|---|
| Claude Code | `entroly wrap claude` |
| OpenAI Codex CLI | `entroly wrap codex` |
| Aider | `entroly wrap aider` |
| Gemini CLI | `entroly wrap gemini` |
| Qwen Code | `entroly wrap qwen` |
| OpenCode | `entroly wrap opencode` |
| Charm CRUSH | `entroly wrap crush` |
| Hermes | `entroly wrap hermes` |
| Pi Coding Agent | `entroly wrap pi` |
| Ollama | `entroly wrap ollama` |

### MCP-aware IDEs (auto-merge `mcp.json`)

| IDE | Command | Config file written |
|---|---|---|
| Cursor | `entroly wrap cursor` | `.cursor/mcp.json` |
| Windsurf | `entroly wrap windsurf` | `.windsurf/mcp.json` |
| VS Code MCP clients | `entroly wrap vscode` | `.vscode/mcp.json` |
| Claude Desktop | `entroly wrap claude-desktop` | OS-specific Claude config dir |
| Claude Code (MCP mode) | `entroly wrap claude-code` | Claude Code MCP config |
| Zed | `entroly wrap zed` | `~/.config/zed/settings.json` |

### Other IDEs (copy-paste snippet)

`entroly wrap <agent>` prints a best-effort endpoint/config hint. If your installed tool supports custom OpenAI-compatible endpoints, paste the shown URL into that tool's documented base URL / endpoint field and restart. These print-only helpers are not vendor certifications.

| Agent | Slug |
|---|---|
| Cline (VS Code) | `cline` |
| Roo Code (VS Code) | `roo` |
| Continue | `continue` |
| Sourcegraph Cody | `cody` |
| Sourcegraph Amp | `amp` |
| Kiro | `kiro` |
| Qoder | `qoder` |
| Trae | `trae` |
| Antigravity | `antigravity` |
| Amazon Q Developer | `amazonq` |
| Verdent | `verdent` |
| JetBrains AI Assistant | `jetbrains` |
| Helix | `helix` |
| Tabby | `tabby` |
| Twinny | `twinny` |
| Sublime Text | `sublime` |
| Emacs (gptel / aider.el) | `emacs` |
| Neovim (avante / codecompanion) | `neovim` |
| Fitten Code | `fittencode` |
| Tabnine Enterprise | `tabnine` |
| Supermaven | `supermaven` |

### Any agent that supports custom base URLs

Entroly's proxy (`localhost:9377`) works with tools that let you override their API endpoint. If your agent supports `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `GOOGLE_GEMINI_BASE_URL`, or similar documented settings, point it at the proxy and test the workflow with your provider/model.

> **Cloud-hosted agents** (Devin, Jules, Replit Agent, etc.) run in the vendor's cloud, not on your machine. Check your provider's documentation to see if they support custom base URLs before attempting to proxy through entroly. Always review the provider's Terms of Service.

### Library / framework integration

| Use case | One-liner |
|---|---|
| **LLM APIs with compatible base-URL configuration** | `entroly proxy` → HTTP proxy on `localhost:9377` |
| **LangChain / LlamaIndex / your code** | `from entroly import compress, compress_messages` |
| **Nous Hermes (Local/ChatML)** | `from entroly.integrations.hermes import safe_compress_hermes` |
| **CI / token-budget gate** | `entroly batch --budget 8000 --fail-over-budget` |

Also: OpenAI-compatible APIs, Anthropic-compatible clients, OpenRouter, Ollama, vLLM, and other providers/tools that allow compatible endpoint configuration.

> Don't see your tool? `entroly wrap` (no agent) prints the full grouped list, and the [Cookbook](cookbook/README.md) has copy-paste recipes for the most common workflows.

---

<a id="benchmarks"></a>

## Compared to

Entroly **selects and compresses** context. The difference is ordering: it ranks the repo first, then compresses lower-priority material to signatures or references instead of blindly compressing/truncating whatever was provided.

| | **Entroly** | Compression tools | Top-K / RAG | Raw truncation |
|---|---|---|---|---|
| **Approach** | Information-theoretic selection + compression | Text compression | Embedding retrieval | Cut-off |
| **Token savings** | **Tested 70-95% on large-repo release checks; workload-dependent** | 50–70% | 30–50% | 0% |
| **Quality loss** | **No measured loss in listed release checks** | 2–5% | Variable | High |
| **Multi-resolution** | **Full / Skeleton / Reference** | One-size | One-size | One-size |
| **Learns over time** | **Yes (PRISM RL)** | No | No | No |
| **Latency** | **Local; commonly tens of ms end-to-end** | 50–200ms | 100–500ms | 0ms |
| **Reversible** | **Yes** — full content always retrievable | Varies | Yes | No |
| **Runs locally** | **Yes** | Varies | Varies | Yes |

> **Why selection + compression matters:** Compressing a bad selection is still a bad selection. Entroly ranks files first, then compresses or preserves them at a resolution appropriate to the budget. The AI receives structural context, not just fewer tokens.

---

## Watch It Run — Live Notifications

Three chat integrations ship in the box. They can report selected gap detections, skill synthesis events, and dream-cycle changes in real time:

```bash
export ENTROLY_TG_TOKEN=...          # Telegram (2-way: /status /skills /gaps /dream)
export ENTROLY_DISCORD_WEBHOOK=...   # Discord
export ENTROLY_SLACK_WEBHOOK=...     # Slack
```

---

## Portable Skills (agentskills.io)

Skills Entroly creates aren't locked in. Export to the open agentskills.io v0.1 spec:

```bash
node node_modules/entroly-wasm/js/agentskills_export.js ./dist/agentskills
python -m entroly.integrations.agentskills ./dist/agentskills
```

Structurally synthesized exports carry `origin.token_cost: 0.0`, so zero-token provenance travels with those skills.

---

## Python and Node.js Surfaces

Python is the reference CLI/runtime. The Node.js WASM package exposes the Rust engine and a matching surface for core workflows; check command help for feature-specific availability:

| Capability | Python | Node.js (WASM) |
|---|---|---|
| Context compression | Yes | Yes |
| Self-evolution | Yes | Yes |
| Dreaming loop | Yes | Yes |
| Federation | Yes | Yes |
| Response distillation | Yes | Yes |
| Chat gateways | Yes | Yes |
| agentskills.io export | Yes | Yes |
---

## Single Binary — `entroly-rs` (no Python)

A standalone, zero-dependency Rust binary that runs the context-compressing proxy with **no Python runtime** — the frictionless drop-in for cutting input tokens. It auto-detects **Anthropic, OpenAI, and Gemini** by request path, compresses the request context, and streams responses straight back. Compression is **cache-aligned** by default (each field compressed independently so unchanged prefixes stay byte-stable → the provider's prefix cache keeps hitting), with concurrency, gzip-correct streaming, an upstream connect timeout, and per-request panic isolation.

> **Scope:** this is the lightweight proxy slice. For the full product — WITNESS hallucination guard, RAVS model routing, the learning engine, vault, MCP tools, dashboard — use the Python package: `pip install entroly[full]`.

**Build & run (works today):**

```bash
git clone https://github.com/juyterman1000/entroly
cd entroly/entroly-core
cargo build --release --bin entroly-rs --features proxy

# Point your agent at it (Anthropic shown; OpenAI/Gemini auto-detected too):
./target/release/entroly-rs proxy --upstream https://api.anthropic.com
ANTHROPIC_BASE_URL=http://localhost:9377  your-agent
```

**Compress a file or stdin (no proxy, no Python):**

```bash
./target/release/entroly-rs compress --budget 4000 path/to/file
cat path/to/file | ./target/release/entroly-rs compress --budget 4000
```

**Proxy flags:** `--port` (default 9377) · `--upstream` · `--budget` (default 4000) · `--no-cache-align` (use a single global token budget instead of the prefix-stable default).

> Prebuilt binaries and a one-line `curl | sh` installer ship with the next tagged release (`scripts/proxy_e2e.py` and `install.sh` are already in the repo).

---

## Deep Dive

Architecture, Rust modules, 3-resolution compression, provenance model, RAG comparison, CLI reference, Python SDK, LangChain integration → **[docs/DETAILS.md](docs/DETAILS.md)**

---

<p align="center">
  <b>Measure and reduce wasted context tokens with local, evidence-aware tooling.</b><br/>
  <code>npm install -g entroly && entroly</code>&nbsp;&nbsp;|&nbsp;&nbsp;<code>pip install entroly && entroly go</code>
</p>

<p align="center">
  <a href="https://github.com/juyterman1000/entroly/discussions">Discussions</a> •
  <a href="https://github.com/juyterman1000/entroly/issues">Issues</a> •
  Apache-2.0 License
</p>
