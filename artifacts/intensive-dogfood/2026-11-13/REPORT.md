# Intensive Entroly dogfood: independent execution evidence (2026-11-13)

## Status — executed vs unverified

Campaign PR: [#592](https://github.com/juyterman1000/entroly/pull/592); baseline `main` `6286baff5e29022ff1a593a649ba717496172df9`.

This is **measured deterministic dogfooding**, not full production-validation or the pre-registered 400-task model experiment.

| Phase | Evidence | Status |
| --- | --- | --- |
| 0 baseline / architecture | Exact-head GitHub Actions source archive, static import graph | EXECUTED |
| 1 install/user paths | Installed native engine on GitHub Actions; local Python 3.13 fallback | EXECUTED ON LINUX, not a fresh Windows/macOS consumer install |
| 2 true agent task completion | No comparable paid/live coding-agent runs | NOT EXECUTED |
| 3 proxy fidelity/performance | Two-provider 25-turn in-process replay (50 cumulative requests); file view and fatal marker checked | EXECUTED (synthetic tool outputs) |
| 4 provider invoices / full ROI | No paid API calls; no provider bills; byte/token estimates labeled correctly | NOT EXECUTED |
| 5 adversarial trust | Proof-bundle, context commit, recovery fuzz, malformed usage/rates, proxy protection | PARTIAL / tests executed |
| 6 code graph and audit | 412 Python modules, 1,149 static edges, 46 statically unreached modules (14,953 lines), 4 import cycles | PARTIAL (static only) |
| 7 defects and repairs | Nonfinite pricing rejected in usage ledger; infinite provider token counters do not raise OverflowError | FIX IN PR |
| 8 production gates | Independent CI + preexisting CI; security and real-provider oracles outstanding | CONDITIONAL |
| 9 machine-readable artifacts | GitHub Actions artifacts and exact SHA/provenance | EXECUTED |
| 10 GitHub delivery | Draft PR #592 with code, experiments, tests, workflow | OPEN / NOT MERGED |

## Run A — Python fallback (local)

Using the exact-head source snapshot at `10a426696cccc85fd8f9caefdbc1a7d318f9cf21`:
- 48 planned source-evidence trials, 40 measured, 8 returned no usable fragment at the stated budget.
- First-pass source hit rate in completed trials: 40%.
- Exact full-source recovery succeeded for all 40 completed trials.
- Median *estimated* context reduction among completed trials: 78.84%.
- A constrained budget on larger functions can be infeasible. This does not mean all source is unrecoverable or that an LLM necessarily fails.

## Run B — production native engine 1.0.88 on Linux GitHub Actions

Reproducible [trial workflow](https://github.com/juyterman1000/entroly/actions/runs/43483092179), head `15de3f21a74925a1d34dc94d0481a675078dd06c`.
- 48/48 selections produced a usable result; target source appeared in 48/48.
- Exactly full original target source appeared on first pass in 8/48 (16.7%) cases; excerpts are **not** automatically failures of an LLM task.
- The full target original was recoverable in 48/48 using the content-digest-verified closed-loop arm.
- Median first-pass **estimated** context reduction: 89.13%, native median selector latency: 33.658ms; p95 37.088ms (GitHub runner, no provider).
- These are context-retention measurements, not agent task-success rates or actual provider tokens.

### 25-turn proxy transform replay

Same [native workflow](https://github.com/juyterman1000/entroly/actions/runs/43483092179), synthetic tool data, 25 cumulative turns for each provider:
- OpenAI-format: final-request serialized-byte reduction 94.91%; median transform 37.606ms; p95 39.690ms.
- Anthropic-format: final-request serialized-byte reduction 94.92%; median transform 37.781ms; p95 39.492ms.
- Initial file views remained byte-exact and repeat views retained pointers to the original; the fatal `E_CONNRESET` marker survived.
- **Do not call these token savings, invoice savings, HTTP latency, or an agent-success comparison.**

## Local static and adversarial verification

- 245 existing tests passed over agent arms, task miner, proof bundles, context commits, 73 recovery fuzz cases, proxy fidelity, and header forwarding.
- Before the usage fix, `UsagePricing.from_values(..., 'Infinity')` accepted a nonfinite amount; `NaN` and invalid strings leaked `decimal.InvalidOperation`.
- Before the overflow fix, OpenAI/Anthropic/Gemini usage with `Infinity` token counters raised `OverflowError` in `_int`.
- New regression cases validate fail-closed price handling, numeric-literal compatibility, and the three provider usage parsers. Existing policy treats malformed provider token counters as zero: this prevents an exception **but is not evidence of accurate provider spending**.
- Python source compileall succeeded and static codebase graph completed.

## Independent replay commands

```bash
python benchmarks/intensive_dogfood_trials.py --out native-source-evidence.json
python benchmarks/intensive_proxy_replay.py --out native-proxy-replay.json
python -m pytest -q tests/test_usage_ledger.py tests/test_net_savings.py
python -m pytest -q tests/test_proof_bundle.py tests/test_context_commit.py tests/test_recovery_fidelity_fuzz.py
python scripts/codebase_graph.py --json graph.json
```

`Intensive Dogfood Evidence` runs these against the installed, exact PR head on GitHub Linux runners and uploads JSON and test logs.

## Remaining blockers

No live provider credentials, authorized paid billing, side-by-side real coding agents, or independent production invoices were available. No attempt was made to fabricate them. A native model-free corpus cannot prove no task-quality regression; output-token spend, retries, cache economics, and actual bill savings remain **NOT TESTED**. An exhaustive 412-module manual audit was not executed.

**Readiness verdict:** valuable new evidence and regression hardening, but **production ROI and real-agent-task non-inferiority remain unverified**. Keep this PR draft until required CI passes and reviewers approve the scope; do not merge merely because source-retention metrics are green.
