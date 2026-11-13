# Entroly intensive dogfooding: corrected, falsifiable evidence (2026-11-13)

**Campaign:** [draft PR #592](https://github.com/juyterman1000/entroly/pull/592). Baseline main: `6286baff5e29022ff1a593a649ba717496172df9`. These are **in-process and model-free trials**, not evidence of agent task success or billable dollars saved.

## Critical falsification: the first report was overly optimistic

The original `benchmarks/intensive_dogfood_trials.py` silently shortened source functions to the first **2,700 characters**. **7 of 12** test definitions in the initial checkout were longer. Earlier claims of “48/48 byte-exact recovery” were comparisons to these truncated examples, **not** full original code. The corrected harness retains complete AST source, verifies it with a regression test, and removes recovery decisions based on the known testcase target. Instead it uses only the user query and omitted-source registry and refuses ambiguous matches.

**Do not reuse pre-correction performance/recovery headlines without this qualification.**

## Corrected real-source results — native engine, head 18d94013

[GitHub Actions run](https://github.com/juyterman1000/entroly/actions/runs/43489247044) independently installed native engine on Ubuntu. Data artifact: `intensive-dogfood-native-source-evidence`; its embedded SHA and expected head both equal `18d94013c130abeeb0adf01cfff6579360194eec`, and `provenance_validated=true`.

| Metric | Value | Qualification |
| --- | --- | --- |
| Planned / completed | 48 / 48 | 12 actual full-source AST symbols × 2 budgets × 2 seeds |
| Target source selected | 48 / 48 | Source presence **may be excerpted** |
| First pass contains entire source | 8 / 48 | 16.7%; no oracle label in retrieval |
| Complete source restored | 48 / 48 | Exact symbol-name lookup from query, not autonomous agent detection |
| Second-attempt context exceeds original budget | **40 / 48** | **83.3%**; essential cost and capacity caveat |
| Median first-pass estimated reduction | 93.56% | Estimated characters/4; not provider tokens |
| Median native selection latency | 60.198 ms | In-process, no network |
| p95 native selection latency | 185.879 ms | In-process, no network |
| Total modeled input tokens, RAW | 636,048 | Sum over all 48 trials |
| Modeled input tokens, first pass + one recovery attempt | 127,370 | Sums 51,488 first-pass + 75,882 second-pass |
| Overall modeled paired reduction | **79.97%** | Both attempts, not outputs, cache or invoices |

The local Python fallback full-source experiment completed **40/48**, with **8 selection-unavailable** and 60% second-attempt budget overruns among completed trials. Local measurements were on an exploratory working copy, not a published model-quality result. The final branch reruns the native experiment and now emits paired cost fields directly.

### Proxy transform replay

The same native workflow ran 25 cumulative turns per OpenAI/Anthropic payload format using synthetic tool outputs. Both retained the initial exact source file, a traceable repeated-read pointer and the fatal error marker. The last request body was ~94.9% smaller in serialized bytes. This is **not** billed-token reduction or model end-to-end latency.

## Independently exercised installation and security surfaces

The `43489247044` run's **fresh isolated wheel** jobs all succeeded for Ubuntu, macOS and Windows (Python 3.12). Wheels were built from the branch, installed in new virtual environments outside the source checkout, and the native runtime / CLI were checked. The cross-platform success pertains to that SHA; newer commits require fresh CI.

The existing repository Deep Dogfood Runtime independently includes installed MCP entrypoints, Bandit, pip-audit and Cargo advisories, while CodeQL runs independently. This is meaningful coverage, **not** a comprehensive 400+ module manual security review.

## Confirmed defects corrected in PR #592

1. Local or operator pricing rates `Infinity`, `NaN`, or malformed text could leak nonfinite amounts or unhandled decimal exceptions. Invalid rates now fail closed, with tests.
2. Invalid provider usage values such as `Infinity`, non-integer, negative and over-large counters could crash or be silently converted to **zero**, creating false priced spend. Now malformed present fields raise `ValueError`; the proxy catches accounting errors rather than changing provider responses. Missing counters in a legitimate partial stream remain zero.
3. A provider reporting more cached input than total input was silently clamped and still priced. Contradictory counts now fail closed.
4. Empty usage metadata was able to masquerade as valid zero-cost evidence; now a provider-specific recognized counter is required.
5. The source-retention harness clipped original source and supplied the expected target directly to recovery. It now retains full source, uses deterministic query-symbol retrieval without testcase target labels, and counts second-attempt context separately.

Local regression suite after fixes: **191 passed** for pricing/receipt/proxy/retrieval cases. `tests/test_intensive_dogfood_trials.py` checks full-source fidelity, exact-symbol retrieval, ambiguity refusal and digest-gated restoration.

## Phases still unverified

| Phase | Evidence |
| --- | --- |
| Live coding-agent RAW/COMPRESS/CLOSED_LOOP A/B/C | NOT EXECUTED: no authorized runnable model endpoint in this investigation |
| Independent provider invoice reconciliation | NOT EXECUTED: no API credentials, billing invoices or paired live traffic |
| Statistically powered task success comparison | NOT EXECUTED: preregistered 400-task holdout experiment has not run |
| Exhaustive security/manual module audit | NOT EXECUTED: existing static and adversarial checks are **partial** |
| End-to-end model success/quality, retry effects and billed dollars | UNKNOWN: do not infer from source selection or synthetic bytes |

**Verdict: CONDITIONAL / RESEARCH ONLY.** Keep draft PR #592 unmerged pending latest-head CI and review. Do not claim a production ROI, user task-success improvement, or 400-task non-inferiority result from these artifacts.
