# Entroly production dogfooding — evidence and remaining gates
Date: 2026-11-14. Draft PR: [#594](https://github.com/juyterman1000/entroly/pull/594). Starting baseline main: 2c3ff80f67ecd948308430d180488e450cdf1a0f.

## A. Billing verification software — TESTED, but NOT genuine provider billing

[Workflow 43613862356](https://github.com/juyterman1000/entroly/actions/runs/43613862356) recorded **42 passed** contract tests (invoice mismatch, missing or duplicate request IDs, nonfinite money, foreign currency, unpriced cost, unsafe patches and sandbox restrictions).

The request-level reconciliation script, benchmarks/provider_invoice_reconcile.py, requires a normalized vendor export containing request_id, provider, model, billed_usd, currency and complete Entroly UsageLedger.export_csv() data. It fails closed when IDs or charges cannot be matched, and hashes source exports for audit. It does not authenticate the vendor, reconstruct request charges from aggregate invoices, or verify invoice-period completeness.

**No actual OpenAI, Anthropic or Gemini billed export is available. No real dollar-savings claim is licensed.**

## B. Real-commit model experiment — EXECUTED, NEGATIVE

The same [remote run](https://github.com/juyterman1000/entroly/actions/runs/43613862356), artifact production-dogfood-real-history-model, used exact head 1eb8e57644de0afb55f0bd888261b320cf396537 and Qwen2.5-Coder 3B model digest f72c60cabf6237b07f6e632b2c48d533cef25eda2efbd34bed21c5e9c01e6225.

Eight previously validated Entroly source/test commits were **revalidated 8/8**. Under the frozen 18,000-byte context limit, **6/8 were too large**. Two real coding bugs were eligible for RAW, COMPRESS and RAW_REPEAT, producing six genuine generations, tested by independent networkless Docker pytest oracles.

| Metric | RAW | COMPRESS | RAW_REPEAT |
| --- | ---: | ---: | ---: |
| Eligible real-repository bugs | 2 | 2 | 2 |
| Applicable model-generated fixes | **0/2** | **0/2** | **0/2** |
| Model-reported prompt tokens | 7,072 | 1,998 | 7,072 |
| Model-reported output tokens | 1,536 | 988 | 1,536 |

The model often generated truncated, malformed or disallowed unified diffs. COMPRESS used about 71.75% fewer prompt tokens, but RAW also generated zero applicable fixes. Therefore this dataset cannot establish coding task-quality preservation or production cost savings.

The code in tests comes from public Entroly history. Generated patches are limited to approved source files and are tested in a nonroot, network-disabled, cap-dropped Docker environment with a read-only worktree bind mount and CPU/memory/PID limits. Exclusions and negative model answers remain in the machine-readable artifact.

## C. Preregistered 400-task decision gate — IMPLEMENTED, CRITERIA NOT MET

The new benchmarks/preregistered_study_verdict.py rejects:
- Fewer than 400 distinct real tasks, or fewer than 200 each in independent dev and holdout repositories; missing paired RAW/COMPRESS/CLOSED_LOOP arms; repeated task IDs and mismatched seeds.
- Invalid fail-to-pass oracles, missing model/token provenance, repo leakage between partitions and changed model identity within a task.
- A >10 percentage-point RAW success shift between dev and holdout (F1), <30% measured median input-token reduction or a failing -3pp repository-cluster confidence lower bound (N1).
- Missing provider-authenticated invoices and closed-loop cost-per-success dominance (D1), or paid study spend above the frozen budget (B1).

Even if supplied metrics meet every arithmetic threshold, the script returns external_review_required rather than production-certified: JSON cannot authenticate underlying invoices or source code.

## D. Security scope

The earlier [PR #592](https://github.com/juyterman1000/entroly/pull/592) passed full CI, CodeQL and multi-OS native wheel tests. This PR adds patch-mode and local-endpoint guards, nonroot networkless test execution, and adversarial accounting checks. This is meaningful but partial coverage, **not** a full code-by-code manual security review or independent penetration test.

## Explicit blockers — still NOT EXECUTED

1. 400+ independent, real-repository test-oracle coding tasks, repository-disjoint development and holdout sets. Do not pad with repeated seeds or synthetic tasks.
2. True multi-turn tool-using agents performing read/edit/test/recovery under paired budgets and end-to-end latency measurement.
3. Authorized real paid model requests plus actual vendor billing exports, covering input, output, cached tokens, retries and failed generations.
4. Statistical preregistered noninferiority and success-per-dollar gates on independent tasks, with external provenance verification.
5. Full independent audit of trust boundaries, model output execution, key handling, injection, filesystem permissions, recovery leakage and dependency vulnerabilities.

## Final assessment

Some engineering and security contracts are verified. The real-commit pilot executed and found a capability floor; its quality result is negative/inconclusive. End-to-end accuracy claims, invoice-backed ROI and comprehensive deep dogfooding remain **NOT VERIFIED**. Keep PR #594 draft and do not publish production noninferiority or dollar-savings claims.
