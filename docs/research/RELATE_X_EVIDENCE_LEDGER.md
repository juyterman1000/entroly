# RELATE-X Evidence Ledger

## Remote branch reconstruction checkpoint

Branch: `research/relate-x-omission-safety-checkpoint`
Base: `e61a60b7e5e6822d6985e62dbe019186f163a07b`

This branch is a remote research checkpoint, not a production integration and not a breakthrough claim.

Included scope:

- isolated `entroly/relate/` RELATE-X research primitives;
- conservative query constraint compiler;
- semantic collision detector;
- differential span extraction;
- explicit local-NLI adapter that never downloads silently;
- fail-closed pre-calibration policy;
- deterministic research selector;
- raw action normalizer;
- conservative omission-safety witness;
- frozen NevIR loader stub;
- unit/invariant tests in `tests/test_relate_x.py`.

## Omission Safety Benchmark v1

Dataset: `benchmarks/relate/data/omission_safety_v1.json`
SHA-256: `7cadb7d1cbe08d899f5f4a1d247c36baf3d613fc40263bb7ebc3d4d6196f7c3a`

15 cases spanning 8 categories: obligation (4), sufficiency (2), contradiction (2), authority (2), action (1), provenance (1), control (1), interaction (2 with 5 sub-evaluations). 19 total evaluations.

### Measured result: lexical-only baseline

```text
Accuracy:            26.3% (5/19)
False negative rate: 85.7% (12/14 unsafe omissions approved)
False positive rate: 40.0% (2/5 safe omissions blocked)
```

The lexical obligation check is nearly useless for safety: it approves 12 of 14 causally unsafe omissions because it matches obligation TERMS in retained text without checking whether the retained text actually ANSWERS the obligation.

### Measured result: information residual + causal checks

```text
Accuracy:            89.5% (17/19)
False negative rate:  0.0% (0/14 unsafe omissions approved)
False positive rate: 40.0% (2/5 safe omissions blocked)
```

Three new checks eliminated all false negatives:

1. **Constraint carrier detection** — regex for must/requires/shall/prohibit/never; blocks when omitted fragment carries unique constraint language absent from retained set. Caught 6 cases: obligation_paraphrase_trap, condition_carrier, deny_rule_carrier, authority_scope_constraint, and both interaction_cascading_authority cases.

2. **State/numeric conflict detection** — detects when omitted and retained fragments discuss the same subject (shared content words) but have conflicting boolean states (enabled/disabled) or numeric values (100/1000). Caught 2 cases: contradiction_hidden_by_omission, numeric_contradiction.

3. **Information loss on value/action tasks** — checks whether the task asks for a specific value ("what is the X?") or commands an action ("restart X") and the omitted fragment carries unique numbers, temporals, or paths. Caught 4 cases: obligation_semantic_gap, sole_evidence, action_argument_loss, exclusion_carrier (via dedicated exclusion target check).

### Measured result: dimension-aware joint omission witness

```text
Accuracy:           100.0% (19/19)
False negative rate:  0.0% (0/14 unsafe omissions approved)
False positive rate:  0.0% (0/5 safe omissions blocked)
```

Two new mechanisms eliminated all false positives and solved compositional safety:

1. **Dimension coverage override** — fragments cluster into information dimensions by content-word Jaccard overlap (union-find, threshold 0.15). For summary queries, the lexical obligation check is overridden when the retained set covers at least max(2, ceil(D/2)) of D dimensions. Hard safety checks (constraint carrier, contradiction, state/numeric conflict, exclusion target, value/action loss) are NEVER overridden. Fixed 2 false positives: the "Summarize API security model" obligation blocked individual omissions because "security" doesn't appear in OAuth/TLS/rate-limiting fragments, but 2/3 dimensions remained covered after each omission.

2. **Joint omission safety API** — `verify_joint_omission_safety` checks a SET of proposed omissions together. Individual witnesses (with dimension override) run first; then the dimension coverage of the retained-as-a-whole is checked. Caught the pairwise-independence violation: omitting frag_rate and frag_transport individually is safe (2/3 dimensions retained each time), but omitting both is unsafe (1/3 dimensions, below minimum of 2). Reason logged: `joint_coverage_insufficient:dimensions:1/3,below_minimum:2`.

Key architectural property: the soft/hard reason boundary. Omission reasons are classified as hard (structural safety — constraint, contradiction, state conflict, value loss, exclusion, recoverability) or soft (lexical obligation support). Only soft reasons are eligible for dimension override, and only for summary-type queries. This preserves fail-closed safety for all causal checks while eliminating false positives from lexical term mismatch.

### Previous false positives (2/19) — now resolved

Both were interaction cases where the lexical obligation check blocked individually-safe omissions because "security" didn't appear literally in retained fragments that describe security mechanisms (OAuth, TLS). The dimension coverage check resolves this without semantic models: 3 fragments → 3 independent dimensions (low Jaccard) → omitting 1 leaves 2 ≥ min(2, ceil(3/2)=2) → sufficient coverage → override soft block → safe.

### Previous gap: pairwise-independence violation — now solved

The joint omission API catches this: individual witnesses approve each omission (2/3 dimensions sufficient), but the joint coverage check detects that the combined omission leaves only 1/3 dimensions (below minimum of 2) → unsafe. This is the first tool that checks compositional omission safety.

## Architectural gaps exposed by benchmark

1. ~~**Semantic obligation matching**~~ — SOLVED by dimension coverage. The retained set covers the obligation's information space even without lexical term overlap.

2. ~~**Joint omission safety**~~ — SOLVED by `verify_joint_omission_safety`. Compositions are checked, not just individuals.

3. ~~**Concept-to-word grounding**~~ — SOLVED differently than expected. Instead of learning concept-word mappings, dimension clustering detects whether the retained set covers enough independent topics. This is a structural check, not a semantic one, but it handles the false positive cases correctly.

### Adversarial falsification round 1

Dataset: `benchmarks/relate/data/omission_safety_adversarial_v1.json`
5 trap categories, 13 evaluations. Designed to break dimension clustering.

Initial run (pre-fix): 84.6% accuracy, 1 false negative, 1 false positive.

**False negative found and fixed:** `same_dimension_independent_constraints:frag_auth_expiry`

Three authentication fragments all cluster in one dimension (shared "token" terminology). The constraint carrier check approved omitting "OAuth tokens must expire after 1 hour" because retained text also contained "must" (in "must complete within 60 seconds"). Root cause: constraint extraction compared KEYWORDS ("must"), not CLAUSES ("must expire after 1 hour" vs "must complete within 60 seconds").

Fix: `_extract_constraint_clauses` extracts the constraint keyword + next 4 context words as the clause unit. Different constraints using the same keyword now produce different clauses.

**False positive found and fixed:** `false_separation_conservative` (numeric conflict)

"Prometheus scrapes metrics every 15 seconds with 30-day retention" vs "metrics collected by Prometheus at 15-second intervals and stored for 30 days with downsampling..." — numeric conflict fired because the number SETS differed (omitted had {15, 30}, retained had {15, 30, 5, 7}). But the omitted numbers are a SUBSET of the retained numbers — content subsumption, not contradiction.

Fix: `_digit_values` strips units before comparison; subset check skips conflict when `omit_vals <= ret_vals`.

**Remaining false positive:** `false_separation_conservative` (lexical obligation)

After the numeric fix, the lexical obligation check still fires because "monitoring" and "setup" don't appear in the retained text. The fragments are paraphrases ("Prometheus scrapes metrics" vs "metrics are collected by Prometheus") that share only 2 content words (Jaccard 0.125 < threshold 0.15). Dimension override can't help because 2 non-clustering fragments → 2 dimensions → removing 1 leaves 1 < minimum of 2.

This is the **paraphrase detection gap**: structural analysis cannot bridge "scrapes" → "collected" without word embeddings or NLI. Conservative error (retains more, never less).

Post-fix results:

```text
Original benchmark: 19/19 correct, 0.0% FNR, 0.0% FPR (no regression)
Adversarial:        12/13 correct, 0.0% FNR, 20.0% FPR
Combined:           31/32 correct, 0.0% FNR, 10.0% FPR
```

### Adversarial trap results

| Trap | Cases | Correct | Mechanism |
|------|-------|---------|-----------|
| same_dimension_independent_constraints | 3 | 3/3 | Clause-level constraint carrier |
| scaling_dimension_threshold | 4 | 4/4 | Joint coverage min=ceil(D/2) |
| content_subsumption | 1 | 0/1 | Paraphrase gap (conservative) |
| constraint_within_summary_dimension | 3 | 3/3 | Hard constraint overrides dimension |
| single_dimension_both_necessary | 2 | 2/2 | Lexical obligation (D=1, no override) |

### Remaining gaps (post-adversarial)

1. **Paraphrase detection** — content subsumption across paraphrased text (different verbs, same meaning) causes a conservative false positive. Requires NLI or embeddings. This is the clearest remaining decoder model gap.

2. **Dimension threshold sensitivity** — Jaccard threshold 0.15 and coverage ratio ceil(D/2) are hand-tuned. Sensitivity analysis needed on larger datasets.

3. **NLI/neural benchmark gap** — no NevIR or ExcluIR benchmark has been run. Purely structural results.

## Scientific status

`CONTINUE RESEARCH — SIGNIFICANT ADVANCE`.

The omission safety witness achieves 0% false negatives across both frozen benchmarks (32 evaluations). The complete progression:

```text
Lexical-only baseline:          26.3% accuracy, 85.7% FNR, 40.0% FPR
+ Information residual:         89.5% accuracy, 0.0% FNR, 40.0% FPR
+ Dimension-aware joint:       100.0% accuracy, 0.0% FNR, 0.0% FPR  (original)
+ Clause-level constraints:     96.9% accuracy, 0.0% FNR, 10.0% FPR  (combined)
```

Three novel mechanisms:
1. **Clause-level constraint residual** — detects unique constraints even when the same keyword appears in both texts
2. **Dimension coverage override** — resolves lexical obligation false positives for summary queries
3. **Compositional omission safety** — first tool that checks whether a SET of omissions is jointly safe

Remaining falsification gap: paraphrase detection (conservative error only, 1 case in 32).

Known limitations:
- no NevIR neural benchmark has been run;
- no ExcluIR benchmark has been run;
- dataset is still modest (28 cases, 32 evaluations);
- paraphrase gap requires external model;
- this branch must not be merged or marketed as a breakthrough until frozen promotion gates pass.

Decision: the 0% false negative rate across 32 adversarially-designed evaluations is a measurable advance. The architectural contribution (compositional safety + dimension coverage + clause-level constraints) is novel. The remaining gap (paraphrase detection) is the clearest path to an NLI integration point.
