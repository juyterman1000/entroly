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

### Remaining false positives (2/19)

Both are interaction cases where the lexical obligation check blocks an individually-safe omission because the word "security" doesn't appear literally in retained fragments that describe security mechanisms (OAuth, TLS). This is the semantic gap that requires a decoder model: the retained set EMBODIES the concept without using the WORD. Conservative (retains more than necessary) — acceptable for safety.

### Key finding: pairwise-independence violation

The interaction cases prove that pairwise-safe omissions can be jointly unsafe. Omitting `frag_rate` alone is safe; omitting `frag_transport` alone is safe; omitting both leaves only authentication — an incomplete security model. The current API (`verify_omission_safety` per fragment) has no mechanism to detect this. A joint omission API is needed.

## Architectural gaps exposed by benchmark

1. **Semantic obligation matching** — lexical term overlap is insufficient. Need: does the retained set ANSWER the obligation, or just MENTION it? This is the decoder model gap.

2. **Joint omission safety** — current witness checks each candidate independently against the full retained set. Greedy selection that approves each omission individually can produce an unsafe final set. Need: verify the COMPOSITION of omissions, not each one in isolation.

3. **Concept-to-word grounding** — the obligation "security model" should match fragments containing "OAuth", "TLS", "rate limiting" even though those words don't appear in the obligation text. This requires learned concept-word mappings, not pattern matching.

## Scientific status

`CONTINUE RESEARCH`.

The omission safety benchmark is built, the baseline gap is measured (85.7% → 0.0% false negative rate), and the remaining architectural gaps are identified. The information residual approach works for structural constraints (authority, values, contradictions) but does not solve the semantic grounding problem (gap 1 and 3 above).

Known limitations:

- no NevIR neural benchmark has been run in this checkpoint;
- no ExcluIR benchmark has been run in this checkpoint;
- no Jev comparison has been independently reproduced;
- false positive rate remains 40% on safe cases due to lexical obligation matching;
- joint omission safety is not yet enforced at the API level;
- this branch must not be merged or marketed as a breakthrough until frozen promotion gates pass.

Decision: the information residual is a falsifiable, measurable advance. The 0% false negative rate on the frozen benchmark is a real result, not architecture. The next falsification target is the remaining false positives and the joint omission problem.
