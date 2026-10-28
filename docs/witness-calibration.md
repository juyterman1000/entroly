# WITNESS calibration: marginal scope and unavailable evidence

Calibration is an optional, caller-supplied policy input. WITNESS otherwise
uses its existing heuristic workload defaults. Neither path is a proof of
semantic correctness, source completeness, or safety of an individual request.
This change does not enable automatic calibration or remote model calls.

## Two different order-statistic contracts

`conformal_quantile(scores, alpha)` is an **upper coverage** helper for finite
scores in `[0,1]`. With `n` exchangeable calibration scores and one future score,
it uses rank `ceil((n+1)*(1-alpha))`. If that rank exceeds `n`, the result is the
known upper bound `1`, not the observed maximum. This supplies marginal coverage
of at least `1-alpha`, including the calibration randomness. Returning `1` for
an empty sample is only a trivial upper bound, not evidence for a PASS action.

`calibrate(...)` needs the opposite direction: a bound on the **strict lower
tail** of hallucination scores that escape stronger actions. It uses

```text
r = floor((n_hallucinated + 1) * (1 - alpha))
tau = 0 if r == 0, otherwise the r-th sorted hallucination score
```

For a fixed scorer, workload, labels and alpha, conditional exchangeability of
the hallucination-score population implies

```text
P(future_score < tau | future claim is hallucinated) <= 1 - alpha.
```

Ties are conservative because policy comparisons are strict. Alpha's decimal
representation is used for rank arithmetic so decimal boundaries do not move
because of floating-point subtraction. Inputs must be finite probabilities and
scores; Boolean labels/scores are not silently accepted as numeric evidence.
Invalid calibration inputs, thresholds and deployed scores raise `ValueError`
rather than manufacturing a valid risk-based action.

The existing alpha names, defaults and ordering are retained. They are
**escalation targets**, not confidence levels or direct false-pass budgets:

| Threshold | Default alpha | Marginal upper bound, conditional on a hallucinated claim |
| --- | ---: | --- |
| `tau_pass` | 0.80 | `P(PASS) <= 0.20` |
| `tau_hedge` | 0.50 | `P(PASS or HEDGE) <= 0.50` |
| `tau_warn` | 0.20 | `P(PASS or HEDGE or WARN) <= 0.80` |

The last row does **not** mean that 80% are suppressed. The second row controls
escape below WARN; the last controls escape below SUPPRESS. These are not
false-discovery rates among accepted claims, high-confidence population bounds
conditional on the observed calibration set, or optimal-retention guarantees.
Here the action events mean `ThresholdSet.action(score)` only. The full
`apply_witness_policy` pipeline has additional eligibility checks and overrides,
including a summarization SUPPRESS-to-WARN downgrade; this table does not certify
that pipeline. `CalibrationStore` is an explicit Python API: production policy
does not automatically load its profiles. Callers must deliberately supply the
returned thresholds and separately validate the resulting end-to-end policy.
Exchangeability and label correctness are caller assumptions, not facts proved
by a profile name, checksum, passing test or persisted metadata.

The rank argument is classical: [Kuchibhotla, Sections 2.4 and 3.3](https://arxiv.org/html/2005.06095v3).
This implementation does not claim a new conformal method or transfer general
risk-control theorems to an unmeasured LLM workflow.

## No positive-class evidence and persisted records

No hallucination-labelled samples means no learned positive-class guarantee.
The result is marked `unavailable_no_hallucinations` and yields all-SUPPRESS
thresholds `(0,0,0)`. Class-specific empirical rates with no observations are
`None`, not a measured zero. The normal default WITNESS policy is unchanged;
this behavior applies when a caller explicitly requests calibration.

New results use method `exchangeable-hallucination-lower-tail-v2` and status
`calibrated_marginal` when positive-class evidence exists. Old persisted results
lack this method and cannot silently reactivate thresholds from the incorrect
tail/rank procedure. A requested legacy or invalid stored profile fails closed
to `(0,0,0)`; recalibrate explicitly to activate a learned profile. A genuinely
missing profile continues to use the existing heuristic defaults without a
statistical guarantee. An unreadable or corrupt existing store is not treated
as an absent store, and is not rewritten during loading.

Store lookups refresh from disk. Updates reuse Entroly's local advisory lock,
reload inside that lock, and atomically replace a uniquely named temporary file
so cooperating writers cannot erase each other's profiles. The lock file stays
in place. Duplicate JSON keys, impossible rank-zero/equal-rank thresholds, and
profile-identity collisions are rejected. This is local cooperating-process
coordination, not protection against noncooperating edits or distributed storage.
Invalid canonical names stay quarantined even when later aliases look valid.
Enumeration excludes quarantined profiles and exposes no profiles after a
globally failed load, rather than providing a bypass around fail-closed lookup.
The data replacement and audit append are not a single crash-atomic transaction.

The metadata is not authentication and does not bind an external truth oracle,
scorer implementation or future distribution. Keep calibration and evaluation
data separate from scorer fitting and policy selection. Scope or distribution
changes require new evidence; passing a marginal test is not permission to
reuse a guarantee under arbitrary adaptive changes.

## Compatibility and rollback

No provider, package, native or release versions changed. Existing policy
defaults and explicit `ThresholdSet` overrides remain available. Recalibrated
profiles may retain fewer claims: the new procedure intentionally corrects the
old permissive boundary rather than preserving it. Loading does not migrate,
delete or overwrite existing calibration files.

Rollback is the inverse of the calibration/test/documentation patch. Reverting
also restores the old mathematical defect, so it is not a safe way to recover
the earlier permissive learned thresholds. User-owned calibration files remain
available for deliberate revalidation; no schema rewrite is required.
