# Evidence-adaptive context granularity

Status: implemented as an evidence-gated SDK primitive. It is not enabled as
an automatic proxy policy because no repository result yet shows that a
particular compressed representation is safe for a public workload.

## The research signal

The May 2026 *Compute Optimal Tokenization* paper studies language-model
training, not agent context management. Across 988 latent-tokenized and 320
subword-tokenized training runs, it reports that:

- bytes of training data are a more stable scaling unit than tokenizer tokens;
- validation loss has a non-monotonic relationship with compression rate;
- the estimated optimum changes with compute, language, tokenizer family, and
  task difficulty; and
- inference cost and downstream quality can prefer different granularities.

Source: [Compute Optimal Tokenization](https://co-tok.github.io/paper.pdf).

Those results do **not** prove that compressing an agent's prompt improves its
work. Entroly uses the paper as motivation for a narrower hypothesis: context
resolution should be learned per task/model/tokenizer scope from matched
outcomes instead of being fixed globally.

No code, coefficients, or fitted constants from the paper are used here. In
particular, Entroly does not hardcode a target loss or compression number.

## The invention: Paired Evidence Resolution Frontier

For one exact deployment scope, run the same task against the same source bytes
with:

1. a full-context baseline; and
2. one candidate representation, such as structural context or a recoverable
   reduced context.

Each paired observation records the source digest, both delivered-context
digests, actual token counts, versioned evaluator judgments, and catastrophic
failures. A candidate earns promotion only when all gates pass:

- enough paired trials exist;
- the one-sided lower bound on paired quality remains inside the configured
  non-inferiority margin;
- the Wilson upper bound on verifier regressions remains below policy;
- the full-context baseline succeeds often enough to be a valid reference;
- declared evaluator-family diversity meets policy on every paired trial;
- no catastrophic failure was observed when zero-catastrophe policy is active;
  and
- the candidate reduced actual delivered tokens by the required amount.

The frontier then selects the lowest-token candidate among the eligible arms.
If any required evidence is missing, full context remains selected.

Evidence is isolated by task class, model, tokenizer, evaluator protocol, and
baseline identity. A versioned workload ID is also mandatory, and candidate
arms compete only when their exact trial cohorts match. Results cannot silently
transfer between those scopes.

## SDK example

```python
from entroly import (
    EvaluatorVerdict,
    GranularityScope,
    PairedGranularityObservation,
    VerifiedGranularityFrontier,
)

scope = GranularityScope(
    task_class="code-repair",
    workload_id="heldout-repo-repairs-v1",
    model_id="provider/model-version",
    tokenizer_id="provider/tokenizer-version",
    evaluator_protocol="repo-tests-plus-grounding-v1",
)

frontier = VerifiedGranularityFrontier()
frontier.record(
    PairedGranularityObservation(
        scope=scope,
        trial_id="task-0001",
        candidate_id="structural-context-v1",
        source_sha256="...64 lowercase hex characters...",
        baseline_context_sha256="...64 lowercase hex characters...",
        candidate_context_sha256="...64 lowercase hex characters...",
        source_bytes=125_000,
        baseline_tokens=18_400,
        candidate_tokens=9_700,
        verdicts=(
            EvaluatorVerdict(
                evaluator_id="repo-test-harness@sha256:...",
                evaluator_family="deterministic-repo-tests",
                baseline_score=1.0,
                candidate_score=1.0,
                baseline_passed=True,
                candidate_passed=True,
            ),
        ),
    )
)

receipt = frontier.decide(scope)
```

The default policy requires 40 paired trials. Before that threshold, the
receipt returns `retain_baseline`. State can be atomically saved and restored
with `frontier.save(path)` and `VerifiedGranularityFrontier.load(path)`.

## Evidence boundary

The receipt is content-addressed and internally self-consistent, not
authenticated or omniscient. It records what observations the controller used
and how it applied policy. An external signature or transparency-log anchor is
required for adversarial tamper evidence. The receipt also cannot prove that a
harness ran the claimed model, that evaluator families are statistically
independent, or that a workload represents future traffic. Those remain
external evidence obligations and are stated in every decision receipt.
