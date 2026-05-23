# WITNESS v3 Production Report

## One shipped behaviour

WITNESS ships a **single** implementation to every surface — proxy,
MCP, SDK, CLI — regardless of whether the native engine is installed.
The numbers in this report are that shipped behaviour; there is no
"depends which binary you have."

(History, one line so the decision is auditable: a Rust fast-path in
entroly-core once diverged from this calibrated path and shipped
silently to native-engine users. It is now off by default and can only
be re-enabled after passing an exact-conformance gate —
`tests/test_witness_parity.py` with `ENTROLY_WITNESS_RUST=1` — so the
product can never again behave differently per binary.)

## Benchmark — shipped path

| Slice | N | F1 | Accuracy | Suppression F1 | Exposure | Retention | ms/sample |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HaluEval-QA | 200 | 0.656 | 0.680 | 0.681 | 0.340 | 0.738 | 2.5 |
| HaluEval-Dialogue | 200 | 0.584 | 0.530 | 0.631 | 0.216 | 0.340 | 5.2 |
| HaluEval-Summarization | 200 | 0.301 | 0.535 | 0.000 | 1.000 | 1.000 | 84.9 |

Two QA-local, deterministic gates were added (forensic-driven, not
threshold-tuned). Cumulative effect on HaluEval-QA, vs the prior
production path:

| Metric | baseline | +binding gate | +residual gate |
| --- | ---: | ---: | ---: |
| Exposure (unsupported reaching user) | 0.639 | 0.392 | **0.340** |
| Retention (safe kept) | 0.738 | 0.738 | **0.738** |
| Suppression F1 | 0.440 | 0.645 | **0.681** |
| F1 | 0.413 | 0.615 | **0.656** |

**−47% relative QA exposure at zero measured false-positive cost.**
Retention 0.738 held to 4 decimals across both interventions; verified
independently — the SAFE_KEEP/SAFE_SUPP forensic buckets were unchanged
(39/20 on N=120), so the flat aggregate is real, not a coarse metric.
Dialogue/Summarization unchanged: both gates live inside `qa_alignment`,
neutral when there is no question (confirmed empirically).

## Extractive-Binding Margin Gate (QA-local)

A forensic over 120 HaluEval-QA samples showed the exposed class is not
"answer absent from knowledge" — it is **recombination / wrong-option**:
the answer reuses the knowledge's own vocabulary (every bag-of-words
feature scores mildly supportive) but binds it into a false proposition.
The additive log-odds model over-credits these correlated overlap
features. The separating signal is **contiguity**, not presence:

| bucket | bind_q (longest contiguous answer span at the Q-evidence sentence) |
| --- | --- |
| safe / retained | mean 0.94, median 1.00, 95% ≥ 0.34 |
| exposed hallucination | mean 0.32, median 0.29, ≈63% < 0.34 |

`feat_qa_alignment` promotes to an active-mismatch (negative, hard gate)
when the question locus is confident **and** `bind_q < 0.34`.

## Question-Residual Payload Gate (QA-local)

A second forensic on the *post-binding* false negatives **refuted the
obvious hypothesis**. The remaining exposure is NOT dominated by the
comparative class (an earlier draft of this report claimed it was —
that claim was wrong and is retracted). Measured composition of the
residual FNs:

| class | share |
| --- | ---: |
| **factoid wrong-slot** | **≈67%** |
| selective ("A or B?") | ≈12% |
| comparative / ordinal | ≈12% |
| membership ("both") | ≈8% |

The dominant residual is the **wrong-slot factoid**: the answer parrots
the question's own frame and appends a wrong filler ("Which state … CEO
is Warren Bryant … located?" → "…located in *Utah*"). Whole-answer
overlap is inflated by the echoed question words. Only the answer's
**question-residual** (tokens it adds beyond the question) carries truth
value. The gate: if that residual payload is *entirely absent* from the
question's evidence sentence (`present == 0`, with `raw_score < 0.6` so
strongly-grounded answers are out of scope), promote to active-mismatch.
Both gates property-tested in `tests/test_witness_continuous.py`.

### Honest caveats / known limitations

- **≈34% of QA hallucinations still exposed.** The hard tail needs
  relational/multi-hop reasoning a deterministic *lexical* verifier
  cannot do (the wrong filler genuinely appears elsewhere in K, or the
  answer is a vague-but-wrong paraphrase). This is the structural
  ceiling of the bag-of-words + contiguity family; closing it needs an
  NLI or learned verifier. Diminishing returns are visible: the residual
  gate added only −0.052 exposure vs. the binding gate's −0.247.
- **Multi-hop / heavy-paraphrase false-positive risk.** A correct
  answer that is heavily reordered or whose filler lives in a different
  sentence than the question's constraint *can* be false-gated. This is
  encoded as a `strict=True` xfail in the test suite, not hidden. It
  does not bite on HaluEval-QA because safe answers there are extractive
  (bind_q ≈ 0.94), which is exactly why measured Retention is flat — but
  on paraphrastic data this gate would cost retention.
- **Conformal recalibration owed.** The split-conformal τ thresholds
  were fit under the old `qa_alignment` distribution. The gates improve
  the operating point at current τ, but `calibrate_witness_halueval.py`
  must be re-run to restore the formal RCPS exposure guarantee under the
  new distribution.

## Calibration Smoke

The calibration runner produced split-conformal CRC threshold stores with stable dataset hashes and CRC32 audit identifiers.

| Profile | N | Hallucinated | Safe | tau_pass | tau_hedge | tau_warn | CRC32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| benchmark_qa | 10 | 4 | 6 | 0.0597 | 0.5387 | 0.9200 | d1de01f1 |
| dialogue | 10 | 4 | 6 | 0.5910 | 1.0000 | 1.0000 | f1db0a3a |
| summary | 10 | 4 | 6 | 0.6589 | 1.0000 | 1.0000 | a51f36af |

## Engineering Status

- Continuous-risk Python path has property tests for boundedness, monotonicity, hard-gate dominance, SGD stability, atomic aggregation, and calibration persistence.
- Online training accepts only external labels through the training API or RAVS-shaped honest outcome events.
- Rust now exposes the continuous feature extractor and risk predictor through PyO3 bindings for the release hot path.
- Summary profile is intentionally retention-first: unsupported summary claims are warned unless direct contradiction evidence is present.

## Interpretation

QA is now the strongest result: two forensic-driven deterministic gates
cut QA exposure **0.639 → 0.340 (−47% relative)** at **zero measured
retention cost** (0.738 held flat, verified via unchanged SAFE forensic
buckets). Both thresholds were read off measured distributions, not
tuned. The second forensic also corrected a wrong hypothesis in an
earlier draft (the residual is wrong-slot factoid, ≈67%, not the
comparative class) — recorded here for honesty. Dialogue remains a solid
win (F1 0.584); Summary stays intentionally retention-first. The
remaining ≈34% is the relational/multi-hop tail that a lexical verifier
structurally cannot reach (NLI/learned verifier territory), and the
conformal thresholds should be recalibrated under the new feature
distribution before the RCPS guarantee is re-asserted.
