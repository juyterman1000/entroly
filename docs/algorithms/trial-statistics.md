# Trial and benchmark statistics

What `entroly trial --report` and `bench/accuracy.py` can support at the
sample sizes they actually see, and where that support stops.

Implementation: `entroly/trial_statistics.py` (standard library only,
deterministic). The exact binomial bound is the one in
`entroly/binomial_stats.py`, shared with the granularity promotion gate and
`entroly decisions trial`.

## Why

`entroly trial --report` and `bench/accuracy.py` are where a saving or a
"quality retained" statement comes from.

The trial report summed tokens per arm and labelled three runs "directional",
without saying what three runs can and cannot support. The benchmark printed
one interval per arm; two separately computed intervals overlap even when one
arm is clearly worse, and say nothing about the difference between arms.

## What the trial report adds

When a trial is comparable (matched command, balanced arms, traffic gates
passed, three or more runs per arm) the report carries
`comparison.statistics`. Below that threshold it is `null`, for the same
reason the token difference is: a statistic about a comparison must not exist
while the status says the comparison is unsupported.

| Field | What it states | Rests on |
|---|---|---|
| `input_tokens`, `cost_usd` | Exact one-sided rank-sum p-value in each direction; finding is `supported`, `not-supported` or `contradicted` | Arm labels are exchangeable if the arm has no effect. No distributional assumption; ties handled exactly |
| `reduction_lower_bound` | Tokens (or dollars) saved per run, at least | A constant difference between arms |
| `relative_reduction_lower_bound` | Fraction saved per run, at least | A constant ratio between arms |
| `task_success` | Lower bound on the change in evaluated success rate, and whether a drop larger than the margin (10 points) is ruled out | Independent runs; exact per-arm bounds with the error budget split |
| `order` | Whether the arms were interleaved, run as blocks, or unknown | Default receipt names, which begin with the run's timestamp |
| `summary` | The same results as sentences that can be quoted as they stand | Nothing beyond the fields above |

Cost is compared only when every run recorded one. It is reported separately
from tokens because the two can disagree: provider-reported input tokens can
fall while the bill rises, for example when a changed prompt prefix stops the
provider's cache from hitting. In that case the report says `fewer_input_tokens:
supported` and `lower_cost: contradicted`.

Results are exact up to 60 runs per arm. Beyond that a tie-corrected normal
approximation is used and each affected figure is labelled `"exact": false`.

## What three runs support

Three clean runs per arm can support "fewer input tokens" at exactly
p = 0.05, the smallest value three runs per arm can produce; one overlapping
run loses it. Four runs per arm can reach p = 0.014.

Three successes out of three in each arm do **not** show that quality was
preserved: a drop of 71 percentage points is still compatible with them.
Ruling out a 10-point drop needs 36 runs per arm if every run succeeds, and
more if any fail. The report states this number so the cost of the claim is
visible before the runs are bought.

## The benchmark

`bench/accuracy.py` adds an exact paired comparison over the items both arms
answered: accuracy change, its lower bound, and whether a drop larger than
5 points is ruled out. Items where the arms agree carry no information about
the difference, so this is far tighter than comparing two separate intervals.
Example: 90/100 against 88/100 with the arms disagreeing on 4 items gives a
lower bound of about -8 points paired, against -15 points from the two
separate bounds.

## Where it stops

* Arms are chosen by the operator, not assigned at random. The statistics
  describe the runs; they do not separate the arm from run order, cache warmth
  or time of day. Alternate baseline and optimized runs; the report warns when
  each arm was run as one block. No causal or general savings claim follows.
* Each conclusion is tested on its own at 95%. They are not guaranteed
  jointly.
* The reduction bounds assume the arms differ by a constant amount (or ratio).
  The p-values do not.
* Task success is whatever the attached external evaluation reported. A run
  counts as a success only when it reports both `task_success` and
  `evidence_retained`.
* The benchmark comparison assumes its items are independent draws from the
  workload of interest. It has unit tests but has not been re-run here against
  a live model.
