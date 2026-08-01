# Neutral Competitive Evidence Contract

Entroly must not claim superiority from README numbers, unpinned `latest`
packages, different evaluators, different prompts, or aggregate token totals
that mix compressor output with provider billing usage.

A competitive run is **claim-ready** only when every system is evaluated under
one frozen experiment contract and every artifact is independently
identifiable.

## Required identity fields

For Entroly, raw context, and each competitor, record:

- exact package version and version-command output;
- resolved executable path and SHA-256;
- SHA-256 of the installed wheel, npm package, source archive, container image,
  or explicitly enumerated artifact tree;
- complete command and timeout;
- canonical input JSONL SHA-256;
- evaluator/model/prompt/seed/budget contract SHA-256;
- environment and platform fingerprints;
- output SHA-256, return code, timeout state, stdout, stderr, and latency.

A version string by itself is insufficient because mutable environments can
contain locally modified or repacked artifacts.

## Comparability gates

The harness must refuse aggregation when any of these differ unexpectedly:

1. canonical input digest;
2. evaluator and experiment-contract digest;
3. model, prompt template, seed set, and budget schedule;
4. hardware/platform fingerprint for latency claims;
5. environment fingerprint for same-machine comparisons;
6. dataset split or calibration/holdout membership;
7. success, timeout, or error-accounting policy.

A failed, missing, timed-out, or identity-mismatched runner remains visible in
the report. It must never be silently removed from the denominator.

## Token categories

Report these separately for every sample:

- raw context tokens;
- selected tokens before final emission;
- emitted context tokens;
- provider input tokens;
- provider output tokens;
- provider total tokens.

Compression savings use emitted context. Provider totals are billing and model
usage observations; they are not relabelled as compressor output.

## Quality and safety outcomes

At minimum, report:

- paired baseline/treatment task success;
- gains and regressions;
- exact McNemar significance;
- paired bootstrap confidence intervals;
- answer/evidence presence before selection, before emission, and after
  emission when a task oracle permits it;
- accepted-error versus coverage for assurance-gated modes;
- bypass, expansion, degraded, uncertain, and hard-budget-uncertified rates;
- p50/p95 latency and receipt overhead.

Expected answers or task oracles may be used only after selection for scoring
and failure attribution. They must never enter ranking or compression inputs.

## Leadership rule

Entroly may claim a win only on a clearly named regime where it is no worse on
all declared primary dimensions and strictly better on at least one, with the
full per-sample evidence and immutable run manifest published.

A result on one dataset, model family, context size, or workload type is not a
universal win. Conflicting regimes must be reported rather than averaged away.

## Current status

The repository contains the evidence schema, paired statistics, command runner,
assurance receipts, calibration gates, and selector-overhead harness needed to
produce this evidence. A neutral, exact-artifact Entroly–Headroom–LeanCTX
holdout has not yet been completed, so no overall superiority claim is made.
