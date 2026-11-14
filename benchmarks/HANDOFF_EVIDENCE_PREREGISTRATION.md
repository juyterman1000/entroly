# Preregistration: authenticated handoffs and current verification evidence

**Status:** criteria locked at the commit that introduces this document,
before the harness or the implementation under test exists. Any later edit to
sections 3–9 is a protocol revision and must be declared as v2, with the
reason, in the results.

- **Harness (to be written):** `benchmarks/handoff_evidence.py`
- **Result artifact:** `benchmarks/results/handoff_evidence.json`
- **Implementation under test (to be written):** an evidence-applicability
  check and a host-assertion handoff acceptance on the `entroly-work` CLI
- **Repository state when locked:** `main` at `7e8d2bc5`, Entroly 1.0.90

## 1. Question

A developer switches coding agents midway through a refactor. The successor
receives a claim that verification passed. Should it rerun the verification,
reuse the result, or stop?

Can Entroly tell the successor which of these is correct, with fewer redundant
verification runs than the strongest runnable baseline and without accepting a
stale result? And can Entroly refuse a successor whose authority it cannot
verify?

## 2. Hypothesis (falsifiable)

**H1 (safety).** Binding a verification result to the inputs the verification
process actually read gives zero stale acceptances on the challenge suite in
section 4. Those inputs are code, data and configuration files, the listings of
directories the import system and test runner consulted, the environment
variables read, installed distributions and the interpreter. The baselines
that do not bind configuration, environment or patch-level dependency inputs
accept at least one stale result on the same suite.

**H2 (efficiency).** On transitions that do not touch those inputs, the result
is reused. The method runs at least 20% fewer redundant verification
operations than the strongest runnable baseline.

**H3 (authority).** A successor gains write authority only through a valid
assertion from a pinned host key. The assertion must name the expected issuer,
audience, successor, action scope, workspace, task, exact handoff digest and
one attempt, inside its validity window. Concurrent acceptances produce one
winner, an interrupted acceptance is recoverable, and the predecessor's
writes are refused afterwards.

H2 is expected to tie against a test-level cache on this suite, and to lose to
it on test executions in the real-transition replay (section 7). A tie ends
the claim of algorithmic novelty for H2. The work is then judged on H1 and H3
as an integration benefit, not as a better cache.

## 3. Comparators (decision policies applied to the same frozen transitions)

Each policy receives the same predecessor artifacts and decides one of:
**reuse** (do not rerun), **revalidate** (rerun the verification command), or
**stop** (refuse to proceed). All policies are deterministic code; no language
model is involved. This measures the decision handed to a successor, not an
agent's behaviour.

| Id | Policy | Decision rule |
|---|---|---|
| M0 | Trust the handoff summary (native continuation without revalidation) | reuse whenever the predecessor's summary states the command passed; revalidate only when there is no such statement |
| M1 | Structured handoff, always revalidate (also the full-rediscovery control for verification) | always revalidate |
| M2 | Current Entroly on `main` | `entroly-work handoffs --to-agent <successor>`: reuse iff the handoff entry reports `current=true`, else revalidate |
| M3 | pytest-testmon 2.3.0 (dependency-aware test cache) | run the command with `--testmon` on the predecessor's `.testmondata`. **reuse** iff zero tests execute, else **revalidate** (counted at test level) |
| M4 | Implementation under test | reuse / revalidate / unknown from the applicability check (unknown is executed as revalidate; a refused acceptance is stop) |

M3 runs inside the same environment the successor uses. For M3 the
"verification operations" metric counts test executions. The pytest
collection it always performs is reported separately as time.

## 4. Frozen challenge suite (synthetic fixture, built by the harness)

### 4.1 Fixture

A pure-Python package `invoice` with a src layout and these files:

- `pyproject.toml` with `[tool.pytest.ini_options] pythonpath=["src"]`,
  `testpaths=["tests"]` and `addopts="-q"`.
- `src/invoice/pricing.py`, the refactor target. It uses `tax.rate()` and
  `ratelib.round_money`, and calls a helper that emits a `DeprecationWarning`.
- `src/invoice/tax.py`, which reads `src/invoice/data/rates.json`.
- `src/invoice/report.py`, which the verification does not import.
- `tests/conftest.py`, `tests/test_pricing.py` (the verification target) and
  `tests/test_report.py` (not part of the verification command).
- `docs/notes.md`.

The dependency is a local distribution, `ratelib`. Version 1.0.0 rounds half
up. Versions 1.0.1 and 1.1.0 round half to even, which changes the expected
totals.

The environment is a Python 3.12 venv with `pytest==9.2.0`,
`pytest-testmon==2.3.0`, `coverage==7.17.3` and `ratelib==1.0.0`. A second
venv on Python 3.13 has the same pins.

The predecessor applies a benign refactor: it extracts `_subtotal()` in
`pricing.py` without changing behaviour. It then runs the verification command
`python -m pytest tests/test_pricing.py`, which passes, records the result and
hands off. The successor is a separate process started with `env -i`, holding
`PATH`, `HOME` and `LANG`, under a different agent id.

### 4.2 Transitions and ground truth (fixed before any run)

"Required" means rerunning is the only correct choice, because an input the
verification depends on changed or the evidence is missing. "Consequential"
means a rerun's outcome differs from the recorded pass.

| Case | Transition applied after handoff | Required | Consequential |
|---|---|---|---|
| A1 | `tax.py` logic change | revalidate | yes (fails) |
| A2 | `rates.json` value change (non-Python input) | revalidate | yes |
| A3 | comment-only edit in `pricing.py` | revalidate | no |
| A4 | new package `src/invoice/tax/__init__.py` shadows `tax.py` (no existing file edited) | revalidate | yes |
| A5 | `tests/conftest.py` fixture data change | revalidate | yes |
| B1 | `pyproject.toml` addopts gains `-W error::DeprecationWarning` | revalidate | yes |
| B2 | successor environment sets `PYTEST_ADDOPTS=-W error::DeprecationWarning` (no repository change) | revalidate | yes |
| B3 | successor environment has `ratelib==1.0.1` (patch release; no repository change) | revalidate | yes |
| B4 | successor environment has `ratelib==1.1.0` (minor release) | revalidate | yes |
| B5 | successor uses the Python 3.13 venv | revalidate | no |
| C1 | `docs/notes.md` edited | reuse allowed | no |
| C2 | `src/invoice/report.py` edited | reuse allowed | no |
| C3 | `tests/test_report.py` edited | reuse allowed | no |
| C4 | new file `scripts/release.sh` | reuse allowed | no |
| C5 | the refactor is committed (new HEAD, identical content) | reuse allowed | no |
| C6 | successor environment sets an unrelated variable `EDITOR=vi` | reuse allowed | no |
| D1 | the predecessor's evidence record is deleted before acceptance | revalidate | no |
| D2 | the evidence record is altered after the offer was created | revalidate | no |
| D3 | the predecessor recorded no evidence; only a summary says "tests pass" | revalidate | no |

Under "reuse allowed", a rerun is correct but redundant. In a D case, reusing
the result accepts evidence that is missing or unauthenticated. That is
counted as a stale acceptance.

## 5. Outcomes and scoring

The ground truth used for scoring is the table above. Each consequential case
is also checked by an actual rerun recorded in the result artifact. If a
"consequential: yes" case does not fail on rerun, that is reported as a
fixture defect and the case is excluded from the consequential count, not
relabelled.

Primary, per policy, over all 19 cases:

1. **Stale evidence accepted.** The policy chose reuse in a case marked
   revalidate.
2. **Missed required code or evidence.** A stale acceptance in A2, A4, B1–B5
   or D1–D3. These cases test inputs other than an edited Python file the
   verification already imported, and a missing record.
3. **Correct next action.** Revalidate in revalidate cases, reuse or
   revalidate in reuse-allowed cases, never stop unless authority is refused.

Secondary:

4. **Redundant verification operations.** Reruns in C1–C6 for M0, M1, M2 and
   M4 (command level), and test executions in C1–C6 for M3. The verification
   command has a fixed number of tests, which is reported, so the two units
   convert exactly.
5. **False stops.** Stop or unknown in reuse-allowed cases.
6. **Elapsed time** for decision and rerun, median of 3 repetitions.

## 6. Authority conformance suite (synthetic inputs, mock host key)

The mock host key is an Ed25519 key that the harness generates. It stands in
for a trusted host and is not a real identity provider or model provider. In
each case below, the acceptance must be refused and the authority state must
be unchanged, except where stated otherwise:

1. unknown issuer
2. a signature that verifies only under a key that is not pinned
3. wrong audience
4. successor id differs from the handoff's `to_agent`
5. task differs
6. workspace differs
7. handoff digest differs (an altered offer)
8. expired assertion
9. assertion not yet valid
10. lifetime longer than the configured maximum
11. replayed attempt id after a successful acceptance
12. altered signature byte
13. missing claim
14. unknown extra claim
15. action scope lacks accept
16. a recovery acknowledgement is outstanding
17. Eight concurrent acceptances of one offer by eight valid assertions
    (distinct attempts). Exactly one succeeds and seven are refused as
    superseded.
18. Interruption before the commit point leaves no authority change. A retry
    with the same assertion after the commit point is idempotent and returns
    the same epoch.
19. After acceptance, the predecessor's fenced write (`handoff` for that
    workstream) is refused. So is the holder's write after its assertion
    expires.
20. A valid acceptance succeeds and the authority epoch increases by one.

Each case passes or fails, and all 20 must pass.

## 7. Held-out real transitions (secondary)

This is a repository the implementation was not developed against: `click` at
`2247b35` and the 40 first-parent commits before it, taken in order. The
verification command is `python -m pytest tests/test_options.py -q
-p no:cacheprovider`. At each commit `i`, the predecessor records evidence and
testmon data, the tree moves to `i+1`, and every policy decides. Ground truth
is the actual rerun at `i+1`.

**Consequential stale** means a policy reuses a result while the rerun's
pass/fail set differs. The replay reports reruns and test executions per
policy, and consequential stale counts. If the repository cannot be built or
fetched, this section is reported as not run with the reason. It is not
replaced by another repository after results are seen.

## 8. Thresholds

**Ship the safety claim (H1)** only if M4 accepts zero stale results in all 19
cases, and each comparator's stale acceptances are reported with denominators.

**Ship the efficiency claim (H2)** only if M4 has at least 20% fewer redundant
verification operations than the strongest runnable baseline, which is M3,
with no loss in correct next actions against M1 (noninferiority margin: zero
cases). A tie or a loss ends the novelty claim for H2. A comparison against
the zero-stale baselines is reported as well. It is labelled as such and never
substituted for the M3 comparison.

**Ship the authority claim (H3)** only if all 20 conformance cases pass.

**Kill.** Any stale acceptance by M4 in the challenge suite kills H1 for this
design until fixed and re-run under a declared v2. A failed conformance case
blocks the PR.

## 9. Uncertainty and reporting

Every attempted case is reported, including fixture defects and harness
errors. Zero observed stale acceptances is a finite-suite result. It is
reported with its denominator and a one-sided 95% Clopper–Pearson upper bound.
For example, 0 of 13 revalidation cases bounds the rate at 20.6%.

There is one synthetic repository and one real repository. Results are
reported per repository and never pooled into a single rate. With one
repository per kind, no claim generalises beyond these two. The click replay
is reported as clustered by repository: n = 1 cluster. A moat requires
held-out transitions across more repositories and runtimes. This
preregistration cannot establish one.

Labels used in the results:

- **fixture:** the synthetic repository and its transitions.
- **mock host:** the harness-generated Ed25519 key.
- **real entry point:** the installed `entroly-work` console script, run as a
  subprocess.
- **not run:** authenticated provider operation. No provider credentials or
  host controls are available in this environment.
