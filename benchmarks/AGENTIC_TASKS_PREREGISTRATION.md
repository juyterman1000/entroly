# Preregistration — Experiment 0: Agent-Task Evidence Bridge

**Status:** protocol commitment, locked at the commit that introduces this
document, before the task harness produces any result. Criteria edits after the
first full run require a declared protocol revision (v2), mirroring
`benchmarks/model_recovery_protocol_v*.json` practice.

- **Harness:** `benchmarks/coding_tasks.py` (modes exist; task source and
  wiring to the real proxy paths are the work this document gates)
- **Result artifact:** `benchmarks/results/agentic_tasks_v1.json` (+ dated
  report under `benchmarks/results/`)

## 1. Question

Every committed Entroly headline result is single-shot extractive
(`benchmarks/results/compression_frontier.md`, scope limits: "SQuAD v2 contexts
test extractive answer retention, not full agent behavior"). This experiment
measures the claim the product actually makes: **does Entroly compression
preserve (or improve) multi-turn agent task success at materially lower token
cost?**

## 2. Task source

Tasks are mined from real repository histories, not hand-written:

1. Select public repos with test suites (plus this repo's own history).
2. Find commits that (a) modify source and test files, (b) whose test suite
   passes at the commit and fails with the source change reverted.
3. A task = repo snapshot with the fix reverted; the agent must make the
   failing test pass. The test is the oracle — deterministic ground truth.
4. Contamination control: prefer commits authored after the evaluated model's
   training cutoff; record commit dates per task; report results split by
   pre/post-cutoff.

Target n ≥ 400 mined tasks. Dev/holdout split by **repository** (no repo
contributes to both), 50/50.

## 3. Arms (paired — same task, same model, same seed policy)

| Arm | Description |
|---|---|
| RAW | full context, no Entroly |
| COMPRESS | Entroly compression only (QCCR/optimize path), no recovery loop |
| CLOSED-LOOP | compression + verification + omitted-span retrieval + retry once (`entroly/compression_verification_loop.py`) |

Fixed model and decoding parameters across arms (Entroly never modifies
request generation parameters). Per-task wall-clock and token caps identical
across arms.

## 4. Metrics

- task success (oracle test passes), per arm
- input tokens, output tokens, estimated cost per task
- paired success differences: exact McNemar test
- token savings: paired bootstrap 95% CI
- success-per-dollar frontier across arms
- CLOSED-LOOP internals: retry rate, spans retrieved, verifier flag precision

## 5. Preregistered criteria (locked)

| ID | Criterion |
|---|---|
| **N1** (non-inferiority) | COMPRESS task success is within 3 percentage points of RAW on the holdout split (paired), at ≥ 30% median input-token reduction |
| **D1** (dominance) | CLOSED-LOOP is strictly above COMPRESS on the holdout success-per-dollar frontier: success ≥ COMPRESS and cost/success < COMPRESS, or success > COMPRESS at ≤ equal cost |
| **F1** (falsification) | if RAW's success on the holdout differs from its dev success by > 10pp, the task mix is unstable and no arm comparison is claimed |
| **B1** (budget) | total benchmark spend is capped and recorded in the artifact; exceeding the cap voids the run (prevents silent cherry-picking by re-runs) |

Power note (honest): with ~10% discordant pairs, n = 300 detects ≈ 6pp at 80%
power (exact McNemar); n ≥ 400 is the target for margins near 3pp. If mining
yields fewer tasks, the non-inferiority margin must be widened *before* the
run and recorded here as v2 — not adjusted afterward.

## 6. Interpretation commitments

- **N1 ∧ D1** → the first task-level frontier claim is licensed, phrased
  exactly: "On n preregistered repository tasks, closed-loop Entroly matched
  raw-context task success within Xpp at Y% fewer input tokens (paired,
  holdout, p=Z)."
- **¬N1 (COMPRESS loses > 3pp)** → publish the negative result; redirect the
  program toward retrieval recall (what evidence was dropped) rather than
  tighter compression. This outcome is informative, not a failure of the
  program.
- **N1 ∧ ¬D1** → the closed loop's verify-retrieve-retry adds cost without
  measurable benefit at task level; the loop stays opt-in and undocumented as
  a win.

## 7. Ordering constraint

Experiment 1 (`benchmarks/VERIFIER_ABLATION_PREREGISTRATION.md`) gates the
CLOSED-LOOP arm's design: if EICV cannot detect compression-induced evidence
gaps at span level, the retrieval trigger must be redesigned around
selection-time receipts before this experiment runs.
