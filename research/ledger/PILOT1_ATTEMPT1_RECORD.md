# Pilot-1 Attempt 1 — quota-invalidated. Record, verification, and Attempt-2 design.

This file is the research record for an experiment that did not produce an
answer. It is kept because the failure is informative and because splicing its
surviving rows into a later attempt would be scientific misconduct.

**Do not overwrite. Attempt 2 writes to a new directory.**

---

## 0. One attribution correction

The handoff describing this state says "Codex also created a local
research/analysis commit reported as `8a36acb1`" and speaks of Codex having
taken over orchestration. That is not what happened and the record should not
carry it.

`8a36acb1`, the frozen harness, the suite, the scorer and every analysis script
were authored in one continuous Claude Code session. **Codex appears in this
program only as Agent B inside the benchmark** — the successor model under test,
invoked through `codex exec`. It never orchestrated, never wrote analysis code
and never committed. Misrecording who authored the scoring code would undermine
exactly the integrity this ledger exists to protect.

---

## 1. Repository state, verified from git

```
branch        fix/empty-index-guidance-names-no-false-cause
HEAD          8a36acb14c5fe1ad7bfe9592b9033851e4ee4617
origin/main   e31f082fb351d48e628b2d519f5a6ee4936db405
working tree  clean
commits beyond origin/main   16  (15 from this program + 3022fcf4)
diff vs origin/main          201 files, +31,322 / −38
```

The handoff reported HEAD as `9876ee71`; that was one commit stale.

---

## 2. Frozen artifacts, every hash recomputed from disk

| artifact | expected | on disk | manifest-recorded |
|---|---|---|---|
| suite | `8bdffb59…d81982` | **MATCH** | MATCH |
| harness | `734f12d3…1b47c433` | **MATCH** | MATCH |
| scorer | `237ddf56…daaa0bdf` | **MATCH** | MATCH |
| native `.pyd` | `cf4d6e83…b53a28a` | **MATCH** | MATCH |

Nothing was regenerated. Attempt 2 is therefore the *same* experiment, not a new
regime.

---

## 3. Attempt 1 outcome

```
rows                    72  (24 tasks x 3 arms, suite ran to completion)
ELIGIBLE                15
ENVIRONMENT_FAILURE     57
matched eligible tasks   5   (p01-p05)
contamination            0
harness exceptions       0
```

**Quota failure confirmed from the traces, not inferred.** All 57 rows carry
the Codex error `"You've hit your usage limit"`, executed **zero** commands and
produced **zero** tokens. 19 of 24 tasks are dead in all three arms. The
ChatGPT account behind `codex exec` exhausted its quota during task 5.

These are `ENVIRONMENT_FAILURE`: a provider refusal is not an agent failure and
is not scored against any arm.

---

## 4. Research integrity: the usage guard was underspecified

The handoff asserts an aggregate-ratio rule was preregistered. **It was not.**
The frozen manifest — the only preregistered artifact — says exactly:

```
usage_guard: "noncached_usage_proxy (successor uncached input + output)
              must not rise >10%"
```

It names the *quantity* and is **silent on aggregation**. Neither median nor sum
was frozen. The earlier design discussion was equally silent. So this is a real
preregistration gap I introduced, not a later deviation by anyone, and the
honest remedy is to report both aggregations rather than to claim one was fixed.

Both are computed on the same matched eligible task set:

| | B1S | B3 | ratio | change | guard ≤ +10% |
|---|---|---|---|---|---|
| **Analysis C** — median | 38,565 | 30,395 | 0.7881 | **−21.2%** | PASS |
| **Analysis F** — sum | 174,333 | 166,862 | 0.9571 | **−4.3%** | PASS |

Per-task proxy, B1S `[29767, 39375, 19892, 38565, 46734]`,
B3 `[30386, 41530, 26358, 30395, 38193]`.

The two differ by a factor of five in apparent effect size (−21.2% vs −4.3%)
while agreeing on the verdict at n=5. That divergence is the whole reason the
aggregation must be fixed in advance: on a larger sample the two could easily
disagree on the guard itself.

**Both are preserved. Neither replaces the other.** For Attempt 2 the manifest
will specify the sum form as primary, with the median reported as sensitivity —
chosen because the sum is the quantity a payer actually incurs and is less
distorted by one cheap task. This specification is being added **after**
Attempt 1 and before Attempt 2, and that ordering is recorded here so it cannot
later look like a post-hoc choice.

---

## 5. Sidecar eligibility implementation — verified, not trusted

`analyse_pilot1.py` was tested against 17 constructed rows covering each stated
rule. All pass.

| case | class | correct |
|---|---|---|
| quota error / rate limit / missing interpreter / expired OAuth | `ENVIRONMENT_FAILURE` | ✓ |
| **agent's own code fails the tests** | `ELIGIBLE` | ✓ |
| agent succeeds | `ELIGIBLE` | ✓ |
| verifier result absent from collector row | `UNKNOWN` | ✓ |
| `verified_success` is None | `UNKNOWN` | ✓ |
| zero activity with no recorded error | `UNKNOWN` | ✓ |
| harness exception / missing checkpoint id / native hash mismatch | `HARNESS_FAILURE` | ✓ |
| reads Entroly source tree / prior pilot artifacts | `CONTAMINATED` | ✓ |
| `Get-Command python`, `site-packages` | `ELIGIBLE` (allowlisted) | ✓ |
| suite not run, no environment cause found | `ELIGIBLE` **and flagged** | ✓ |

The load-bearing one is row 2: a legitimate task failure stays eligible and
counts against its arm. Verifier evidence is read only from the collector
record, since Codex traces contain no trace of the harness's independent
verifier run.

---

## 6. B1N feasibility — still NO

Probed non-destructively, twice (the first attempt failed on argument form, not
auth, and was retried through stdin):

```
echo "..." | claude -p --output-format json --allowed-tools Read
  is_error       True
  terminal       api_error
  result         "Failed to authenticate: OAuth session expired and could not
                  be refreshed"
  tokens in/out  0 / 0
```

A working Claude Code UI session does **not** imply CLI authentication; these
are separate credential stores. No credential workaround was attempted.

```
B1N feasible = NO
```

Natural-predecessor-handoff superiority therefore remains **UNTESTED**, and
Pilot-1 cannot answer it even when it does run.

---

## 7. Codex quota — still exhausted

Minimal non-benchmark probe, `-s read-only`, no benchmark state touched:

```
quota_exhausted  True
agent_messages   0
usage            None
error            "You've hit your usage limit … try again at 1:07 PM."
```

**Attempt 2 must not start until this probe returns a nonzero token count.**

---

## 8. Mechanism audit of p01–p05 — understanding only, no verdict

Keyword proxies over the agent's own messages and command stream. Crude by
construction; used to understand the benchmark, never to score it.

| arm | mean commands | mean trap terms echoed | mean payload references |
|---|---|---|---|
| B0 | **11.8** | 1.4 | 1.4 |
| B1S | 20.8 | 2.0 | 2.0 |
| B3 | 20.2 | 2.0 | **2.6** |

Three observations, each stated as a hypothesis for the full run:

1. **B0 solved all five tasks with roughly 40% fewer commands than either
   handoff arm.** The handoff did not reduce successor work here; it appears to
   have added some, which is consistent with the successor spending turns
   reading and reconciling a handoff it did not need.
2. **B1S and B3 are indistinguishable on trap-term echo (2.0 each).** Whatever
   caused either arm to mention the rejected approach, structure was not it.
3. B3 references payload-only concepts slightly more often (2.6 vs 2.0). Weak
   and keyword-based; it is the only signal pointing at differential payload
   consumption and it needs blinded annotation, not a regex.

**Did these five tasks exercise continuity? Largely no.** Four of five are
`simple_local` — the stratum deliberately included to measure overhead, not
benefit. The strata where continuity should matter (`state_heavy`,
`failed_hypothesis`, `repository_wide`, `long_running`) were never reached. So
these rows are a measurement of the overhead floor, which is exactly what they
were designed to be, and no continuity conclusion follows from them.

---

## 9. Attempt-2 operational designs (design only; not implemented)

Neither changes B0/B1S/B3 prompts, task content, metrics or thresholds.

### 9a. Early quota circuit breaker

Attempt 1 burned 57 provider calls after the quota was already gone. The runner
had no reason to continue and no way to know it should stop.

```
after each arm completes:
    if the event stream contains a TERMINAL provider refusal
       (quota exhausted / plan limit / credit exhausted):
          write the row with class ENVIRONMENT_FAILURE
          write an INTERRUPTED marker carrying the provider message and
            the (task, arm) reached
          stop the run
          exit non-zero
```

Three constraints that make it safe:

* It inspects the stream only **after** a row is complete, so it cannot
  influence a successful row's prompt, tools, timing or scoring.
* It fires only on **terminal** refusals. Transient errors must not stop a run,
  or one blip discards a wave.
* It is runner-level. The arm definitions are untouched, so Attempt 2 stays
  comparable to Attempt 1.

Verification before implementing: run the breaker's predicate over all 72
Attempt-1 rows and confirm it fires on exactly the 57 environment failures and
none of the 15 eligible rows.

### 9b. Safe immutable results and resume

Attempt 1 wrote one `results.json` rewritten after every task, so an
interruption left no way to tell a complete row from a truncated one.

```
one file per (attempt, task, arm):
    pilot1_attempt2/rows/<task_id>.<arm>.json
written atomically: temp file + fsync + rename
each row carries: attempt_id, suite/harness/scorer/native hashes, a
    COMPLETE marker, and the row digest
resume:
    refuse to start unless every frozen hash matches the manifest
    skip a row only if it is COMPLETE *and* carries the SAME attempt_id
    never overwrite an existing complete row
```

**Resume is operational resilience, not permission to splice.** It may only
reuse rows from the same attempt. For the scientific comparison a complete
fresh Attempt 2 is preferred whenever quota allows, because provider state,
cache state, session effects, run order and quota-window position all differ
across attempts.

### 9c. Destination

```
research/ledger/pilot1/            Attempt 1, FROZEN, do not modify
research/ledger/pilot1_attempt2/   Attempt 2 (manifest, rows/, traces/)
```

No rows are ever moved or copied between them.

---

## 10. Status — unchanged

```
Production continuation payload value:   INSUFFICIENT EVIDENCE
Natural predecessor handoff superiority: UNTESTED
Continuity paid wedge:                   INSUFFICIENT EVIDENCE
Confidence:                              LOW
```

Not demonstrated: continuity benefit across the frozen 24-task suite;
state-heavy, failed-hypothesis, repository-wide or long-running performance;
superiority to a natural predecessor handoff; product-market fit; enterprise
ROI.

The next milestone is unchanged: run the entire frozen ContinuityStress-24
experiment under a healthy Codex quota, Claude → Codex, and obtain enough
matched valid tasks for the predeclared comparison to mean anything.
