# Continuity pilot prerequisites, evidence coverage, and the B3 falsification

Anchor: remote `e31f082f`. Local, unpushed: `c5c6b85c`, `a5a91edf`, `55e75c88`,
`2491bca8`, `9ba8d410`.

Two ledgers kept separate. **ENGINEERING REALITY** is measured on this commit.
**RESEARCH / MOAT** is hypothesis.

---

## 1. ENGINEERING REALITY — honest outcome coverage, measured

`research/experiments/honest_outcome_coverage.py` against the real local log
`~/.entroly/ravs/events.jsonl` (6,518,092 bytes, 9,738 lines, 1 malformed).

```
request records            4,798
outcome records            4,814
strength distribution      strong: 4,814   medium: 0   weak: 0
requests with an outcome   4,788  (99.79%)
```

### 1a. The denominator is not what it looks like, and the real one is unrecorded

The intended denominator is "optimize_context calls a cached PRISM observation
could exist for". **That number is recorded nowhere.** `optimize_context` writes
no `request` record; the only thing it can write to this log is a `trace` event,
and only when `_shadow_plan.decomposed_nodes > 0` (`server.py:1525`) — **1 such
record in 9,738 lines**.

The `request` records that exist come from `source: "hook"` (4,756) and
`source: "real_e2e"` (42); neither producer exists in this tree. And:

```
outcomes written within 10 ms of their matching request:  4,788 / 4,788
median delay 0.001s   p95 0.001s   max 0.001s
```

So **the log contains zero delayed outcomes.** The entire Hindsight Posterior
Correction premise — "RAVS honest outcomes arrive *later*, sometimes seconds,
sometimes minutes after the optimization call" — has no instance in 4,788
recorded pairs. These are synchronous tool-execution captures.

### 1b. The measured number that matters: 0 actionable outcomes of 4,814

```
HonestOutcomeCoverage, actionable:   0.0     (0 of 4,814 strong outcomes)
```

Cause: the log's vocabulary and the bridge's reward table do not intersect.

| in the log | count | `_OUTCOME_REWARD` expects |
|---|---|---|
| `test/pass` | 3,568 | `test_result/passed` |
| `lint/pass` | 838 | — no lint key at all |
| `build/pass` | 230 | — no build key |
| `typecheck/pass` | 139 | — no typecheck key |
| `format/pass` | 27 | — no format key |
| `test/unknown` | 12 | — |

Both halves of the key differ (`test` vs `test_result`, `pass` vs `passed`).
`honest_reward()` returns `None` for every one of the 4,814 events, so the bridge
never reaches PRISM.

**This is not a vocabulary question nobody has answered — the repository already
answers it twice, differently.** `ravs/router.py:583` accepts
`{"success","passed","accepted","pass"}` and event types
`("test","build","lint","typecheck","format","other","test_result","ci_result","command_exit")`.
So the RAVS **router** learns from all 4,814 events and gates model routing on
them, while the **OutcomeBridge** learns from zero. Two divergent outcome
vocabularies in one subsystem.

### 1c. Three independent reasons the honest-outcome loop has never fired

1. `optimize_context` never records a request, so there is no observation to
   correct against for the hook-captured requests.
2. No outcome in the log is delayed, so there is nothing for a *hindsight*
   correction to do.
3. No outcome in the log uses a key the bridge can read.

The temporal-credit defect fixed in `55e75c88` was therefore real in code and
**dormant in production on this log**. Scope stated exactly: *the live MCP
honest-outcome correction used the current EMA rather than the observation-time
baseline, which could invert temporal credit within a session. Fixed locally.*
It would have mattered the moment any of the three blockers above was lifted.

**Not fixed.** Aligning the vocabulary would switch on a learning loop that has
been dark, mutating PRISM weights from 4,814 historical events. That is a
behaviour change, not a bug fix, and it needs a decision rather than a quiet
patch.

### 1d. What this does to Arm B

Arm B (verified outcomes only) is **not** starved for lack of verified outcomes —
4,814 of them exist and 99.8% of tool executions carry one. It is starved because
nothing joins them to an optimization. That reframes the prerequisite: the
problem is the join, not the evidence supply.

---

## 2. ENGINEERING REALITY — crystallizer determinism, fixed

Commit `9ba8d410`.

**Rule chosen: per-cluster document frequency — keep tokens that recur across
member queries, drop the rarest, break ties lexically.**

Why, in the order alternatives were rejected:

| candidate | verdict |
|---|---|
| `sorted(tokens)[0]` / lexical only | deterministic but semantically wrong — a family about "zero-copy writes" permanently evicts its own subject |
| first-seen order | preserves whatever the family opened with; arbitrary once paraphrases drift |
| IDF | the textbook choice, but needs a corpus-wide document frequency this class does not have and should not start collecting |
| stable hash | deterministic and meaningless — same arbitrariness, just reproducible |
| **per-cluster df** | the local form of IDF, already derivable from the queries flowing through here — **chosen** |

**Invariant preserved:** the cap's own stated purpose. The existing comment says
the point is "keeping the cluster's identity recognizable"; a token appearing in
1 query out of 20 is the least identifying thing in the bag, so it is the correct
victim.

**Complexity added:** one dict increment per query token (amortized O(1)) plus an
O(M ≤ 33) scan per eviction, and one `dict[str,int]` per cluster — strictly
smaller than the unbounded `cluster.queries` list already held. No embeddings, no
model call.

Two incidental corrections fell out: the trim is now a `while` loop, because `if`
plus a single `pop` left the documented 32-token bound unenforced whenever a
query contributed several new tokens; and the count map is pruned deterministically
while always retaining resident tokens, since zeroing a resident token's count
would make it the next victim.

### Before / after, identical 48-observation sequence

| `PYTHONHASHSEED` | 0 | 1 | 42 | 12345 | 7919 | 104729 | unseeded ×4 |
|---|---|---|---|---|---|---|---|
| **before** — active clusters | 6 | 5 | 4 | 4 | 6 | 4 | 4,4,5,4 |
| **after** — active clusters | 4 | 4 | 4 | 4 | 4 | 4 | 4,4,4,4 |

After the fix all six compared properties are identical: cluster partition,
centroids, representatives, fragment recipes, candidate emission index, candidate
payloads. Validity gate asserted in both the script and the test — the first
version of this sweep passed with **zero events and every centroid stalled at
16**, which proves nothing; it is recorded as run 1, INVALID.

**Scope kept exact.** What was demonstrated pre-fix is that *cluster partition
and centroids are hash-seed dependent*. A **different candidate event being
emitted was not demonstrated** — the single event in that fixture fired before
the cap mattered and was identical across all ten runs. Fixed because
reproducible learning state is a requirement, not because a production failure
was observed.

Tests: `tests/test_crystallizer_determinism.py`, 6 cases including the
cross-process sweep, the retention rule asserted directly, the tie-break, cap
enforcement under multi-token arrival, the count-pruning invariant, and an AST
guard against reintroducing `set.pop` (a substring guard matched the module's own
docstring and was replaced).

---

## 3. ENGINEERING REALITY — MemoryOS: intended semantics and the real gap

### Intended semantics: **B, mixed memory with provenance** — not verified-only

Source evidence, not field names:

* `docs/memory-ecosystem.md:5` — Entroly decides "what should be remembered,
  recalled, suppressed, shared, verified, persisted, and sent under a token
  budget". Verification is listed as *one function among several*, not a
  precondition for storage.
* `docs/memory-ecosystem.md:58` and `:139` place verification in
  `witness.py` / the WITNESS gateway, applied to **generated answers**, not to
  memory admission.
* `memory_cli.py:42` ships a general `remember` taking `--importance`, `--tier`,
  `--source`, `--tags`, `--safety-policy`. A verified-only store would not expose
  caller-chosen importance and source.
* `tests/test_memory_os.py` contains **no** occurrence of `verified`,
  `provenance`, `trust` or `unverified` — the tested contract is storage, decay,
  budget and safety, not verification.

So arbitrary CLI writes are **not** a violation of the intended invariant.

### Therefore the gap is in the consumer, as hypothesised — and it is real

Traced end to end:

```
memory_fabric.recall(...)
  -> recalled.context.selected  (Memory objects)
     -> TaskDreamer._accept_evidence(content, source, kind="memory_os",
                                     metadata={id, tier, retention, score})
        -> capsule line:  "### memory_os - `<source>` - sha256 <...>"
```

| metadata | survives to task assembly? |
|---|---|
| `source` | **yes** — rendered in the capsule |
| `tier`, `retention`, `score` | yes — carried, not rendered |
| `tags` (incl. `"verified"`) | **NO — dropped at `_recall_memories`** |
| evidence strength / confidence | **no such field is passed or rendered** |

Consequences:

1. A verified outcome episode and an arbitrary remembered sentence render
   **identically** as `### memory_os - \`<source>\``. The only discriminator is
   the free-text `source`.
2. `source` is an **unprotected namespace**: `verified_outcome:` is just a string
   prefix a CLI caller can set.
3. `_refresh_memory` reloads on mtime, so external writes are picked up between
   tasks.
4. The mitigation is an instruction in the capsule — *"Treat recalled memory as
   a lead; verify it against current source before acting"* — not a gate.

**Verdict: defect, in the consumer.** `DEFECT — evidence strength is discarded
before task assembly`. Not fixed: per §19 it does not invalidate the continuity
experiment, which reads the Work Graph rather than MemoryOS.

---

## 4. ENGINEERING REALITY — minimum join key (design only)

Identifiers that exist today at each stage:

| stage | identifier present | persisted |
|---|---|---|
| optimize_context | `request_id` in the result dict | **no** |
| PRISM observe | none — `_n` only | no |
| crystallization | `cluster_id` (uuid4), `event_id` (uuid4) | no |
| SkillEngine candidate | `skill_id` | yes, skill dir |
| benchmark | `evidence_id` = sha256 of the evaluation record, plus `evaluation_scope` | yes |
| promotion | `skill_id`, `fitness_lower_bound` | yes |

The chain breaks in exactly two places: `request_id` is not persisted, and
`CrystallizationEvent` does not carry the request ids of the observations that
produced it.

**Minimum join, two fields, no new store:**

```
1. CrystallizationEvent.source_request_ids: list[str]
   The request_ids of the observations in the window that crossed the bound.
   Requires optimize_context to pass its request_id into crystallizer.observe().
   Answers: "which real observations led to this candidate?"

2. SkillSpec.origin_event_id: str
   The CrystallizationEvent.event_id the candidate was created from.
   Answers: "which independent evidence later evaluated it?" -- by composing
   with the evaluation record already keyed by evidence_id + evaluation_scope.
```

That is sufficient to compute candidate precision. Explicitly **not** proposed:
a global event bus, a provenance graph, or a third correlation key. Design only;
not implemented.

---

## 5. RESEARCH / MOAT — frozen A/B/C experiment

```
A  current    implicit selection geometry        -> candidate-authorizing accumulation
B  verified   strong/medium request-bound outcome -> candidate-authorizing accumulation
C  tiered     weak/proxy  -> exploration & scheduling only
              strong/medium verified -> candidate-authorizing posterior
Unknown outcome = CENSORED. Never negative, never positive.
```

**Primary metric — candidate precision**

```
candidates later passing caller-heldout evaluation
--------------------------------------------------
      candidates evaluated independently
```

Not candidate count. Secondary: useful discovery rate, time to candidate,
observations to candidate, false candidate rate, promotion rate, compute cost.

**Kill condition, frozen now:** if B or C improves candidate precision by less
than 10 absolute percentage points while requiring more than 3× the observations
to first candidate, kill the hypothesis and keep Arm A. Justification: the
stronger-evidence architecture costs a join key plus a schema change on a
persisted artifact; under a 10-point gain that is not repayable, and a 3×
slowdown in discovery on a loop that currently fires rarely would make it
effectively dark.

Blocked on §4 (the join) and now also on §1 (no outcome in the log is joinable
to an optimization). Both are bounded.

---

## 6. ENGINEERING REALITY — B3 falsification, run before spending anything

`research/experiments/b3_handoff_content.py`. Production-reachable calls only,
canonical constructors, no hand enrichment. A realistic interrupted task was
recorded: 2 modified files, 2 explicit decisions, 2 remaining-work items, a
context receipt with omitted-and-recoverable spans and a recovery handle, and an
execution chain whose outcome `state="failed"`, `verification_state="failed"`,
verification `verdict="failed"`.

Then every production read surface was searched for that content.

| surface | leaf fields | contains `"failed"` | contains the remaining-work text |
|---|---|---|---|
| `handoff()` | 21 | **no** | **no** |
| `resume()` | 69 | **no** | **no** |
| `context_scope()` | 29 | **no** | **no** |
| `continuation_proof()` | 23 | **no** | **no** |
| `summary()` | — | **no** | — |
| `unfinished()` | — | **no** | — |
| `snapshot()` | 12,733 B | yes | — |
| `export_state()` | 14,953 B | yes | — |

### Findings

**6a. The handoff receipt is an integrity artifact, not a knowledge artifact.**
Its 21 fields are `graph_commitment`, `workstream_id`, `from_agent`, `to_agent`,
and lists of `node_ids` / `edge_ids` / `evidence_ids` — content-addressed hashes.
`verify_handoff` returns `True`, which is a real and valuable property. But an
agent reading only the handoff receipt learns no fact about the work. My first
census reported "fill_rate 1.0" for it; that metric was misleading and is
corrected here — a list of hashes is non-empty and transfers nothing.

**6b. The failure verdict does not reach any continuation surface.** The graph
retains it (`snapshot`/`export_state` contain `"failed"`), but `resume()` files
it under `verification_ids: ["test:00aa…"]` with `failures: []` and
`failure_ids: []`. The projected evidence record carries
`kind: "test_result"`, `trust: "verified"`, `freshness: "current"` — and **no
verdict**. An agent B is told a verified, current test result exists. It is not
told the test failed. It would have to re-run the test to find out, which is
exactly the Reconstruction Tax the wedge is supposed to remove.

**6c. Recorded remaining work is unreachable.** `remaining_work` was recorded in
the observation's `task_hint`; `unfinished()[0]` has no `remaining_work` key at
all (its keys are `agent_ids, changed_paths, commit_ids, decision_ids,
evidence_ids, failure_ids, kind, label, node_id, status, symbol_ids, task_ids,
trust, updated_at_ms, verification_ids`), and `continuation_proof`'s
`outstanding_work_refs` came back `[]`.

**6d. What does transfer, and it is not nothing.** `resume()` carries the task
label, `status: "blocked"`, `trust: "verified"`, both changed paths, and both
decision **texts** verbatim — including *"budget must stay per-request; a global
default broke cache alignment"*. That is a real recorded constraint a prose
summary might soften, plus a verifiable graph commitment a summary cannot have.

### Why the paid pilot was not run this turn

The B3 arm as currently constructible **cannot tell agent B that the test
failed, and cannot tell it what work remains.** Running 20–30 two-vendor tasks
against that arm would spend real money measuring a crippled baseline, and a
weak result would be uninterpretable: it would not distinguish "continuity is not
a wedge" from "the projection drops the three most valuable facts". `codex` is on
PATH and `OPENAI_API_KEY` is set, so the blocker is not access.

Correct order: project `verdict`, `failures` and `outstanding_work_refs` onto the
continuation surfaces first — the data is already in the graph, so this is a
projection change, not new capability — then run the pilot. Harness design below
is frozen and unchanged by this finding.

---

## 7. Frozen continuity pilot design

Unchanged from the predeclared version except where §6 forces an explicit
precondition.

**Scale.** 24 tasks. Explicitly a **pilot**: enough to expose obvious failure
modes, not enough for statistical certainty. Effect size and viability only; no
p-value interpretation.

**Strata**, reported separately so easy tasks cannot dominate:

| stratum | n | why |
|---|---|---|
| simple / local | 4 | overhead floor |
| multi-file | 5 | |
| repository-wide | 4 | |
| state-heavy | 5 | where continuity should win |
| failed-hypothesis-heavy | 4 | the §6b case |
| long-running | 2 | |

**Interruption.** Deterministic, not time-based: agent A is stopped after its
*k*-th successful tool call, *k* fixed per task in advance. The repository is then
committed to a scratch branch and its SHA recorded, so every arm resumes from a
byte-identical tree.

**Arms.** `B0` repository + task statement · `B1` agent-A-written handoff summary
(the real competitor; cheap to produce) · `B2` native cross-provider resume —
**recorded as NOT APPLICABLE for Claude→Codex, because no shared native
cross-provider resume mechanism exists**; not invented · `B3` Entroly
continuation state, production-reachable only.

**Ground truth allowed in every arm:** repository state, diffs, tool results,
tests, explicitly recorded decisions, explicitly recorded unresolved work,
verified claims, receipts, recorded execution events. **Not allowed:** hidden
chain-of-thought, unrecorded intent. Unknown stays unknown.

**Metrics per task:** verified final success · post-handoff input/output tokens ·
time to first correct action · files reread · duplicate reads / tool calls /
edits · wrong assumptions · reverted edits · recovery calls · human intervention
· wall time · provider cost. Plus four objective booleans: did B correctly
identify what was already done, what remained, the preserved constraints, and
did it avoid the rejected hypothesis.

**Reconstruction Tax**, components kept separate — a single scalar hides the
behaviour:

```
rediscovery input tokens
+ duplicate repository reads
+ duplicate tool executions
+ duplicate edits
+ failed actions caused by lost prior state
```

Not monetized.

**Primary metrics:** (1) verified completion rate; (2) post-handoff provider cost
per verified completion.

**Predeclared kill criteria** — the proposed thresholds are adopted unchanged,
because they are the right shape: B1 is nearly free to produce, so a tool
requiring a runtime and an index must clear measurement noise by a margin.
Entroly must show at least one of

```
>= 20% lower median post-handoff reconstruction tokens
>= 10 percentage-point improvement in verified continuation success
```

against the stronger of B1/B2, **without** a >10% increase in total post-handoff
provider cost.

**Benchmark provenance recorded with every artifact:** `entroly_version`,
runtime implementation path, source commit, local patch commits, model, provider,
task id, repository SHA at checkpoint, handoff id, baseline arm, timestamp.
`implementation_path` is non-negotiable — an editable install silently pointing
at a different checkout already invalidated an earlier round of measurements.

---

## 8. Efficiency pilot design (not run)

S2 and S3 only. `E1` cache-aware baseline with provider prompt caching **on**
versus `E2` Entroly. Measure cost per verified success, verified success, cached
vs uncached input tokens, retries, latency. Kill: <15% improvement in cost per
verified success against the cache-aware baseline at quality non-inferiority →
weak first wedge. Not started; continuity is first and §6 is its gate.

---

## 9. Business interpretation

```
Continuity:
evidence = one zero-cost pre-pilot measurement of what the Entroly arm can
           actually carry (research/ledger/b3_handoff_content.json). No agent
           run, no cost measurement.
effect   = unmeasured. The measurement that exists is NEGATIVE about the current
           implementation: of four continuation surfaces, none projects the
           failure verdict, none projects recorded remaining work, and the
           handoff receipt is hashes only. What does transfer is the task label,
           blocked status, changed paths, verbatim decision texts and a
           verifiable graph commitment.
paid-wedge status = insufficient evidence

Efficiency:
evidence = none beyond the four structural risks already recorded (provider
           prompt caching, already-small contexts, MAX_FILES_CONSIDERED = 12,
           and a learning loop measured at 0 actionable outcomes).
effect   = unmeasured.
paid-wedge status = insufficient evidence
```

The honest position after this turn is that continuity is still the better bet
*and* that its current implementation would fail its own pilot for a fixable
projection reason. That is a better place to be than a weak pilot result of
unknown cause.
