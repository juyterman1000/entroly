# Learning systems, coordination, determinism, and the business benchmark

Anchor: remote `e31f082fb351d48e628b2d519f5a6ee4936db405`.
Local, unpushed: `c5c6b85c`, `a5a91edf`, `55e75c88`.

Two ledgers are kept separate below. **ENGINEERING REALITY** is what the code
does on this commit. **RESEARCH / MOAT** is hypothesis. Nothing moves from the
second to the first without a measurement recorded here.

---

## 1. ENGINEERING REALITY — OutcomeBridge consumer graph

Traced by reading every reference, then confirming the live/dead split at
runtime.

```
 MCP tool                    server.py
 record_test_result     ─┐
 record_command_exit    ─┤
 record_ci_result       ─┼──> _record_honest()  ──> AppendOnlyEventLog (persist)
 record_edit_outcome    ─┘         │                 ravs/events.jsonl
                                   ├──> OutcomeBridge.on_honest_outcome()  LIVE
                                   │        └──> OnlinePrism._alphas  (in memory)
                                   │                  └──> engine.set_weights()
                                   └──> TaskDreamer.remember_verified_outcome()
                                            └──> MemoryOS  (persist, memory.json)

 proxy.py:3573  cache_observation          ─┐
 proxy.py:4021  verification_result        ─┼── UNREACHABLE
 proxy.py:4737  recovery_event             ─┘
```

**The three proxy call sites are dead.** Each is guarded by
`hasattr(self.engine, "_outcome_bridge")`, and `_outcome_bridge` is assigned
nowhere in the tree. Verified at runtime on a constructed engine:

```
hasattr _outcome_bridge: False
has _online_prism:       True
[a for a in dir(e) if 'bridge' in a.lower()] -> []
```

So `verification_result` and `recovery_event` — both in `_OUTCOME_REWARD`, both
`strength="strong"` — never reach PRISM. The bridge's own module docstring says
the engine calls it; the engine does not.

Answers to the eight questions:

| # | Question | Answer |
|---|---|---|
| 1 | Callers of `on_honest_outcome` | `server._record_honest` (live); 3 proxy sites (dead) |
| 2 | Production entry points | MCP STDIO server only. Not the HTTP proxy, not the SDK |
| 3 | Event types that reach it | Exactly 4: `test_result`, `command_exit`, `ci_result`, `edit_outcome` — the only 4 `_record_honest` call sites. **Unreachable** despite being in `_OUTCOME_REWARD`: `verification_result` and `recovery_event` (dead proxy sites), `user_acceptance` (**no MCP tool records it**), `retry_event`, `topic_change`, `escalation_event`, `compiler_exit`. 9 of 13 mapped event types have no caller |
| 4 | Strengths that can mutate PRISM | `strong` (1.0), `medium` (0.5). `weak` → 0.0 → `honest_reward()` returns `None` before any state is touched |
| 5 | Mutates anything besides OnlinePrism | No. Only `_prism._alphas`, plus its own counters. `weights_at_time` is stored and never read |
| 6 | Tests encoding current-EMA behaviour | None. All 12 HPC tests cache and resolve with no intervening `observe()`, so the live EMA and the observation baseline coincide. They pass unchanged after the fix |
| 7 | Persisted artifacts depending on `delta_advantage` | None. It appears only in the live MCP response and a benchmark print. `PrismState` / `load_state` have **zero production callers**, so the posterior is process-lifetime only |
| 8 | Shadow evaluation | `ShadowRunner` does not use the bridge. Only a docstring mentions it |

### The defect and the fix

`on_honest_outcome` computed the honest advantage against
`prism._reward_ema` **read at correction time**. Outcomes are delayed by design,
so unrelated requests move the EMA in between.

Identical verified success (`command_exit`/`success`, reward 0.90), identical
observation (baseline 0.40, implicit advantage +0.20), varying only intervening
traffic:

| EMA at outcome | `delta_advantage` | `w_semantic` α delta | direction |
|---|---|---|---|
| 0.20 | +0.5000 | +0.197990 | reinforce |
| 0.40 | +0.3000 | +0.118794 | reinforce |
| 0.75 | **−0.0500** | **−0.019799** | **penalise** |

`eta` was constant at 0.56568 across the three, so the entire spread is the
baseline choice. The comment above the line already asserted the correct
semantics, so the code contradicted its own stated contract.

**Temporal invariant, verified against PRISM rather than assumed.**
`OnlinePrism.observe` computes `advantage = reward - self._reward_ema` *before*
updating the EMA (`online_learner.py:198-202`), and `engine.py:1794` captures
`pre_baseline` for exactly that reason. So per-request baselines are PRISM's own
semantics, not merely the tidier algebra.

No field was added. `implicit_advantage` was defined as
`implicit_reward - baseline`, so the cached pair determines the baseline:
`baseline_at_observation = obs.implicit_reward - obs.implicit_advantage`.
Both inputs are rounded to 4 dp at `engine.py:1883-1884`, giving ≤2e-4 recovery
error against an unbounded ±0.5 error before — not enough to justify a schema
change. `baseline_at_observation` is now reported in the diagnostic dict so a
surprising correction is traceable.

Commit `55e75c88`. 18 new tests (T1–T8 plus a source-level guard against
re-reading the live EMA); **15 of 18 fail against the previous code**, and the
3 that pass pre-fix are the ones that must be unaffected (weak self-report,
unknown request, duplicate delivery).

### Severity, stated accurately

The corrupted posterior does reach live selection within a session via
`engine.py:1873 set_weights()`. It does **not** survive a restart, because
nothing persists `PrismState`. Classification: **RESEARCH CORRECTNESS** —
it corrupts a learning signal, not user data or receipts.

---

## 2. ENGINEERING REALITY — the learning matrix, 8/8

"Observable" is what the system can actually see. "Authority" is the strongest
consequence it can cause **by itself**, with no further gate.

| # | System | Observable | Interpretation | Mutation | Persistence | Correctable later? | Evidence strength | Maximum authority | Entry point |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **OnlinePrism** `online_learner.py` | `compute_implicit_reward(selected_count, total_fragments, tokens_used, query_present, token_budget)` | **selection geometry only** — budget fill and selectivity. Never reads the query text or any task result | Dirichlet `_alphas`, floored at 0.1 | **None.** `state()`/`load_state()` have no production callers | Yes — that is what OutcomeBridge is for | **E-proxy**: structural, no outcome | Changes live ranking weights after n≥3 via `set_weights()` | every `optimize_context()` |
| 2 | **OutcomeBridge / RAVS** `ravs/outcome_bridge.py` | `(event_type, value, strength)` for one `request_id` | externally produced result; `weak → 0.0` confidence, commented *"never correct from self-report"* | corrective Dirichlet update on the same `_alphas` | event itself appended to `ravs/events.jsonl`; the correction is not persisted | No — one-shot, cache entry popped | **E-verified** for strong, **E-behavioural** for medium | same as (1): live ranking weights | 5 MCP record_* tools |
| 3 | **RewardCrystallizer** `reward_crystallizer.py` | the **same implicit reward** as (1), plus query tokens and fragment ids | sustained-high-reward query family, Hoeffding-LCB vs an external baseline | in-memory clusters; emits `CrystallizationEvent` | none (process-lifetime) | n/a — emission is one-way | **E-proxy** (inherits (1)'s observable) | **authorizes a skill candidate** | `optimize_context()` via `engine.py:1836` |
| 4 | **SkillEngine** `skill_engine.py` | a candidate plus an evaluation record | pass-rate with Wilson bounds at `DECISION_CONFIDENCE` | writes skill dir, `spec.status` | yes — skill dirs on disk | yes — re-benchmark | **`development`** or **`caller_holdout`**, recorded explicitly as `evaluation_scope` | **promotion to production, but only on `caller_holdout`** (`skill_engine.py:1423`) | `promote_or_prune()` |
| 5 | **EvolutionDaemon** `evolution_daemon.py` | clustered failed queries → skill gaps | missing capability | creates + benchmarks + asks for promotion | skill dirs, stats | yes | gap evidence is **E-proxy**; its benchmark is **`development`** | **candidate creation only** (see below) | background thread |
| 6 | **belief coupling** `vault.py` + `proxy.py` | `/outcome` success/failure for one request's injected `claim_ids` | those beliefs helped or did not | Bayesian confidence on vault beliefs; enqueue for reverification | **yes — `vault/beliefs/`** | yes — reverification | caller-asserted at the proxy boundary; now **request-scoped** (`a5a91edf`) | durable belief confidence | proxy `/outcome` |
| 7 | **EpistemicRouter / FlowOrchestrator** | `flow_orchestrator.py:122`: `result.status == "completed" and len(result.beliefs_used) > 0` | **execution-shape / flow-completion proxy** — not verified task success | per-flow success lists; `_self_tune()` every 10 outcomes | only if a `ComponentFeedbackBus` is passed | no mechanism | **E-shape**: weakest of all | adjusts routing thresholds | `execute_flow` MCP tool **only** |
| 8 | **MemoryOS / TaskDream** `task_dream.py` | one of exactly 4 pairs: `test_result/passed`, `command_exit/success`, `ci_result/passed`, `edit_outcome/accepted` | externally verified success, bound to the active request | appends an episode to MemoryOS | **yes — `memory.json`, the most durable surface here** | no removal path found | **E-verified**, allowlisted | durable cross-session memory recalled into future task capsules | `_record_honest` |

### 2a. The distinctions that must stay visible

* **selection geometry ≠ task success.** Systems 1 and 3 observe *only* geometry.
  System 3 nonetheless authorizes a skill candidate.
* **flow completed + beliefs used ≠ verified success.** System 7, confirmed at
  `flow_orchestrator.py:122`. It is reachable only from the explicit
  `execute_flow` MCP tool — **ordinary `optimize_context` does not invoke it**,
  which bounds how often it tunes anything.
* **caller-heldout ≠ development.** System 4 enforces this; system 5 is
  therefore capped.
* **verified result ≠ self-report.** Systems 2 and 8 both enforce it, by
  different mechanisms (`_STRENGTH_CONFIDENCE["weak"] = 0.0`; a 4-pair
  allowlist). The legacy `record_outcome(success: bool)` is tagged
  `agent_self_report` / `include_in_default_training=False` and reaches neither.

### 2b. EvolutionDaemon, stated honestly

```
trigger             background thread; clustered failed queries
gap source          repeated query misses per entity  (E-proxy)
synthesis type      structural template first ($0), then SkillEngine template
benchmark evidence  benchmark_skill(skill_id) with NO validation_cases
                    -> evaluation_scope == "development"
promotion authority NONE. promote_or_prune() returns action="kept",
                    status="testing" for anything that is not "caller_holdout"
                    (skill_engine.py:1423)
persistence         skill directory + spec status
future effect       a candidate visible to the operator; no routing change
```

```
autonomous candidate generation:   YES
autonomous production promotion:   NO
```

This must not be described as self-evolution. The held-out gate is intact and
must stay intact.

### 2c. MemoryOS authority boundary — one real gap

The write path from outcomes is tightly gated: 4-pair allowlist, non-empty
request-bound task, **and** `request_id` must equal the active optimization's
(`server.py:1940`). Episodes are labelled `source="verified_outcome:<rid>"`,
`tags=["verified","task-outcome",<event_type>]`, `safety_policy="block"`.

But:

1. The store is a plain file at `$ENTROLY_MEMORY`. `entroly-memory remember
   <text>` (`memory_cli.py:42`) writes **the same default-resolved path** with
   arbitrary content and a caller-chosen `source`.
2. `TaskDreamer._refresh_memory` reloads on mtime change, so external writes are
   picked up between tasks.
3. `_recall_memories` (`task_dream.py:462`) recalls by relevance and **does not
   filter on the `verified` tag**.

So weak or self-reported outcomes cannot enter *through the MCP outcome tools*,
but the durable layer is a relevance store with provenance *labels that nothing
enforces at read time*. The mitigation is an instruction in the capsule
("Treat recalled memory as a lead; verify it against current source"), not a
gate. Recorded, not fixed — the CLI path is operator-invoked, which is
legitimate; the missing piece is a read-time authority filter.

---

## 3. ENGINEERING REALITY — coordination map

| Surface | Carries signal? | Evidence semantics? | Request identity? | Persists? | Controls authority? |
|---|---|---|---|---|---|
| RAVS event log `ravs/events.py` | yes | **yes** — `strength`, `include_in_default_training`, `STRONG`/`WEAK_OUTCOME_TYPES` | **yes** — `request_id` | yes, `events.jsonl` | yes — gates PRISM correction and memory promotion |
| OutcomeBridge | yes | inherits RAVS | yes | no | yes (PRISM) |
| `ComponentFeedbackBus` `autotune.py` | yes — `(component, metric, value, params)` | no | **no** | yes | no |
| `FeedbackJournal` `autotune.py` | yes | no | **no** | yes | no |
| Work Graph `work_graph.py` | yes | partial (`verified` appears) | **no** | yes | no |
| Vault artifacts `vault.py` | yes | `confidence` + `sources` | **no** — joined via `claim_id` | yes | yes (belief confidence) |
| `SessionReceiptChain` `session_intelligence.py` | integrity only | no | **no** | yes | no |
| Context Receipts `context_receipts/models.py` | selection provenance | no | **no** | yes | no |
| governance audit `governance/audit.py` | yes | `verified` | **no** (`trace_id`) | yes | yes (policy) |
| `ValueTracker` | spend/savings | no | **no** | yes | yes — gates the evolution budget |
| request ids | — | — | proxy: `request_attribution` (new); MCP: `request_id` in `optimize_context` | no | — |

### Conclusion: **B — several partial substrates exist.**

Not A, and the reason is specific rather than impressionistic: of eleven
surfaces, **exactly one** (the RAVS event log) carries *both* request identity
and evidence semantics. Four others control real authority — vault confidence,
governance policy, the evolution budget, skill promotion — and **none of them
records request identity or evidence strength**, so no automated join to RAVS is
possible. Three correlation keys coexist without a mapping: `request_id`
(RAVS, proxy, MCP), `claim_id` (vault), `trace_id` (governance).

Not C either: `request_id` genuinely flows MCP → RAVS → PRISM → MemoryOS, and
`claim_id` genuinely flows proxy → vault. Two real spines exist; they do not
meet.

**Consequence for the research question.** "Entroly learns from verified
outcomes" is true of the RAVS spine and false of the rest. Any claim about
cross-component learning needs a join key that does not currently exist. That
is a concrete, bounded engineering target — not an architecture rewrite.

---

## 4. ENGINEERING REALITY — crystallizer determinism

`research/experiments/crystallizer_determinism.py`,
`research/ledger/crystallizer_determinism.json`.

**Run 1: INVALID.** Every seed agreed, zero events fired, every centroid capped
at 16. The fixture could not reach the mechanism: with a 4-token shared core,
adding 3 new tokens drops Jaccard to 4/19 = 0.21, so the family *split* long
before the 32-token cap. A sweep in which the mechanism never fires certifies
nothing. Recorded rather than deleted.

Run 2 sizes the fixture against the gate: with a shared core of `C` tokens and a
centroid of size `M`, reuse requires `C/(M+F) ≥ 0.25`, so the cap at `M=32`
with `F=1` needs `C ≥ 9`. Nine core tokens plus one fresh token per query, with
a low-reward noise family interleaved to supply the `ext_n ≥ 5` external
baseline. Validity gate now asserted in the script:

```
centroid_cap_exercised: true   max_centroid_size: 32
events_emitted:         1 per seed
```

Identical 48-observation sequence, 6 fixed seeds + 4 unseeded processes:

| `PYTHONHASHSEED` | active clusters |
|---|---|
| 0 | **6** |
| 1 | **5** |
| 42 | 4 |
| 12345 | 4 |
| 7919 | **6** |
| 104729 | 4 |
| unseeded ×4 | 4, 4, **5**, 4 |

`centroids_identical: false`, `partition_identical: false`, and `stats` differ
from the reference for 8 of 9 comparisons.

**Located, as predicted by reading before measuring** —
`reward_crystallizer.py:445`:

```python
cl.centroid_tokens |= qtokens
if len(cl.centroid_tokens) > 32:
    cl.centroid_tokens.pop()      # set.pop() over a set[str]
```

CPython salts `str` hashing per process, so `set.pop()` evicts a
seed-dependent element. Candidates cleared by inspection: `_top_fragments`
(stable sort over a dict built from `tuple(sorted(...))`), `_common_terms`
(sorted by `(count, token)` — total order), `_jaccard` (set algebra → float),
`min(self._clusters.values())` (ties fall back to insertion order).

A second finding the code's own comment gets wrong. It calls the drop
"cheap, unbiased", but the evicted token can be a **core identity token**:
seed 0's largest centroid lost `receipt`, seed 42's lost `dedup`. The cap is
documented as bounding drift; it also erodes cluster identity.

**What is and is not established.** Cluster partition and centroid content are
hash-seed dependent — demonstrated. Partition is the input to every future
emission decision, so this is decision-*adjacent*. But the one event emitted in
this fixture fired at `i=12`, before the cap mattered, and was **identical
across all ten runs**. I have therefore **not** demonstrated that an emitted
`CrystallizationEvent` payload changes with the hash seed. Stating it as
"crystallization output is nondeterministic" would overclaim.

Classification: **ENGINEERING — reproducibility**. No fix applied; it needs a
deterministic eviction rule (lowest-frequency, or oldest-touched token), which
is a behaviour change to a learning component and should not ride along with a
correctness fix.

---

## 5. RESEARCH / MOAT — honest-outcome crystallization

### Is candidate discovery authorized by evidence too weak for its authority?

**Yes, and the asymmetry is now precisely located.** From the matrix: system 3
(crystallizer) authorizes a skill candidate from the *same* observable as system
1 — `compute_implicit_reward`, which never sees the query result. Meanwhile
system 2 exists specifically because that observable is a proxy, and system 4
refuses to promote on anything short of `caller_holdout`.

So the pipeline already distrusts implicit reward at the promotion gate and
trusts it at the discovery gate.

### Severity, not exaggerated

Because `promote_or_prune` hard-requires `caller_holdout`, weak evidence
**cannot** execute unsafe code. The research issue is **candidate precision**:
weak evidence may be generating candidates that cannot survive held-out
evaluation, which wastes the evolution budget and the operator's review
attention. That is an efficiency and signal-to-noise question, not a safety one.

### Arms, with outcomes censored not assumed

```
Arm A (current)  implicit selection reward        -> candidate-authorizing accumulation
Arm B            verified outcomes only          -> candidate-authorizing accumulation
Arm C            weak proxy  -> exploration/scheduling
                 strong verified -> candidate-authorizing accumulation
```

Primary metric: **candidate precision** = (candidates whose later
`caller_holdout` evaluation clears `PROMOTE_THRESHOLD`) / (candidates emitted).
Secondary: candidates emitted per 1,000 observations (recall proxy), and
observations-to-first-true-positive.

Do not assume C wins. Arm B's risk is specific and must be measured first, not
asserted: verified outcomes arrive only when an agent voluntarily calls one of
the 4 live `record_*` tools, so B's sample rate is whatever that call rate turns
out to be. **That rate is currently unmeasured** — it is the first quantity to
read off `ravs/events.jsonl` versus the `optimize_context` count, and if it is
low, B starves. **A request with no outcome is `CENSORED`, never success** — the
estimator must handle censoring explicitly or B and C will both look better than
they are.

Blocked on two prerequisites, both now identified rather than assumed:
1. The determinism defect above. Candidate precision cannot be measured while
   cluster partition varies with the hash seed.
2. No `claim_id`/`request_id` join exists between a crystallization event and
   the later held-out evaluation (section 3). Without it, "this candidate came
   from this evidence" is not reconstructible.

Still worth testing — **after** those two.

---

## 6. RESEARCH / MOAT — the business benchmark

Predeclared before any run. Thresholds justified here, not chosen afterward.

### 6.1 Metric

```
CostPerVerifiedSuccess =
    (provider spend + retry spend + measurable Entroly overhead)
    / verified successful tasks
```

"Verified" means a RAVS-strong signal: tests passed, CI passed, or exit 0 on a
predeclared check. Not an agent self-report. Unknown outcome is `CENSORED` and
excluded from the numerator's success count but **not** from spend.

### 6.2 Wedge A — efficiency

Baselines: `E0` raw agent · `E1` cache-aware raw agent (prompt-prefix stable,
provider caching on) · `E2` Entroly. A competitor is added only when its setup
is reproducible and the comparison is fair; absent that, it is not reported.

Strata, predeclared, reported **per stratum** so easy tasks cannot dominate:

| | Workload | Purpose |
|---|---|---|
| S0 | already fits the window | measures **no-op overhead only** |
| S1 | medium, single-file / local context | |
| S2 | multi-file context pressure | |
| S3 | repository-wide dependency task | |
| S4 | multi-turn | |
| S5 | long-running with context recovery | |

Kill criteria:

* **S2–S5:** ≥15% reduction in `CostPerVerifiedSuccess`, with verified success
  statistically **non-inferior** at a predeclared margin of 5 percentage points
  (one-sided, α = 0.05). 15% is chosen because it is roughly the point at which
  a per-seat tool survives a procurement comparison against "just use a bigger
  context window"; below that, the saving is inside the noise of model pricing
  changes.
* **S0–S1:** total task cost and p95 latency must not be **materially worse**,
  fixed now as ≤5% cost and ≤150 ms added p95. A context tool that taxes the
  easy majority of tasks loses on aggregate regardless of S3 wins.
* **If `E1` erases most of the `E0 → E2` gap: efficiency wedge = weak.** Say so
  plainly. Provider prompt caching is the single largest threat to this wedge.

### 6.3 Wedge B — continuity

Checkpoints: deterministic stop for agent A, then resume under
`C0` same agent / new session · `C1` different instance, same model
· `C2` different model/provider. At least two vendor pairs so the result is not
one vendor's resume quirk; the harness takes the pair as a parameter.

Baselines, deliberately strong — no strawman:

```
B0  repository state only
B1  agent-generated handoff summary   <- the real competitor
B2  native vendor/session resume      <- where available
B3  Entroly recorded-state continuation
```

Metrics: verified final success · post-handoff tokens · time to first correct
action · duplicate reads / tool calls / edits · wrong assumptions ·
context-fault & recovery calls · human interventions · total completion time ·
provider cost.

```
ReconstructionTax = rediscovery tokens
                  + duplicate operations
                  + failed continuation cost
                  + human recovery effort
```

Human effort is counted as **interventions and wall-clock**, physical units
only. No dollar conversion until the hourly-rate assumption is written down
separately.

Kill criteria: ≥20% reduction in post-handoff reconstruction tokens **and/or**
a materially better verified completion rate at not-materially-higher total
cost, measured against **the strongest of `B1`/`B2`** — not against `B0`.
20% is chosen because `B1` is nearly free to produce: a tool that needs a
runtime and an index must beat "ask the model to write a summary" by more than
measurement noise to justify its existence.

**If `B2` matches `B3`: continuity is not a strong paid wedge.** Write that down
and stop. Do not re-baseline against `B0` afterward.

### 6.4 Why continuity currently looks more interesting — not a conclusion

Efficiency carries four open risks, all now evidenced on this commit:

1. provider prompt caching (`E1` may absorb the gap)
2. already-small contexts (S0/S1 are pure overhead)
3. `MAX_FILES_CONSIDERED = 12` in `entroly-qccr/src/lib.rs:67` — a breadth
   ceiling independent of token budget, with no comment, no citation and no
   override hook
4. the temporal-credit defect above: "Entroly learns to select better over
   time" was unsupported until `55e75c88`, and is unmeasured after it

Continuity is less exposed to all four, but carries its own burden: it must beat
a good handoff summary, must carry only state that was actually recorded, and
must not hallucinate missing intent.

### 6.5 Runtime provenance — benchmark reproducibility first

Minimum fields, each justified by a failure already seen in this program:

| Field | Why | Verdict |
|---|---|---|
| `entroly_version` | version surfaces have gone stale repeatedly | **include** |
| `implementation_path` | an editable install silently pointed at a *different checkout*; every hook-mediated measurement before the repair was invalid | **include — the highest-value field** |
| `native_runtime_version` | native vs Python fallback changes selection semantics | **include**, with the fallback flag |
| `source_commit_if_available` | distinguishes dirty tree from tagged build | **include**, nullable |
| `model` / `provider` | the independent variable in `C1`/`C2` | **include** |
| `task_id` / `handoff_id` | the only way to join a continuity pair | **include** |
| `plugin_version` | already implied by `entroly_version` for bundled plugins | **omit** unless they diverge |

Design only. Not a generalized enterprise provenance system.

---

## 7. Business interpretation

```
Efficiency
  Evidence strength:    WEAK.  No E0/E1/E2 measurement exists on this commit.
  Primary risk:         provider prompt caching erases the gap (E1 ~= E2),
                        compounded by the 12-file breadth cap.
  Falsifying experiment: S2/S3 against E1 with caching ON. If cost per verified
                        success improves <15%, the wedge is dead as stated.

Continuity
  Evidence strength:    PRECONDITION MET, HYPOTHESIS UNTESTED.
                        Audited tree == executed runtime; non-ASCII prompts
                        round-trip to an exact receipt SHA. No continuity
                        measurement has been run.
  Primary risk:         B1 (agent-written handoff summary) and B2 (native
                        vendor resume) are nearly free and may match B3.
  Falsifying experiment: C2 (Claude -> Codex) with B1 and B2 both present. If
                        reconstruction tokens do not drop >=20% against the
                        better of them, continuity is not a paid wedge.

Most likely first paid wedge:  CONTINUITY
Reason:  it is the only one of the two not directly exposed to provider
         caching, already-small contexts, the 12-file cap, or the PRISM credit
         defect -- and its prerequisite (a trustworthy measured runtime) is now
         satisfied while efficiency's prerequisites are not.
Confidence: LOW. This ranks the risks, not the outcomes. Neither wedge has a
         single measurement behind it.
```
