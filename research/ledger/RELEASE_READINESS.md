# Release readiness audit

Scope: the 6 product commits in this branch, audited against the repository's
actual CI gates rather than a remembered list. Track A (release) is independent
of Track B (the continuity business experiment), which remains blocked on
provider quota and is **not** a release gate.

---

## 0. Repository state, verified live

```
branch        fix/empty-index-guidance-names-no-false-cause
HEAD          15a71950
origin/main   e31f082fb351d48e628b2d519f5a6ee4936db405
working tree  clean
```

`git fetch` showed `origin/main` carrying one commit this branch lacked:
`e31f082f "fix(mcp): stop blaming the working directory for an empty index"`.
`git log --left-right --cherry-pick` reduces that to **0** — it is the
squash-merge of this branch's own `3022fcf4`. Same patch, different object. Not
a real divergence, and confirmed by a trial rebase that git auto-skipped as
"previously applied".

---

## 1. Commit classification, from diffs rather than subjects

| commit | purpose | product | test | research |
|---|---|---|---|---|
| `c5c6b85c` | strict UTF-8 machine-protocol ingestion | 3 | 1 | 0 |
| `a5a91edf` | request-scoped belief attribution | 2 | 1 | 0 |
| `55e75c88` | observation-time PRISM credit | 1 | 1 | 0 |
| `9ba8d410` | deterministic reward crystallization | 1 | 1 | 2 |
| `f0719965` | continuation projection (verdict + outstanding work) | 3 | 0 | 2 |
| `e2eb4e93` | work_resume workstream selection | 1 | 2 | 2 |
| 13 others | research ledger, pilot harnesses, analysis | 0 | 1 | 1–13 |

Product surface touched, 11 files / +782 / −31:

```
entroly/agent_activation.py      entroly/request_attribution.py
entroly/cli.py                   entroly/reward_crystallizer.py
entroly/proxy.py                 entroly/work_graph_mcp.py
entroly/ravs/capture.py          entroly/work_graph_store.py
entroly/ravs/outcome_bridge.py   entroly-engine/src/work_graph.rs
                                 entroly-engine/src/engine_contracts.rs
```

---

## 2. Gate matrix

Gates were read out of `.github/workflows/` rather than assumed.

| Gate | Result | Evidence |
|---|---|---|
| working tree | PASS | clean |
| live origin/main | PASS | no real divergence (cherry-pick = 0) |
| Python full suite | PASS | see §3 |
| Rust tests — engine | PASS | `cargo test --all` green |
| Rust tests — core | PASS | 118 passed, 5 ignored |
| Rust tests — qccr | PASS | 29 passed |
| clippy — engine (CI-gated) | PASS | `--all-targets -D warnings`, 0 |
| clippy — core (CI-gated) | PASS | 0 |
| clippy — qccr (NOT CI-gated) | **2 errors, pre-existing** | see §5 |
| `cargo fmt --check` — qccr (CI-gated) | PASS | 0 diffs |
| `cargo fmt --check` — engine (NOT gated) | 105 diffs, pre-existing | see §5 |
| ruff `entroly/` (CI-gated, pinned 0.12.9) | PASS | local ruff 0.12.9, "All checks passed" |
| version staleness | PASS | "No version declaration is older than master 1.0.85" (603 historical refs allowed) |
| package build | PASS | sdist + wheel, `twine check` PASSED both |
| clean install | PASS | fresh venv, wheel only |
| CLI smoke | PASS | `entroly --version` → 1.0.85, `--help` lists 60 subcommands |
| MCP smoke | PASS | all 5 console-script modules import |
| native provenance | PASS | built == loaded, `cf4d6e83…b53a28a` |
| UTF-8 regression | PASS | from the installed wheel |
| request attribution | PASS | from the installed wheel |
| OutcomeBridge | PASS | from the installed wheel |
| crystallizer determinism | PASS | from the installed wheel |
| `work_resume` | **see §4** | split Python/Rust delivery |
| secrets scan | PASS | 0 hits in the product diff |
| artifact inspection | PASS after fix | see §6 |
| perf sanity | PASS | +15 ms (4.3%) on `work_resume` |
| release bug hunt | PASS | see §7 |

---

## 3. Python suite

CI's authoritative invocation is `pytest tests/ -v --tb=short --timeout=60`
(`ci.yml` integration job) and `-x --timeout=60` (python-fallback job).

**Correction to an earlier interim report.** While the run was in progress I
reported "0 failures at 40%" and again at 87%. That was wrong. With
`--tb=line`, `FAILED` lines appear only in the final summary, never in the dot
progress, so grepping a partial log for `^FAILED` returns 0 by construction. I
was reading absence of evidence as evidence of absence. The run finished with
nine failures.

```
passed      5463
failed      9   -> 1 introduced (fixed), 8 proven not introduced
skipped     38
xfailed     3
deselected  1   (test_full_lifecycle_stress, see below)
duration    2696.67s (44:56)
```

### Classification of all nine, by reproduction rather than intuition

| failure | clean checkout, `origin/main` | clean checkout, this branch | CI on `main` | verdict |
|---|---|---|---|---|
| `test_work_resume_selection::test_passed_verdict_is_not_reported_as_failed` | n/a (new test) | n/a | n/a | **INTRODUCED → FIXED** `ff659e8d` |
| `test_release_version_sync::test_release_version_surfaces_match_package_version` | PASS | **PASS** | green | local untracked debris |
| `test_release_surface_consistency::test_install_instructions_do_not_pin_an_older_version` | PASS | **PASS** | green | local untracked debris |
| `test_model_recovery_benchmark::test_fixture_generation_is_deterministic_and_phase_separated` | FAIL | FAIL | green | pre-existing, platform |
| `test_proxy_session_rescue::test_live_proxy_blocks_unrecoverable_overflow_before_upstream` | FAIL | FAIL | green | pre-existing, platform |
| `test_recovery_fidelity_fuzz::test_recovery_store_round_trip_is_byte_exact[megaline]` | FAIL | FAIL | green | pre-existing, platform |
| `test_recovery_fidelity_fuzz::test_recovery_reference_byte_length_matches_the_content[megaline]` | FAIL | FAIL | green | pre-existing, platform |
| `test_session_rescue::test_soft_pressure_defers_when_provider_cache_is_warm` | FAIL | FAIL | green | pre-existing, platform |
| `test_session_rescue::test_pressure_without_safe_candidate_is_reported_without_fake_savings` | FAIL | FAIL | green | pre-existing, platform |

Evidence for each class:

* **The one I introduced.** `test_passed_verdict_is_not_reported_as_failed`
  passed in isolation and as a file, then failed in the suite on
  `assert view["outstanding_work"] == []`, seeing
  `['wire the new config field through the Rust engine']`. Cause: the fixture set
  `ENTROLY_SOURCE` and `ENTROLY_NO_SELF_HEAL` but not `ENTROLY_DIR`, and
  `_store_root()` defaults to `~/.entroly/work-graphs` — the developer's real
  store, shared across the session. A sibling test's content-bearing workstream
  was therefore visible to a test that had deliberately recorded none, and the
  content-based selector picked it, which is the selector behaving as specified
  (W3 covers multiple workstreams). The test was asserting about accumulated
  machine state rather than about what it recorded. Fixed by giving each test
  its own store under `tmp_path`; the exact reproduction
  (`bindings + store + multiprocess + this file`) went from 1 failed / 26 passed
  to **28 passed**. Test-only change.
* **The two debris failures.** Caused by untracked files in the working
  directory, neither known to git: `entroly-wasm/pkg/package.json` pinned at
  `1.0.83` (a stale local `wasm-pack` output) and a deeply nested
  `openclaw_repo/.artifacts/vitest-worker-cache/...` path that breaks an
  `rglob`. `git ls-files` confirms 0 tracked files under `openclaw_repo/` and
  that the wasm `package.json` is "not known to git". Both **pass from a clean
  checkout of this branch**, which is what CI does.
* **The six platform failures.** They fail identically from a clean worktree at
  `origin/main` *and* from a clean worktree of this branch, while
  `gh run list --workflow ci.yml --branch main` reports `ci.yml` **success at
  `e31f082f`** — main's head. So they are Windows-local and invisible to CI's
  Linux runners, present on both branches, and not attributable to this change
  set.

Separately deselected: `tests/test_deep_functional.py::test_full_lifecycle_stress`
("D-32 FULL LIFECYCLE STRESS, 20 turns"), labelled
**PRE-EXISTING / ENVIRONMENTAL TIMEOUT, NON-BLOCKING FOR THIS CHANGESET** —
established, not assumed: run in isolation from an `origin/main` worktree on the
same machine at the same 60-second limit, it times out identically. The file is
not in this branch's diff and CI runs it green at a stricter timeout. **It is not
fixed**, and it is recorded here rather than silently omitted.

---

## 4. `work_resume` — the one finding that changes what may be claimed

The continuation-projection fix is **split across two distributions**:

* the Python half — selection in `work_graph_mcp.py`, outstanding-work refs in
  `work_graph_store.py` — ships in the `entroly` wheel;
* the Rust half — `ResumeView.outstanding_work`, `VerificationView` carrying the
  verdict, `WorkItemView.remaining_work` — lives in
  `entroly-engine/src/work_graph.rs` and reaches users only through a **new
  `entroly-core` wheel**.

The clean-install smoke proved this rather than inferring it. The fresh venv
resolved `entroly-core>=1.0.85,<2` to the **published** 1.0.85 — 8,569,344 bytes,
sha `fd79db35…` — not the local build (8,599,552 bytes, `cf4d6e83…`). Against
that published native, `resume()["verification"]` is still a list of label
strings, so the verdict fix is inert for a user installing today.

Both `entroly` and `entroly-core` 1.0.85 are already on PyPI. Consequence:

> Delivering the continuation-projection fix requires a coordinated version
> bump (1.0.86) across the Python package and the Rust crates, published
> through the tag-driven pipeline. A Python-only release at 1.0.85 would not
> carry it.

No bump is made on this branch. `CLAUDE.md` specifies `scripts/bump_version.py`
and a tag on the **post-merge commit on main**, and bumping ~57 targets here
would mix release mechanics into a correctness PR.

**Claim discipline:** until that coordinated release exists, the four
Python-side fixes may be described as shipped; the continuation-projection fix
may not.

---

## 5. Pre-existing lint debt — not introduced, not CI-gated

* `cargo fmt --check` on `entroly-engine`: **105 diffs on `origin/main` itself**,
  verified in a clean worktree. CI runs `cargo fmt` without `--check` for that
  crate, so it is not a gate. Not touched; reformatting 105 files inside a
  correctness release would be the wrong trade.
* `cargo clippy` on `entroly-qccr`: 2 errors,
  `unnecessary use of 'clone' to create a slice from a reference`. The crate has
  **0 files** in this branch's diff, and its CI gate
  (`qccr-signature-completion-gate.yml`) runs only `cargo fmt --check` and
  `cargo test` — not clippy. Recorded as debt.

---

## 6. Artifact hygiene — one real defect, fixed

144 raw agent event traces under `research/ledger/pilot1/traces/` were tracked,
and 9 of them embed the absolute temp-workspace path of the machine that ran the
benchmark, plus a `".config/git/ignore: Permission denied"` line. This
repository is public, and `hatch`'s sdist target has no exclusions configured,
so everything not VCS-ignored is an sdist candidate.

Fixed in `15a71950`: untracked (not deleted — all 144 remain on disk as the
Attempt-1 record) and `.gitignore`d. The structured record was verified
personal-path-free and stays versioned: `results.json`, `analysis.json`, both
manifests and `PILOT1_ATTEMPT1_RECORD.md` return 0 hits.

Post-fix artifact audit:

```
wheel   404 members, top level {entroly, entroly-1.0.85.dist-info}
        0 suspicious paths, 0 files containing the personal path
sdist   1,932 members
        0 of 1,770 text files contain the personal path
```

---

## 7. Adversarial pass over the product changes

* **Serialization back-compat.** `ResumeView.verification` changed from
  `Vec<String>` to `Vec<VerificationView>`. Contained: `ResumeView` is a derived
  runtime view and is **not persisted**; its only internal consumer
  `WorkScope::from_resume` does not read `.verification`; no golden fixture or
  JSON schema pins the old shape; and `entroly-wasm/js/cogops.js`
  `result.verification` is an unrelated epistemic-flow object. It is still an
  **MCP tool-output shape change** and belongs in release notes.
* **Strict-UTF-8 error paths.** Both call sites fail closed and visible.
  `cli.py` catches `UnicodeError` and returns a well-formed hook response
  declaring the integration inactive; `ravs/capture.py` catches
  `UnicodeDecodeError` and drops the event rather than recording mojibake.
  Neither crashes the host.
* **Cross-request leakage.** `RequestAttributionStore` is lock-guarded with a
  TTL and bounded FIFO eviction; exactly one caller can win
  `PENDING → IN_FLIGHT`, exercised by a 16-thread barrier test. No "latest
  request" fallback exists by construction, and a source-level test asserts the
  old global attribute cannot return.
* **Determinism.** Crystallizer trimming is total-ordered; the cross-process
  sweep over `PYTHONHASHSEED` 0/1/42/12345/7919/104729 plus unseeded runs is
  identical on all six compared properties.
* **Stale native loading.** Gated by a test that fails when the loaded `.pyd`
  does not match the local build — the defect that silently invalidated an
  earlier measurement round.
* **Perf.** `work_resume` gained one graph load: 15.2 ms median, 4.3% of its
  354 ms. Not pathological.

---

## 8. Decision

```
RELEASE READY

release commit   15a71950 (rebased: b52e50c5)
branch           pushed as its own branch for PR; main not pushed directly
remote target    origin
blockers         0
```

Known non-blocking issues:

1. `work_resume` Rust half needs a coordinated 1.0.86 publish (§4). Claim
   discipline recorded.
2. `entroly-qccr` clippy: 2 pre-existing errors, crate untouched, not CI-gated.
3. `entroly-engine` `cargo fmt`: 105 pre-existing diffs, not CI-gated.
4. `test_full_lifecycle_stress` times out on this machine, proven identical on
   `origin/main`.
5. The passive placeholder workstream is still created, no longer selected.
   Recorded cleanup debt.

Continuity business benchmark status: **INCOMPLETE and independent of this
release.** Pilot-1 Attempt 2 is frozen and blocked on Codex quota. The frozen
Entroly commit is preserved by the local tag `pilot1-attempt2-frozen` →
`5361e27b`, so release work moving HEAD cannot disturb it; the benchmark must
run against that commit and its recorded native hash, never against a later
tree.

```
Production continuation payload value:   INSUFFICIENT EVIDENCE
Natural predecessor handoff superiority: UNTESTED
Continuity paid wedge:                   INSUFFICIENT EVIDENCE
```
