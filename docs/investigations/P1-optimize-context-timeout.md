# P1 — `optimize_context` MCP tool times out (investigation record)

**Status:** **ROOT-CAUSED AND FIXED** — see "Resolution" below. The elimination
work is kept because it is what forced the search to the right place, and
because two plausible-sounding hypotheses were disproved along the way.

## Resolution

A `py-spy` thread dump of the live server (PID 50232) found the incremental
watcher permanently blocked:

```
Thread 61176: "entroly-watcher"
    _wait_for_tstate_lock (threading.py:1105)
    join -> _communicate -> communicate -> run (subprocess.py:512)
    _git_ls_files (auto_index.py:187)
    _reconcile_index (auto_index.py:769)
    reconcile_index (auto_index.py:746)   <- holds _index_mutation_lock
```

`subprocess.run(capture_output=True, timeout=10)` does not bound this call. The
timeout bounds the *wait*; `communicate()` then joins the stdout/stderr reader
threads, which exit only when the pipes close. A git that stops for credentials,
opens a pager, or blocks on `index.lock` holds a pipe open, so the join never
returns and the timeout never fires. Two orphaned `_readerthread`s were parked
in the same dump. The watcher then held `_index_mutation_lock` **forever**, so
every later ingest/reconcile/auto-index blocked and the index silently stopped
updating.

Fixed by `_run_git`: explicit `Popen`, `stdin=DEVNULL`, `stderr=DEVNULL`, kill on
timeout plus a bounded reap, and a strictly non-interactive git environment.
Discovery degrades to the filesystem walk instead of hanging. Regression tests:
`tests/test_git_discovery_cannot_hang.py`.

**Lesson worth generalising:** no component measurement could have found this.
Every phase was fast in isolation; the fault was a *stuck thread holding a lock*,
visible only in a dump of the live process. When components measure fast but the
system is slow, dump the process before restarting it.

**Journey affected:** D (MCP user) / E (Claude/Codex/Cursor) — "call the primary
context tool". `optimize_context` is documented as *"the core tool — call it
before sending context to the LLM."*

## Reproduction

Live MCP server (PID 50232, started 2026-07-24 19:01, entroly 1.0.66, native
core built 2026-07-23T20:28), repo `entroly-codebase-intelligence` @ `e5f20a4`.

```
optimize_context(query="proxy inject compressed context", token_budget=1200)
-> MCP error: tool "optimize_context" timed out after 60s
```

Deterministic. Reproduced at budgets 1000, 1200, 3000. `get_stats` and
`prepare_task_dream` on the **same server** return promptly, so the process is
healthy and responsive — the fault is specific to this tool.

## Measured components (all ruled out)

Measured in-process against the same persisted index (906 fragments, warm-started
from `~/.entroly/checkpoints/0e1db5e15dd3/index.json.gz`):

| component | measured | verdict |
|---|---|---|
| `engine.optimize_context()` | 0.93–3.90s | not the cause |
| `task_dream` | budget < 1024 skips it entirely; **still times out**. `prepare_task_dream` alone returns fast | ruled out |
| vault belief bridge (`load_vault_beliefs`) | 0.47s, project vault is 588 files / 6.2 MB | ruled out |
| `rebuild_dependencies()` | 0.74s (136,465 edges) | ruled out |
| `advance_turn()` | Rust delegate + periodic `gc.collect()` | ruled out |
| CCR `capture_recoverable_fragments` | 0.00s | ruled out |
| `_auto_checkpoint()` | 3.71s (39.8 MB `export_state`) | contributes, not sufficient |

Sum of the full measured path: **~4.6s**. The observed failure is **>60s**.

## Hypotheses explicitly retracted

Two earlier hypotheses were **disproved by measurement** and must not be revived
without new evidence:

1. *"The 134k-edge dependency-graph rebuild costs ~27s."* — False.
   `rebuild_dependencies()` is **0.74s**. The 27s previously observed was the
   re-ingest of a changed-file backlog, not the graph rebuild.
2. *"`optimize_context` blocks on `_index_mutation_lock` held by a background
   reconcile."* — False. That lock is acquired **only** in `auto_index.py`
   (`auto_index`, `reconcile_index`); the read/optimize path never takes it.

## Leading remaining hypothesis (untested)

The tool has **never** completed on this server instance during the session,
while other tools stay fast. If the MCP runtime dispatches tool calls serially,
one stalled first call would poison every subsequent call — each new request
queues behind the stuck one and hits the 60s client timeout, forever. That is
consistent with every observation: healthy process, fast unrelated tools,
deterministic timeout independent of budget.

Testing this requires either (a) instrumenting the running server, or (b)
restarting it — which destroys the stuck-state evidence. Capture a thread dump
of the live process **before** any restart.

## Adjacent defect found while measuring (real, separate)

`_auto_checkpoint()` writes a **39.8 MB** state file and takes **3.71s**, at
`auto_interval: 5`. `~/.entroly/checkpoints` holds **127 checkpoints / 264 MB**
(`own_checkpoints: 0`, `peer_checkpoints: 127`) with no observed retention or
GC. This is unbounded disk growth plus multi-second latency on a user-facing
path, and should be fixed independently of the timeout.

## Do not

- Do not ship a timeout fix without a reproduced root cause.
- Do not raise the client timeout to mask it.
- Do not re-test the two retracted hypotheses without new evidence.
