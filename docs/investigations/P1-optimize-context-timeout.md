# P1 — `optimize_context` MCP tool times out (investigation record)

**Status:** reproduced, **not root-caused**. No fix shipped — a speculative fix
would be worse than none. This record exists so the next engineer does not
repeat the elimination work.

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
