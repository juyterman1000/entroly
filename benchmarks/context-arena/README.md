# Context Arena

**Can your AI agent's context layer survive a real repository's history?**

Context Arena is an open, reproducible benchmark for *context systems* —
retrieval layers, context compressors, memory/RAG stacks, and agent context
"OSes". It does not benchmark models. It benchmarks the thing that decides
**what the model sees**.

> Working name — final name/branding TBD before public release.

## Why another benchmark

Every context tool advertises token savings. Almost none measure the thing
that matters: **does the agent still succeed at the task?** Published results
in this category are typically single-shot extractive QA (SQuAD-style), not
agent work; position papers on continual-learning agents openly note that
evaluation "is conducted in static offline setups rather than true continual
learning settings." Context Arena fills exactly that gap:

- **Tasks are mined from real git histories, not hand-written.** A task is a
  real bug-fix commit, reverted; the repository's own failing test is the
  oracle. Deterministic ground truth, no LLM judging, scales to hundreds of
  tasks per repository.
- **Fail-to-pass construction.** At the fix commit the touched tests pass;
  with the source change reverted (tests kept), they fail. An agent (or a
  context layer feeding one) must supply the understanding needed to make
  them pass again.
- **Falsification-first rules.** Criteria are preregistered before runs;
  failed runs stay in the record; paired trials with exact McNemar tests;
  artifacts carry payload hashes. If a result can't be regenerated with one
  command, it doesn't exist.

## What gets measured

| Metric | Meaning |
|---|---|
| task success | oracle test passes after the agent's attempt |
| input tokens | context cost actually paid |
| success-per-dollar | the frontier that decides real deployments |
| paired deltas | exact McNemar on success; bootstrap CIs on tokens |

## Protocol

The frozen v1 protocol lives in [PROTOCOL.md](PROTOCOL.md). Summary:

1. **Mine** — scan a repo's history for commits touching source + tests
   together (`runner.py discover`).
2. **Validate** — prove each candidate is a real task via the fail-to-pass
   check in an isolated worktree (`runner.py validate`).
3. **Run arms** — for each validated task, run the agent once per arm
   (no-context baseline / your context layer / others), same model, same
   budgets.
4. **Report** — paired stats only; dev/holdout split by repository.

## Plugging in your context layer

Implement one class ([adapters/base.py](adapters/base.py)):

```python
class ContextAdapter:
    name = "my-context-layer"

    def prepare_context(self, task: dict, token_budget: int) -> str:
        """Return the context string your layer selects for this task."""
```

`task` carries the repo snapshot path, the failing test command, and the
task description (the commit subject). Your adapter sees exactly what every
other adapter sees.

## Honesty rules (non-negotiable)

- Preregister thresholds before running; commit the preregistration.
- Every run's artifact is committed — including failures.
- No post-hoc metric or threshold changes without a declared protocol bump.
- Contamination: record commit dates; report pre/post model-cutoff splits.

## Quick start

```bash
# 1. discover candidate tasks from any git repo
python runner.py discover --repo /path/to/repo --out candidates.jsonl

# 2. validate them into real fail-to-pass tasks (bounded)
python runner.py validate --repo /path/to/repo \
    --candidates candidates.jsonl --out tasks.jsonl --max-validate 25
```

Phase 2 (agent-loop arms + leaderboard) follows the preregistered protocol in
`benchmarks/AGENTIC_TASKS_PREREGISTRATION.md` of the entroly repository.
