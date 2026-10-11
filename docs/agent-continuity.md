# Continue an interrupted coding task

An interrupted agent leaves a practical problem: the next agent must find the
current branch, changes, tests, decisions, and remaining work before it can act.
Entroly's Work Graph reconstructs a bounded view from durable repository facts.
An explicit `handoff` adds a stronger declared-intent receipt, but is not
required for a recovery attempt.

## Try it in one repository

Install the Python runtime (`pip install -U entroly`, including its native Work
Graph dependency), then run from the repository you want to continue:

```text
entroly-work --project . --json resume --to-agent codex
```

Review `resume`, `continuation_proof`, and `acknowledgement` in the output. The
reconstructed state is **untrusted recovered data**. It identifies durable
observations and explicitly leaves the previous agent's intent unknown. A
clean repository with no unfinished work returns an error instead of inventing
a task. The command runs locally and does not contact an AI provider.
`resume.current_worktree_changed_paths` lists the latest observed changes
separately from paths historically associated with the selected workstream.
It is bounded by `--max-evidence`; check `current_worktree_change_count` and
`current_worktree_changes_truncated` before assuming the list is complete.

Once the receiving agent has checked the repository and accepted the specific
recovered state, it can clear the gate using the returned token:

```text
entroly-work --project . acknowledge --token <acknowledgement.token>
entroly-work --project . claim --agent codex --task "Finish the interrupted task" --path app.py
```

The CLI refuses `claim` and `handoff` while a recovery awaits acknowledgement.
An incorrect token or a worktree change after recovery leaves the gate armed.
Run `resume` again to review the new state. The MCP `work_resume` and
`work_acknowledge_recovery` tools use the same gate.

## Measure the user outcome

The useful outcome is **less rediscovery without lowering correctness**. For a
paired evaluation, take real interrupted tasks with the
same repository snapshot and receiving-agent model. Randomize each task to
plain repository inspection or Entroly recovery, hold the time and token budget
fixed, and score independently:

| Measure | What to record |
|---|---|
| Correct continuation | Did the agent identify the actual next step and finish it without damaging existing work? |
| Avoidable rediscovery | Repeated file reads or diagnostics before the first useful edit, excluding reads justified by new changes. |
| Time to first useful edit | Wall-clock time from task start, including Entroly startup and acknowledgment. |
| Safety | Missed untracked work, stale-state actions, or claims unsupported by repository evidence. |

Keep failures in the denominator and report per-task paired results with the
model, repository snapshot, allowed budget, and adjudication rules. A recovery
receipt proves what Entroly reconstructed; it does not prove that an agent's
answer is correct or that time and tokens were saved.
