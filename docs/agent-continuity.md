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

## Hand off to an independent reviewer from the CLI

The installed `entroly-work` command can complete the local review loop without
an MCP client. Both agents must use the same repository and `ENTROLY_DIR` state.
This is a local workflow; it does not synchronize state between machines or
attest which model produced an agent ID.

```text
entroly-work --project . --json claim --agent writer --task "Fix auth" --path app.py
entroly-work --project . --json state
entroly-work --project . --json handoff --workstream <state.unfinished node_id> --from-agent writer --to-agent reviewer
entroly-work --project . --json handoffs --to-agent reviewer
```

`handoff` persists the compact receipt to the same bounded inbox used by MCP
and returns `handoff_ref`. The reviewer confirms that the inbox entry has
`current=true`, checks the source and tests independently, then runs:

```text
entroly-work --project . --json resume --workstream <workstream_id> --to-agent reviewer
entroly-work --project . --json acknowledge --token <acknowledgement.token>
entroly-work --project . --json review --handoff-ref <handoff_ref> --reviewer reviewer --verdict changes_requested --summary "Check auth fallback" --finding app.py 42 high "Fallback bypasses validation" --check "focused tests" 0
entroly-work --project . --json reviews --workstream <workstream_id>
```

Use `--verdict approve --summary "..."` with no `--finding` for a clean review.
`--finding PATH LINE SEVERITY MESSAGE` and `--check NAME EXIT_CODE` can be
repeated. Check results and verdicts are **reviewer statements**, not verified
command logs or correctness proofs. Each finding is bound to the current file
digest. A changed repository makes the inbox receipt stale and prevents a new
review against it; prior `reviews` entries then show `current=false`. For a
budget-exhausted writer with no explicit handoff, use the earlier `resume` flow
and report findings to the user without inventing a writer receipt.

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
