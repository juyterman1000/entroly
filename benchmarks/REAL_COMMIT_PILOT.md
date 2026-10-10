# Real-commit local-model pilot

`real_commit_pilot.py` runs a development comparison on fail-to-pass commits
validated by `agentic_task_miner.py`. It is separate from the preregistered
400-task, cross-repository experiment. Its full-context control contains the
candidate's changed source and touched tests, not the entire repository.

Freeze a JSON protocol before any model call. Required fields are `schema`
(`entroly.real-commit-pilot.v1`), `source_sha`, `tasks_sha256`, ordered
`candidate_shas`, `model`, `model_digest`, `seed`, `max_corpus_bytes`,
`selection_budget`, `max_output_tokens`, `context_window`, `model_timeout_s`,
and `test_timeout_s`. Supply the exact validated JSONL named by the protocol.
The result path must be outside the repository and must not already exist.

```bash
python -m benchmarks.real_commit_pilot \
  --protocol /path/to/frozen-protocol.json \
  --tasks /path/to/validated-tasks.jsonl \
  --repo /path/to/entroly \
  --out /path/outside/repo/pilot-result.json
```

The runner only contacts a loopback Ollama HTTP endpoint. It records the model
digest and provider token counts. For each eligible task it reconstructs the
broken source in a detached worktree, reconfirms the failing oracle, and runs
three arms with fixed decoding settings: full, selected, and full repeat.
Model patches may touch only the declared source files; each arm gets a fresh
worktree and the exact touched tests. Oversized corpora, changed oracles,
selection errors, unusable patches, and test failures remain distinct results.
An in-progress artifact is checkpointed after every row; only `status=complete`
means the run finished.
The private result artifact retains unusable model output for diagnosis. Keep
it outside the repository; it may echo the task's source or test content.

This small, single-repository development set cannot establish a population
risk bound, causal context effect, non-inferiority, answer quality improvement,
or token/cost savings. Full/full-repeat is a control, not a subtraction rule.
