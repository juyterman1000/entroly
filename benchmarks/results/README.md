# Benchmark results

The [benchmark evidence guide](../../docs/BENCHMARKS.md) links the measured
artifacts in this directory and documents their workloads, baselines, and
limitations.

## End-to-end coding agent evaluation

The goal is to reduce expected API cost, latency, and recovery cost while
keeping verified task success within a prespecified margin of the baseline.
The planned headline metric is **verified coding tasks completed per dollar**
(`verified_tasks_per_dollar`).

The [agent task evaluation protocol](../AGENTIC_TASKS_PREREGISTRATION.md) defines
a matched comparison on repository tasks such as bug fixes, test additions,
and refactors. Each task runs in isolation and has a fixed test oracle.

| Planned mode | What the agent receives |
|---|---|
| `raw` | Full uncompressed task context. |
| `entroly_compress_only` | Entroly compression with auto-recovery and verification disabled. |
| `entroly` | Entroly compression with WITNESS verification, omitted-span recovery, and one retry. |

The planned measurements include pass rate from the task's test command, input
and output tokens, API cost, elapsed time including recovery, and the number of
repair attempts. Per-run traces and per-mode summaries would show how token
savings affect task success and cost per successful task.

## What runs today

The [coding task harness](../coding_tasks.py) lists two fixture tasks in dry-run
mode. Its live execution path still needs model calls, proxy routing, and
execution of each task's test oracle. A live run stops with an error until
those pieces are wired.

```bash
python benchmarks/coding_tasks.py --dry-run
```

For harness development, `--simulate` writes fabricated metrics to
`benchmarks/simulated/`. The simulated JSON includes per-mode `summary` and
per-run `raw_traces` so developers can exercise the report format.
