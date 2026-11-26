# Benchmark results

The [benchmark evidence guide](../../docs/BENCHMARKS.md) links the measured
artifacts in this directory and documents their workloads, baselines, and
limitations.

## End-to-end coding agent evaluation

The [coding task harness](../coding_tasks.py) currently lists its fixture tasks in dry-run mode.
Its live execution path still needs a model call, routing through the Entroly
proxy, and a test oracle for each task. Until those are wired, a live run
stops with an error.

```bash
python benchmarks/coding_tasks.py --dry-run
```

For harness development, `--simulate` writes fabricated metrics to
`benchmarks/simulated/`. Those files are for exercising the report format.

The [agent task evaluation protocol](../AGENTIC_TASKS_PREREGISTRATION.md) defines
the planned comparison between raw context, compression alone, and Entroly with
recovery and verification. The primary outcome is verified coding tasks
completed per dollar, with task success checked against a fixed test oracle.
