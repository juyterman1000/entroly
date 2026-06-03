# End-to-End Coding Benchmarks

This directory contains the results of the Entroly End-to-End Coding Agent Benchmark (`coding_tasks.py`).

## The North Star Metric

We optimize for one thing:
**minimize expected dollars + latency + recovery cost subject to P(verified task success) >= baseline - epsilon**

The headline metric reported here is:
**Verified Coding Tasks Completed per Dollar (`verified_tasks_per_dollar`)**

A 3x improvement in verified tasks per dollar is more valuable and defensible than a synthetic 100x context compression ratio. 

## Methodology

The `coding_tasks.py` harness runs deterministic coding tasks (fixing bugs, adding tests, refactoring) in isolated local workspaces against local git repositories.

We evaluate three modes:
1. **`raw` (Baseline)**: The agent runs with the full uncompressed context of the repository.
2. **`entroly_compress_only`**: The agent runs with Entroly's context compression, but *without* the auto-recovery (CCR) and verification (WITNESS) loops.
3. **`entroly`**: The full closed-loop control plane. Includes compression, PRISM learning, and the auto-recovery/verification cascade.

### Metrics Collected

For each run, we measure:
- **Pass Rate (`passed`)**: Did the task's `test_command` exit with the expected code (usually 0)?
- **Tokens In/Out**: Measured context cost.
- **Dollars**: End-to-end API cost based on the model's pricing.
- **Latency**: Total time to complete the task (including any recovery loops).
- **Repair Count**: How many times the verification cascade rejected a hallucinated/incomplete answer and auto-recovered.

## Reproducing the Benchmark

To run the benchmark yourself:

```bash
# Dry run to validate tasks
python benchmarks/coding_tasks.py --dry-run

# Run full evaluation (e.g., 5 runs per task on gpt-4o)
python benchmarks/coding_tasks.py --model gpt-4o --runs 5
```

The output will be saved as a JSON trace file in this directory.

## Interpreting the JSON Traces

The resulting `coding_tasks_<timestamp>.json` files contain the full trace of every run.

- `summary`: Aggregated statistics across all runs, broken down by mode.
- `raw_traces`: Array containing the per-run metrics for each task. You can use this data to analyze token savings vs. pass rate trade-offs.
