# Evaluating Entroly alongside other context tools

This draft intentionally avoids competitor scorecards. Fast-moving projects change too quickly for an unversioned feature table to stay fair or useful.

## Start with the job to be done

Entroly is designed for teams that need to select context under a budget, retain provenance, recover captured omitted content, inspect receipts, and gate learning on verified outcomes. Another tool may optimize a different part of the agent loop. Those capabilities can be complementary.

## A fair comparison protocol

Compare any two tools on the same:

- public repository and revision;
- model, provider endpoint, and pricing source;
- task set and random seed;
- input budget and cache state;
- hardware and warm/cold process state;
- answer-quality grader and confidence interval;
- failure, retry, and timeout policy.

Report raw requests and responses when licensing and privacy permit. Keep failures in the denominator. Separate source-token reduction, provider cache savings, output quality, latency, and recovery correctness.

## Entroly-specific checks

Verify these capabilities directly instead of inferring them from marketing copy:

1. `entroly verify-claims` for a bounded local install smoke.
2. A representative proxy or MCP run for source and selected token counts.
3. Context receipts for included and omitted evidence.
4. Recovery tests for captured content and retained-state boundaries.
5. WITNESS benchmarks for evidence-grounding signals and known limitations.
6. Verified-outcome gates for any adaptive policy or synthesized skill.

The current artifacts and reproduction commands live in the [public evidence ledger](../public-evidence.md).

## How to write the result

Name the exact versions and date. Say “did not expose this capability in the tested path” rather than “does not have it.” Do not imply endorsement, attack maintainers, or generalize one workload into universal superiority.

The strongest conclusion is often conditional: which tool is more dependable for a particular workload, under a stated protocol, with evidence another developer can reproduce.
