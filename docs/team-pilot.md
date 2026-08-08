# Entroly two-week team pilot

Entroly pilots answer one business question: **does recoverable context compression improve this team's AI-agent economics without an unacceptable task-quality regression?** A pilot is not a universal savings demonstration and should not begin with a percentage target copied from another workload.

## Who should run one

- Teams with measurable API or agent infrastructure spend.
- Repeated coding, support, research, CI-agent, or RAG workloads with large inputs.
- Security or governance teams that need an audit trail for what entered a model request.
- Teams willing to preserve a matched baseline instead of comparing unrelated weeks.

## Required design

1. Select representative, non-secret workloads and record their inclusion criteria.
2. Freeze the provider, model, output limits, prompt-cache policy, task rubric, and sampling method.
3. Run a baseline arm without Entroly and an Entroly arm with the same task inputs and token cap.
4. Capture provider-reported input, output, and cache tokens where the provider exposes them.
5. Record task success, recovery events, latency, failures, and exact Entroly receipt identifiers.
6. Report exclusions, missing provider fields, and operational incidents. Missing data is not zero.

The machine-readable measurement contract is [team-pilot-contract.json](team-pilot-contract.json).

## Decision gates

Before the pilot, the team must choose its own thresholds for:

- task-quality non-inferiority;
- provider-observed input-token reduction;
- p95 added latency;
- unrecoverable omission count, which should be zero for any path advertised as recoverable;
- cache-read regression;
- operational failure rate.

Entroly does not pre-fill these thresholds because the acceptable trade depends on the workload and risk class.

## Start a pilot

Use the [Team Pilot request](https://github.com/juyterman1000/entroly/issues/new?template=team-pilot.yml). Do not include source code, credentials, customer data, provider keys, private pricing, or confidential logs in a public issue. A private evaluation can begin from the same contract without opening an issue.

## Exit artifacts

- Frozen pilot configuration and workload definition.
- Baseline and Entroly raw measurements.
- Context Receipt and recovery sample.
- Quality-scoring rubric with reviewer agreement.
- Incident and exclusion log.
- A decision: adopt, revise, or stop—with the evidence that supports it.
