# Real-agent workload matrix and execution status

No live agent A/B/C trials were executed in this session. Every row below is an explicit **NOT TESTED** future evidence gate, not a passed benchmark.

| Workload | No Entroly (A) | Defaults (B) | Tuned (C) | Required oracle | State |
| --- | --- | --- | --- | --- | --- |
| Cross-file bug repair | Required | Required | Required | Tests pass and patch matches expected behavior | NOT TESTED |
| Architecture Q&A | Required | Required | Required | Human-verified answers with cited source spans | NOT TESTED |
| Refactoring across modules | Required | Required | Required | Functional tests and no API regressions | NOT TESTED |
| CI failure diagnosis | Required | Required | Required | Root cause and successful fix, not just log summaries | NOT TESTED |
| 25+ turn file-reading agent | Required | Required | Required | Exact source lines recovered when necessary | NOT TESTED |
| Structured JSON/tool response | Required | Required | Required | Critical fields preserved or reversibly retrievable | NOT TESTED |
| Multi-agent and concurrent sessions | Required | Required | Required | No cross-request or cross-session leakage | NOT TESTED |
| Memory reuse across a restart | Required | Required | Required | Correct source and freshness validation | NOT TESTED |
| Negative, no-repo-context query | Required | Required | Required | Zero unsolicited context and no unnecessary latency | NOT TESTED |
| Recovery after missing evidence | Required | Required | Required | Verified byte-exact source recovery | NOT TESTED |

## Required experiment controls

- Record SHA, OS, CPU, memory, Python/Rust/native version, provider model, temperature, context budget, exact task and tool access.
- Run A/B/C on the same fixed repository revision and with identical instructions. Disclose stochastic or tool-order differences.
- Repeat trials; report median, p95 and variance, not one successful trace.
- Preserve prompts, replies and diffs only in an authorized private location; never commit secrets or user data.
- Use independent correctness oracles rather than token reduction or an agent's self-reported satisfaction.
- Evaluate successful tasks divided by total tasks and dollars **per successful task**; include retries and output cost.
- Unavailable provider/model combination is `NOT TESTED`, not zero spend.

## Existing synthetic coverage

The `Deep Dogfood Runtime` GitHub workflow covers installed MCP protocol, several internal feature families and a release artifact black box; separate PR #587 tests exercise agent-session proxy fidelity. These are valuable regression gates but not controlled live task-completion comparisons.
