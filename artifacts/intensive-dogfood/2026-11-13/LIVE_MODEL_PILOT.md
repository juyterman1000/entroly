# Live-model coding-task pilot: measured negative result

Date: 2026-11-13. Project: Entroly draft PR #592.

## Run 1: CPU Ollama Qwen 2.5 Coder 0.5B

Source head: c3a6116464e19db20baa998acae30db380f59d0d. GitHub Actions run [43499995592](https://github.com/juyterman1000/entroly/actions/runs/43499995592); artifact **intensive-dogfood-real-model-pilot**. The runner verified the checkout SHA. Model tag qwen2.5-coder:0.5b, registry digest 4ff64a7f502a08b7616edb8ca0a79eb1853fc363d842b7df4b46915d11a3fb09.

Measured totals (4 dependency-bearing **synthetic** coding tasks, one seed 17, single seed-matched prompt per initial arm, deterministic pytest oracle):

| Measure | RAW | COMPRESS | CLOSED_LOOP |
| --- | ---: | ---: | ---: |
| Executable task fixes passed | 0 / 4 | 0 / 4 | 0 / 4 |
| Ollama prompt tokens | 2,248 | 994 | 2,054 |
| Ollama output tokens | 221 | 242 | 448 |
| Total model generation wall time, seconds | 19.439 | 11.844 | 21.891 |

Measured reduction in input tokens: COMPRESS vs RAW **55.78%**; CLOSED_LOOP (four real second calls, both attempts charged) vs RAW **8.63%**.

### Negative outcome and confounds

RAW did not solve any task. Its failure modes were a mix of unresolved bugs and unusable Python replies. The tiny model is unable to establish a useful passing baseline; **zero discordant pairs do not imply accuracy noninferiority**. No improvement claim is licensed. The CLOSED_LOOP recovery procedure is deterministic import closure over omitted source after a failed pytest oracle, **not** autonomous agent error diagnosis. All four retries failed with this model.

Input/output counts are reported by the local Ollama runtime and are real model-token measurements, **not** paid-provider invoice data. Model serving energy and compute are not priced. The prompt context estimate is not substituted for the measured token count. The runner preserves per-task prompt/response hashes, selected paths and oracle failure snippets in the ZIP artifact.

### Next staged, separate execution

The workflow now launches an additional Qwen 2.5 Coder **3B** run with seeds 17 and 42 on the same four tasks, to address this floor effect; its outcome is not assumed. A separate historical-commit miner checks real repository fix/revert test oracles as task-source evidence; it does not claim to have executed those commits through the LLM.

### Gates not completed

- Preregistered >=400 mined test-oracle tasks across repository-separated dev/holdout.
- Real coding agents with tools and multi-turn behavior, beyond the single-file local-model pilot.
- Invoice-backed OpenAI/Anthropic/Gemini token cost reconciliation and caching behavior.
- Statistically powered accuracy noninferiority and success-per-dollar frontier.
- End-to-end latency for hosted providers and exhaustive audit across all modules.

This negative result must remain visible in all dogfood reporting. The PR stays draft and unmerged until appropriately reviewed. 

## Run 2: CPU Ollama Qwen 2.5 Coder 3B — genuine paired task-oracle results

Exact source head `858753de44ecf7ad66a86f7e2c7e21fd07a17b62`; [GitHub Actions run 43500819779](https://github.com/juyterman1000/entroly/actions/runs/43500819779); model digest `f72c60cabf6237b07f6e632b2c48d533cef25eda2efbd34bed21c5e9c01e6225`.

Four synthetic dependency-bearing tasks × seeds 17 and 42 = **eight paired attempts** (not eight independent tasks):

| Metric | RAW | COMPRESS | CLOSED_LOOP |
| --- | ---: | ---: | ---: |
| Executable task successes | **4 / 8** | **4 / 8** | **6 / 8** |
| Ollama prompt tokens (measured) | 4,496 | 1,988 | 2,926 |
| Ollama completion tokens (measured) | 356 | 354 | 544 |
| Model inference wall time, seconds | 60.477 | 39.452 | 57.753 |

COMPRESS used **55.78% fewer measured prompt tokens** than RAW with exactly the same success set in this sample; paired McNemar raw-vs-compress has zero discordant pairs and p=1, which is **not** a noninferiority demonstration. CLOSED_LOOP used **34.92% fewer prompt tokens** than RAW including real retries and fixed two extra task attempts (the fee-in-cents integration bug under both seeds). The TTL-unit bug remained unsolved for both seeds and all arms. Repeated seeds on the same task are not independent tasks.

**Accounting caveats:** Model latency excludes context selection and external service startup. Token totals exclude context cache effects (Ollama does not supply commercial invoice records), GPU/CPU energy cost, storage and retries beyond one. No monetary return, success-per-dollar or statistically powered accuracy claim is possible. The enhanced next run executes generated code in an isolated networkless Docker pytest oracle rather than directly on the CI host, and preserves model output/patch data for audit.

## Historical mined-code task-source evidence

Workflow 43500819779 found **59 candidate source+test commits** and validated the first **8 of 8** with pass-at-fix and fail-after-revert in separate worktrees. These are real Entroly-history failure oracles, **not** eight model-solved real repository tasks. Independent repository holdout and n>=400 remain unexecuted.
