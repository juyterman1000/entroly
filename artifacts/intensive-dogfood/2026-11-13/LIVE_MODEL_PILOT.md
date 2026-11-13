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
