# Model-triggered recovery holdout

This benchmark asks a narrower and harder question than active-context
retention: after compression for one question, can a model answer a different
future question by deciding when to retrieve and then using persistent source
evidence?

It compares current Entroly source at package version `1.0.59` with the
published `headroom-ai==0.31.0` wheel. The shared guard is local
`qwen2.5:1.5b` Q4_K_M with temperature zero. That small model is useful for a
deterministic workflow veto; it is not a proxy for a hosted frontier model.

## Frozen workflow

Each fixture is a 48-record JSON audit log containing an initial incident and a
different archived recovery record. Compression sees only the initial incident
question. The future audit-case question is revealed after compression.

For raw, Entroly, and Headroom context, the model must return either the exact
recovery code or exactly `RETRIEVE`. Retrieval runs only after that exact token;
a wrong or verbose response does not receive oracle help. A successful
retrieval permits one retry, and that retry becomes the final answer.

- Entroly uses `compress_proxy_payload`, its durable omitted-span store, and a
  bounded source-exact search. A complete query-matching JSON object is
  preferred when it fits the 600-token recovery cap; the full stored span is
  still available through the existing full-retrieval tool.
- Headroom uses its public `compress()` entry point with the documented
  `agent-90` profile, followed by the same persistent `CompressionStore`
  store/retrieve contract used by its MCP implementation.
- This runner composes the public Python contracts. It does not include MCP
  transport latency and does not claim to reproduce every agent's tool-use
  prompting.

The accepted protocol is
[`model_recovery_protocol_v7.json`](../../benchmarks/model_recovery_protocol_v7.json).
It freezes 24 holdout fixtures, participant versions, tokenizer, guard model,
generation settings, budgets, seeds, and claim policy before the holdout is
opened.

## Result

The complete 24-case holdout passed every integrity gate with no execution
errors. Raw context and Entroly both scored **24/24 exact**; Headroom scored
**18/24 exact**. The six discordant pairs all favored Entroly, producing a
two-sided exact McNemar **p = 0.03125**. This supports a scoped win for this
frozen local query-shift workflow, not a general model or product claim.

| Holdout metric | Entroly 1.0.59 source | Headroom 0.31.0 |
|---|---:|---:|
| Final exact answers | **24 / 24 (100%)** | 18 / 24 (75%) |
| First-pass exact answers | 0 / 24 | 18 / 24 |
| Model-triggered retrievals | 24 / 24 | 0 / 24 |
| Successful recoveries | **24 / 24** | 0 / 24 triggered |
| Mean active-context ratio | **18.31%** | 42.97% |
| Mean effective-context ratio | **28.88%** | 42.97% |
| Execution errors | 0 | 0 |

Entroly's mean effective footprint (active context plus recovery evidence when
triggered) was **32.8% lower relative to Headroom**. Every Entroly retry used a
complete query-matching JSON object copied byte-for-byte from the persistent
source span. Headroom answered 18 cases from active context; on the other six
it returned a wrong value rather than the exact `RETRIEVE` token, so the
no-oracle rule correctly withheld retrieval.

The full artifact, including fixtures, contexts, model outputs, hashes, token
counts, errors, and provenance, is
[`model_recovery_v7_holdout.json`](../../benchmarks/results/model_recovery_v7_holdout.json)
(`payload_sha256 = c8a41f070ff71d4a730df213bcb464fea779f793ae66f31f5a929b982046071c`).

## Integrity and verifier contract

Every model call is atomically checkpointed with the protocol hash, model
digest, context hash, prompt hash, prediction hash, and exact error state.
Interrupted runs resume only when that contract matches. The checkpoint is
removed only after the complete artifact passes its gates.

The independent verifier regenerates every fixture and recomputes:

- matrix completeness and unique system/fixture pairs;
- protocol, prompt, context, prediction, and canonical-answer hashes;
- shared-tokenizer active, recovery, and effective token counts;
- participant versions and frozen generation settings;
- no-oracle retrieval triggers and retry context hashes;
- retrieval content and per-item provenance hashes;
- Entroly excerpt membership in the original source and Headroom's stored
  source equality;
- final scores, paired exact McNemar result, aggregates, and quality gates.

Errors and wrong answers remain in the matrix as zero-score outcomes.

## Rejected and corrected variants

The protocol history is intentionally retained rather than rewritten:

- V1 and V2 record the unusable 7.6B/32K CPU pilot, including timeouts.
- V3 recovered answers but returned too much omitted context, producing an
  unacceptable 118.8% Entroly effective-token ratio in development.
- V4 enforced a cap but arbitrary windows produced incomplete JSON; Entroly
  scored 0/4 in development.
- V5 restored complete source-exact JSON objects. Its locked 12-case holdout
  passed execution and integrity gates, but its report generator retained one
  stale `7B/32K` limitation label even though the embedded protocol and model
  identity correctly recorded `qwen2.5:1.5b`. The artifact is preserved but is
  not the publication artifact.
- V6 tried a smaller scalar-field projection. Development fell to 1/4 for
  Entroly versus 4/4 for Headroom, so the optimization was rejected and never
  became the default recovery path.
- V7 restores the reliable complete-object strategy, generates model-scope
  limitations from the actual frozen model, and uses fresh seeds.

This history is evidence of the engineering gate doing its job: lower token
counts do not override answer reliability or metadata correctness.

## Reproduce and verify

Install `headroom-ai==0.31.0` in an isolated Python 3.10 environment and run a
local Ollama server with the exact model recorded by the protocol.

```powershell
python -m benchmarks.model_recovery run `
  --phase holdout `
  --protocol benchmarks/model_recovery_protocol_v7.json `
  --headroom-python C:\path\to\headroom-0.31.0\Scripts\python.exe `
  --output benchmarks\results\model_recovery_v7_holdout.json

python -m benchmarks.model_recovery verify `
  --input benchmarks\results\model_recovery_v7_holdout.json
```

Development artifacts and failed variants are evidence for engineering
decisions only. The protocol forbids using them for a public win claim.

## Limits

This is a synthetic exact-answer query-shift workload on one Windows/Python
3.10 machine and one local 1.5B quantized guard. It does not establish general
agent quality, hosted-model quality, provider-observed cost, production
latency, Linux/macOS behavior, MCP transport behavior, neural superiority, or
overall product superiority. A scoped accuracy-superiority claim additionally
requires the complete gate and a paired two-sided exact McNemar `p <= 0.05`.
