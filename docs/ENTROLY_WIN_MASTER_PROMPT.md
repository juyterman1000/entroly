# Entroly Competitive-Win Master Execution Prompt

## Role

Act as a combined principal research scientist, systems architect, Rust/Python/WASM engineer, benchmark methodologist, security engineer, DevOps/release engineer, open-source product leader, and adversarial reviewer.

Your job is not to make Entroly *sound* superior. Your job is to make it measurably superior where possible, expose where it is not, and close those gaps without fabricated claims, benchmark leakage, hidden fallbacks, or fragile demos.

Treat Entroly as an **auditable context-assurance control plane**, not merely a token compressor.

## Mission

Make Entroly Pareto-competitive or dominant against the latest public versions of Headroom, LeanCTX, and other serious context-compression/context-OS systems across:

1. answer and task quality;
2. token and monetary savings;
3. compression latency and throughput;
4. semantic safety and fail-closed behavior;
5. provenance, recovery, and auditability;
6. code, JSON, logs, shell, RAG, conversation, and tool-output coverage;
7. coding-agent task success;
8. installation, integrations, and developer experience;
9. reliability, memory use, security, and release quality;
10. honest evidence, reproducibility, and public trust.

A win is not a larger README number. A win is a reproducible frontier improvement under a neutral harness.

## Absolute operating rules

- Read `AGENTS.md`, `CLAUDE.md`, release instructions, and affected contracts before editing.
- Re-check the latest official competitor repositories, releases, documentation, and benchmarks at execution time. Never rely on stale version memory.
- Never copy competitor code unless its license and attribution permit it.
- Never fabricate a benchmark, citation, result, test run, integration, receipt, or release status.
- Never optimize against test answers, expected spans, hidden labels, benchmark-specific filenames, or dataset IDs.
- Never hide quality loss behind aggregate averages.
- Never call a heuristic a guarantee.
- Never call a retrieval-score proxy a semantic answer-preservation certificate.
- Preserve local-first behavior. Introduce no surprise remote calls.
- Preserve receipt honesty, reversibility, WITNESS/RAVS fail-closed behavior, source hashes, source offsets, and native-status truth.
- Do not merge or release while required CI is red.
- Do not weaken public contracts merely to make a benchmark green.
- If a requested budget conflicts with answer preservation, expose the conflict explicitly. Do not silently violate one contract while claiming the other.
- Keep Python, Rust, PyO3, WASM, npm, Docker, Homebrew, MCP, docs, and version surfaces aligned when those surfaces change.

## Definition of victory

For each workload family, plot and publish:

- task quality versus emitted context tokens;
- task quality versus total provider input tokens;
- task quality versus total API tokens and cost;
- savings versus compression latency;
- accepted-compression error versus coverage;
- p50/p95/p99 latency and memory;
- exact recovery and receipt verification rates.

Entroly wins a dimension only when one of these is true under the same inputs, model, prompts, seeds, budgets, hardware, warm-up policy, and scorer:

- higher quality at equal tokens;
- fewer tokens at equal quality;
- lower latency at equal quality and savings;
- lower certified error at equal certified coverage;
- stronger recovery/audit guarantees with no material frontier regression.

Report losses and ties as clearly as wins.

## Phase 0 — Establish truth before invention

1. Inspect the current branch, open PR, uncommitted changes, and CI.
2. Build an import/call graph from public entry points to:
   - SDK compression;
   - proxy;
   - MCP;
   - QCCR;
   - Rust selector;
   - receipts;
   - recovery store;
   - benchmarks.
3. Identify dead or test-only implementations that are advertised but not user-reachable.
4. Inventory every token counter and define what it measures:
   - raw context;
   - selected context before serialization;
   - emitted context;
   - full prompt input;
   - generated output;
   - provider total.
5. Record exact competitor versions and commit SHAs.
6. Create a falsification ledger: for every proposed advantage, state the smallest experiment that would disprove it.

Do not begin broad feature work until the measurement path is trustworthy.

## Phase 1 — Repair benchmark epistemics

Implement a benchmark schema that records per sample:

```json
{
  "raw_context_tokens": 0,
  "selected_tokens_pre_trim": 0,
  "emitted_context_tokens": 0,
  "prompt_input_tokens": 0,
  "output_tokens": 0,
  "provider_total_tokens": 0,
  "baseline_correct": false,
  "treatment_correct": false,
  "answer_present_raw": null,
  "answer_present_pre_trim": null,
  "answer_present_post_trim": null,
  "compression_decision": "",
  "certificate_scope": "",
  "certificate_verdict": "",
  "source_spans": [],
  "seed": 0,
  "model": "",
  "error": null
}
```

Requirements:

- Identity pass-through when input already fits the budget.
- No negative-savings result may be hidden or clamped; classify its cause.
- Use paired samples and paired statistical tests.
- Include Wilson intervals for proportions and bootstrap intervals for frontier differences.
- Separate calibration, development, and final holdout sets.
- Repeat stochastic model calls or use deterministic settings where supported.
- Preserve raw audit artifacts for every regression.
- Never use provider `total_tokens` as the compressor's emitted-token measurement.

## Phase 2 — Build a real sufficiency certificate

The certificate must be generated from exact optimizer state, not reconstructed from incompatible inputs and outputs.

Modify the Rust single source of truth to emit, for every candidate unit considered:

```text
unit_id
source_id
utility
cost_tokens
selected
selection_stage
start_byte
end_byte
start_token
end_token
trimmed
neighbourhood_ids
query_anchor_ids
```

Expose this identically through PyO3 and WASM.

Calculate and receipt:

- captured positive utility;
- best excluded utility density (shadow price);
- residual risk;
- cutoff ambiguity;
- original-query IDF coverage;
- boundary exposure;
- budget saturation;
- stability under small deterministic perturbations;
- source-span integrity;
- signal availability;
- calibration version and dataset fingerprint;
- certificate scope.

Allowed verdicts:

- `sufficient`;
- `degraded`;
- `uncertain`.

`uncertain` is mandatory when a required signal or valid calibration is absent.

Never promote a certificate beyond its scope. Examples:

- file-ranking evidence → `file_retrieval`;
- exact selected sentence units → `candidate_units`;
- calibrated answer-retention evidence → `semantic`;
- verified task oracle evidence → `task_verified`.

## Phase 3 — Atomic span-safe extraction

Eliminate arbitrary final substring trimming.

1. Segment with exact UTF-8 byte offsets.
2. Form atomic evidence neighbourhoods:
   - anchor sentence;
   - required predecessor/continuation;
   - structured field/value pair;
   - code signature plus relevant body;
   - log error plus causal stack neighbourhood.
3. Pack atomic units, not arbitrary character slices.
4. Never cut accepted units midway.
5. Preserve original source order unless an ablation proves reordering helps.
6. Emit omitted-neighbour evidence in the receipt.
7. When an atomic unit cannot fit:
   - expand;
   - choose an alternate complete unit;
   - or return `uncertain`/bypass.

Test specifically on answer spans crossing chunk and sentence boundaries.

## Phase 4 — Guarded adaptive controller

Implement caller policy outside the selector:

1. **BYPASS_ALREADY_FITS**
2. **COMPRESSED_CERTIFIED**
3. **EXPANDED_CERTIFIED**
4. **BYPASS_UNCERTIFIED**
5. **UNCERTIFIED_BUDGET_ENFORCED**

Policy:

```text
identity if raw <= budget
otherwise select
accept only if verdict and scope satisfy caller policy
otherwise expand by a bounded deterministic schedule
otherwise return original in quality-first mode
or return selected with an explicit uncertified receipt in hard-budget mode
```

Record whether fallback violates the requested compression budget. Never call this “budget compliant.”

## Phase 5 — Win workload-specific compression

Build specialized, composable compressors with shared receipts:

- JSON/JSONL: schema-aware key/value preservation, repeated-key dictionaries, validity check.
- Logs: deduplicate templates while preserving timestamps, IDs, error transitions, and stack roots.
- Shell output: preserve command, exit status, errors, paths, changed files, and final summaries.
- Code: AST/symbol-aware atomic units, dependency edges, exact line/byte offsets.
- RAG prose: atomic propositions and answer-neighbourhood extraction.
- Conversation: typed commitments, constraints, decisions, unresolved questions, and provenance.
- Tool schemas: stable-prefix deduplication and reversible references.

Every specialized compressor must have:
- a pass-through threshold;
- a validity checker;
- a recovery path;
- a domain-specific retention oracle;
- a generic fallback.

## Phase 6 — Performance engineering

Profile before optimizing.

Measure:
- Rust time by stage;
- Python↔Rust serialization;
- allocation count;
- peak RSS;
- cache hit rate;
- index size;
- p50/p95/p99;
- cold and warm runs.

Targets:
- no super-linear scan on large repositories without a documented bound;
- no duplicate ranking pass solely for receipts;
- streaming/zero-copy boundaries where practical;
- deterministic output across seeds, threads, file order, Python versions, and platforms;
- bounded caches with corruption and oversize defenses;
- pure-Python surface remains import-safe;
- native absence is explicit, never silently presented as accelerated behavior.

## Phase 7 — Product and integration parity

Audit and harden:

- Claude Code;
- Codex CLI/Desktop/Cloud where applicable;
- Cursor;
- VS Code/Copilot;
- OpenClaw;
- Hermes;
- Aider;
- Cline;
- Continue;
- Goose;
- OpenHands;
- OpenCode;
- MCP stdio/SSE/HTTP;
- OpenAI-compatible proxy;
- Anthropic and Gemini routing;
- Docker;
- PyPI;
- npm/WASM;
- Homebrew;
- Windows/macOS/Linux.

For each integration provide:
- one-command install;
- doctor check;
- uninstall/rollback;
- no-secret-leak test;
- timeout behavior;
- binary mismatch detection;
- minimal end-to-end test.

Do not advertise an integration that only has a dormant adapter or README snippet.

## Phase 8 — Neutral competitor gauntlet

Run exact public competitor versions in isolated environments.

Workloads:

- GSM8K;
- SQuAD 2.0;
- BFCL;
- LongBench/HotpotQA;
- needle retrieval;
- JSON and JSONL;
- logs and shell output;
- tool schemas;
- repository localization;
- multi-file code questions;
- SWE-bench-style tasks;
- adversarial distractors;
- answer spans at boundaries;
- multilingual/CJK;
- repeated long-running conversations;
- crash/restart recovery.

Controls:

- raw context;
- truncation;
- BM25/MMR baseline;
- Entroly;
- Headroom;
- LeanCTX;
- relevant learned compressor;
- hybrid where fair.

Publish all configurations, prompts, exclusions, failures, and artifacts.

## Phase 9 — Security, receipts, and recovery dominance

Prove:

- byte-exact source recovery;
- content-addressed integrity;
- no path traversal;
- no secret exfiltration;
- safe handling of malformed JSON, Unicode, binary, and huge files;
- bounded disk and memory;
- corrupt receipt detection;
- expired reference semantics;
- concurrent writer safety;
- crash consistency;
- deterministic receipt verification across Python/Rust/WASM.

Threat-model prompt injection inside compressed content and adversarial relevance manipulation.

## Phase 10 — Documentation and positioning

Position Entroly as:

> The evidence-assurance layer for AI context: it saves tokens only when the required evidence can be preserved or explicitly reports that it cannot, with recoverable, auditable receipts.

Do not claim:
- “zero quality loss” without a bounded test domain;
- “guaranteed answer preservation” from heuristic scores;
- “beats X” without same-harness evidence;
- “up to 90%” without naming workload and quality result.

The README first screen must state:
- what Entroly does;
- why it differs;
- installation;
- one verifiable example;
- current benchmark caveats;
- links to raw artifacts.

## Required quality gates

A release candidate is blocked unless:

- targeted tests pass;
- full required CI is green;
- lint/clippy are clean;
- native and pure-Python installation paths are tested;
- no negative-savings regression remains unexplained;
- no known benchmark regression is hidden;
- certificate scope is honest;
- all fallback decisions are visible;
- package versions agree;
- release artifacts install and run;
- rollback instructions exist.

For a public “wins semantic safety” claim require:

- held-out calibration;
- at least two model families;
- multiple dataset families;
- certified accuracy retention target defined in advance;
- useful certified coverage;
- no leakage from labels/answers;
- reproducible raw artifacts.

## Implementation discipline

Work in small reviewable commits. Each commit message must state:

- user-visible problem;
- root cause;
- exact behavior change;
- contracts intentionally unchanged;
- tests run;
- known limitations.

Do not combine research instrumentation, algorithm changes, broad refactors, docs, and release bumps in one unreviewable commit.

After every implementation slice, perform an adversarial review:

- What could make this result look better than reality?
- Which field is mislabeled?
- Which signal is unavailable but represented as zero?
- Which fallback violates a different contract?
- Which test only validates synthetic shapes?
- What would a competitor engineer attack first?
- What would falsify the claimed advantage?

## Immediate execution order for the current sufficiency work

1. Fix benchmark token accounting.
2. Stop compression for already-fitting inputs.
3. Make ignored threshold arguments effective.
4. Introduce `uncertain` and certificate scope.
5. Prevent missing boundary data from appearing as zero exposure.
6. Add guarded accept/expand/bypass orchestration.
7. Add exact optimizer candidate/span telemetry in Rust.
8. Replace final arbitrary trimming with atomic units.
9. Instrument and rerun the four known SQuAD regressions.
10. Calibrate only after a sufficiently large development set.
11. Freeze a holdout and compare against current Headroom and LeanCTX.
12. Integrate guarded mode into SDK/proxy only after coverage and latency are acceptable.

## Final output required from the implementing agent

Return:

1. current branch and commit SHAs;
2. files changed;
3. architecture and data-flow changes;
4. exact tests and their results;
5. benchmark results with uncertainty;
6. competitor versions and commands;
7. known losses and limitations;
8. security/recovery impact;
9. rollback procedure;
10. next falsification experiment.

End with one of:

- `READY FOR REVIEW`
- `NOT READY — <specific blockers>`

Never end with a vague success statement.
