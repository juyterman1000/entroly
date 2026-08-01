# Evidence-Assured Context Compression

Entroly's assured path separates two questions that ordinary compressors often
mix together:

1. **What fits the budget?** The selector ranks and packs evidence.
2. **May the caller trust the compressed result?** The guard accepts, expands,
   bypasses, or explicitly returns an uncertified hard-budget result.

The compatibility SDK remains unchanged. The assured path is opt-in until
held-out calibration and same-harness competitor evidence justify wider use.

## Python SDK

```python
from entroly import compress_assured

result = compress_assured(
    long_context,
    query="Where is payment retry state persisted?",
    budget=2_000,
    required_scope="candidate_units",  # structural evidence, not semantic proof
    fallback="original",              # quality-first
)

print(result.text)
print(result.receipt["decision"])
print(result.receipt["attempts"])
```

`required_scope="semantic"` is the default. Semantic acceptance requires a
validated held-out `CalibrationProfile`; without one, Entroly expands and then
returns the original context in quality-first mode.

### Conversation compression

```python
from entroly import compress_messages_assured

result = compress_messages_assured(
    messages,
    budget=32_000,
    preserve_last_n=4,
    required_scope="candidate_units",
)

messages = list(result.messages)
assert result.receipt["overall_delivered_tokens"] == result.delivered_tokens
```

Recent messages remain byte-identical. Older messages are represented as
separate source fragments, so receipts retain message identity and ordering.

### File-aware compression

```python
from entroly import compress_file_assured

result = compress_file_assured(
    "src/payments.py",
    workspace=repo_root,
    query="Where is retry state persisted?",
    budget=2_000,
    required_scope="candidate_units",
)
```

Instruction and rule files such as `AGENTS.md`, `CLAUDE.md`, `SKILL.md`, and
files under `.claude/rules/`, `.cursor/rules/`, `.github/instructions/`, or
`skills/` bypass compression and remain byte-identical. Small plain-text tool
outputs also pass through unchanged; large minified JSON remains eligible for
validated compression. Workspace escapes, non-UTF-8 files, and oversize files
fail with bounded validation errors.

## Local observability

The assurance ledger stores hashes, token counts, decisions, scopes, verdicts,
and timings. It never stores raw context or model output.

```python
from entroly import AssuranceLedger, compress_assured

ledger = AssuranceLedger(".entroly/assurance-ledger.sqlite3")
result = compress_assured(
    text,
    query="database timeout",
    budget=1_000,
    required_scope="candidate_units",
    ledger=ledger,
)

print(ledger.summary().to_dict())
```

The summary reports accepted coverage, bypass rate, identity rate, mean token
savings, and p50/p95 decision latency. Those are operational measurements, not
answer-quality claims.

## Focused MCP server

```bash
entroly-assurance-mcp
```

Tools:

- `assured_compress_text`
- `assured_compress_file`
- `assured_compress_messages`
- `validate_compressed_output`
- `assurance_stats`
- `repo_impact`
- `repo_overview`
- `repo_smells`
- `repo_context_bundle`

The repository tools use Python AST analysis plus conservative Rust and
JavaScript/TypeScript import extraction. They add deterministic dependency
importance, bounded structural smell findings, Unicode/CJK-aware query matching,
and full-fidelity instruction-file handling. Context bundles contain complete
source lines with exact file and line ranges. They do not claim complete semantic
impact analysis.

JSON validation accepts a single JSON value, JSON Lines, or whitespace-separated
JSON documents. Emitted summaries must remain valid JSON and preserve
query-relevant scalar evidence.

## Opt-in proxy assurance

The compatibility proxy keeps its existing behavior unless assurance is
explicitly enabled. Operators can select structural or semantic policy with
environment variables:

```bash
export ENTROLY_ASSURANCE_MODE=candidate_units  # or semantic
export ENTROLY_ASSURANCE_BUDGET_FRACTION=0.15
export ENTROLY_ASSURANCE_PRESERVE_LAST_N=4
export ENTROLY_ASSURANCE_FALLBACK=original
```

`semantic` mode additionally requires `ENTROLY_ASSURANCE_PROFILE` to reference
a current disjoint-holdout-validated calibration profile. An optional local
ledger can be configured with `ENTROLY_ASSURANCE_LEDGER`. Explicit token budgets
use `ENTROLY_ASSURANCE_BUDGET_TOKENS`; otherwise the configured fraction of the
model context window is used.

When enabled, the assured controller replaces the legacy lossy conversation and
tool-output pruning stages for that request. Unsupported multimodal or structured
message shapes, invalid profiles, initialization failures, and selector errors
return the original request unchanged. Bounded response headers expose the
decision, required scope, final certificate, budget compliance, and token counts;
raw exception messages and context are never placed in headers.

HTTP transport trust is independent from TLS trust. An explicit CA bundle is
used for certificate verification but does not enable ambient `HTTP_PROXY` or
`HTTPS_PROXY`. Environment proxy inheritance requires the separate explicit
`ENTROLY_TRUST_PROXY_ENV=1` opt-in.

## Certificate scopes

| Scope | What it supports | What it does not prove |
|---|---|---|
| `optimizer_proxy` | The optimizer's score distribution looks favorable | Exact answer survival |
| `file_retrieval` | Relevant source files were selected | Relevant passage survival |
| `candidate_units` | Exact atomic units and required neighbours survived | Model answer correctness |
| `semantic` | Held-out calibration supports a declared error target | Correctness outside the calibrated regime |
| `task_verified` | A task oracle verified the concrete outcome | Generalization to other tasks |

Unknown, missing, malformed, or weaker evidence becomes `uncertain`; it never
silently passes.

## Benchmark accounting

The benchmark surfaces now distinguish:

- raw context tokens;
- selected tokens before final emission;
- emitted context tokens;
- provider input tokens;
- provider output tokens;
- provider total tokens.

Compression savings use emitted context. Provider totals are reported
separately and are never relabelled as compressed-context size.

## Performance measurement

```bash
python -m bench.assurance_overhead \
  --input fragments.json \
  --query "payment retry" \
  --budget 2000 \
  --iterations 200
```

This reports compatibility-selector versus audited-selector p50/p95 latency,
receipt size, determinism, and budget compliance. Threshold flags can make the
command a CI gate, but thresholds should be calibrated on representative
hardware rather than copied from another machine.

## Fail-closed decisions

- `COMPRESSED_CERTIFIED`: accepted at the requested budget.
- `EXPANDED_CERTIFIED`: accepted only after a larger budget.
- `BYPASS_ALREADY_FITS`: identity was cheaper and safer.
- `BYPASS_UNCERTIFIED`: the original was returned.
- `UNCERTIFIED_BUDGET_ENFORCED`: a hard transport budget was obeyed, but the
  output is explicitly not certified.

## Current limitation

The implementation creates a strong structural and calibration contract. It
does **not** by itself establish that Entroly beats Headroom, LeanCTX, raw
context, or any other system. That claim requires pinned versions, byte-identical
inputs, disjoint calibration/holdout data, paired task outcomes, latency, and
failure accounting under the same harness.
