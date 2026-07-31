# SKILLS.md

Repeatable workflows for working on Entroly. Each skill states when to reach for
it, the exact commands, and what a passing result looks like.

Entroly is an auditable context-control plane, so most skills end in *evidence*,
not in a green checkmark. A skill is done when you can hand someone an artifact
they can re-run.

> Read `CLAUDE.md` first for the trust invariants and the codebase graph. Read
> `AGENTS.md` if you are driving Codex.

---

## Skill: Orient in the codebase (graph first)

**When:** starting any non-trivial change in an unfamiliar area.

Do not begin by reading files. 112k lines of Python across 215 modules will not
fit in a useful mental model. Build the graph, find the hubs, then read only
what the graph says matters.

```bash
python scripts/codebase_graph.py                    # hubs, cycles, reachability
python scripts/codebase_graph.py --json graph.json  # full adjacency for querying
```

**Read in this order:**

1. The PageRank hubs — highest blast radius (`context_receipts.models`,
   `path_safety`, `esg`, `compression_retrieval_store`, `vault`, `ccr`).
2. The module you are changing.
3. Its inbound edges (who breaks if you change the signature).

**Check before you claim a surface works:** is your module in the reachable set?
44 modules are reachable only from tests and benchmarks. A passing test does not
prove a user can reach the code.

```bash
python -c "import json;g=json.load(open('graph.json'));print('entroly.YOURMOD' in g['unreachable'])"
```

---

## Skill: Verify a public surface actually runs

**When:** before claiming any SDK/MCP/CLI capability in docs or a README.

Import-level success is not a product path. Exercise the real signature and
watch what loads.

```bash
python -c "
import sys
from entroly import sdk
r = sdk.create_context_receipt(
    [('a.py', open('entroly/esg.py').read())],
    query='how is evidence coverage enforced', budget=900, recoverable=True)
print('selected', len(r['selected_context']), 'omitted', len(r['omitted_context']))
print('ratio', r['compression_ratio'])
"
```

Real SDK signatures (do not guess — they differ from the docs in places):

| Function | Signature |
|---|---|
| `compress` | `(content, budget, content_type, target_ratio, profile)` |
| `compress_messages` | `(messages, budget, preserve_last_n, model, client_key, distill, profile, target_ratio)` |
| `optimize` | `(fragments, budget, query)` |
| `create_context_receipt` | `(documents, query, budget, chunk_tokens, overlap_tokens, prefer_rust, recoverable)` |
| `recover_receipt_omission` | `(receipt, chunk_id, *, store_dir)` → **list** of dicts, not one dict |

Regenerate this table any time `sdk.py` changes:

```bash
python - <<'PY'
import ast
# encoding is required: several modules contain non-cp1252 characters
src = open('entroly/sdk.py', encoding='utf-8').read()
for n in ast.parse(src).body:
    if isinstance(n, ast.FunctionDef) and not n.name.startswith('_'):
        print(n.name, '(' + ', '.join(a.arg for a in n.args.args) + ')')
PY
```

---

## Skill: Falsify a trust claim before shipping it

**When:** any claim about exactness, recovery, determinism, or coverage.

The invariant: **the oracle must be the source of truth, never the system's own
output.** A test that compares a component against its own earlier output proves
round-tripping, not correctness.

Checklist for an exactness claim:

1. Compare against the **original bytes on disk**, not an intermediate structure.
2. Use **real repository files**, not synthetic strings. Synthetic input often
   lacks the structure that breaks — indentation, comments, blank lines, Unicode.
3. Report a **denominator** (`n/N`), not a boolean.
4. Vary the input class (code vs prose vs config) and report per class; an
   aggregate can hide a total failure in one class.

Worked example — recovery fidelity, measured against files on disk:

```bash
python - <<'PY'
from pathlib import Path
from entroly.context_receipts.ingest import ingest_documents
text = Path('entroly/esg.py').read_text(encoding='utf-8')
idx = ingest_documents([('entroly/esg.py', text)])
n = len(idx.chunks)
verbatim = sum(1 for c in idx.chunks if c.text in text)
print(f'chunks verbatim in source: {verbatim}/{n}')
PY
```

If that ratio is not `n/n`, exact recovery does not hold for that input class,
whatever the receipt's `verified: true` field says.

---

## Skill: Classify a failure before fixing it

**When:** anything goes red.

Never patch to green. Name the class first — the fix differs completely:

| Class | Signature | Correct response |
|---|---|---|
| **Product defect** | Real users hit it through a shipped entry point | Fix the code; add a test that would have caught it |
| **Test defect** | Test asserts something the product never promised, or its oracle is self-referential | Fix the oracle; do not relax the assertion |
| **Environment defect** | Missing native wheel, locked file, no `$HOME` | Fix setup or fall back explicitly; never silently skip |
| **Benchmark defect** | Dataset construction leaks the answer, or baseline is stale | Rebuild the dataset; re-pin the baseline |
| **Dependency defect** | Upstream change | Pin, then upgrade deliberately |

A backend may honestly return no relevant result. Tests must not demand
unrelated context merely to stay green.

---

## Skill: Run the right tests for the surface you touched

**When:** always, scoped to the blast radius from the graph.

```bash
# targeted first — fast signal
pytest tests/test_context_receipts.py tests/test_recoverable_receipts.py -v --tb=short

# whole suite
pytest tests/ -v --tb=short --timeout=60

# Rust
cd entroly-core && cargo test --lib && cargo clippy --all-targets -- -D warnings

# lint
python -m ruff check entroly/ scripts/
```

Rust changes need `maturin develop --release` before Python tests observe them.
Otherwise you are testing the previously built wheel and the result is
meaningless.

Only expand to the release matrix if packaging/native surfaces changed:

```bash
python -m build && python -m twine check dist/*
python -m pip install --force-reinstall dist/*.whl
entroly doctor
```

---

## Skill: Add a benchmark that survives scrutiny

**When:** producing any number that might reach the README.

Preregister before running. `benchmarks/*_PREREGISTRATION.md` is the existing
pattern — follow it.

Required in the artifact, not just the write-up: sample size, seeds, hardware,
model + version pins, token budget, success oracle, exclusions with reasons,
confidence interval, and the exact command.

```bash
python -m benchmarks.<name> run   --out benchmarks/results/<name>.json
python -m benchmarks.<name> verify benchmarks/results/<name>.json
```

Every benchmark needs that `verify` mode: a third party must be able to check
the committed artifact without re-running the experiment.

Claim discipline:

- Separate **observed**, **modeled**, and **inferred**. Modeled savings are never
  reported as money saved.
- A percentage without a denominator is not a result.
- Never compare Entroly's branch against a stale competitor version.
- Token reduction alone is not a headline. Pair it with a task-success or
  evidence-retention oracle at a matched budget.

---

## Skill: Ship a release without drift

**When:** bumping any version.

The typed synchronizer governs 25 explicit release surfaces. `entroly/__init__.py`
is the runtime master, and every governed Python, Rust, npm, MCP, OpenClaw, WASM,
plugin, and release-test surface must converge before tagging.

```bash
python scripts/sync_release_version.py <new-version>
python -m pytest -q tests/test_release_version_sync.py tests/test_release_surface.py
```

The authoritative allowlist and typed transform for every surface live in
`scripts/sync_release_version.py`; do not replace it with repository-wide text
substitution. Historical release notes and workflow definitions are
intentionally excluded.

Homebrew is **post-publish**: verify the PyPI sdist URL and SHA-256 from the
PyPI JSON API before touching the formula. The synchronizer updates the
Homebrew runbook but deliberately leaves the live formula pinned until that
artifact exists.

---

## Skill: Keep receipts honest

**When:** touching selection, chunking, receipts, WITNESS, RAVS, or recovery.

The non-negotiables. A change that weakens any of these does not ship:

- **Receipt honesty** — selected context, omitted evidence, risks, hashes and
  token ratios stay inspectable.
- **Reversibility** — compressed context traces back to source spans.
- **Fail-closed verification** — degrade visibly; never silently claim confidence.
- **Local-first** — no surprise remote calls for ranking, receipts, or diagnostics.
- **Cache stability** — prompt prefixes stay byte-stable unless changed on purpose.
- **Fingerprints must verify** — a consumer holding the receipt and the recovered
  bytes must be able to recompute the fingerprint and have it match. A hash over
  an internal normalized form is not an audit trail.

Ask on every receipt change: *could a user recompute this field themselves from
public data?* If not, it is decoration, not evidence.
