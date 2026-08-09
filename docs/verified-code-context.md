# Verified code context

Entroly repository intelligence can now return a task-specific partial code
graph whose source fragments and resolved call edges are independently
verifiable. The feature is local-first and performs no network calls.

```powershell
python -m entroly.repository_intelligence --root . context `
  --query "where is payment authorization performed" `
  --token-budget 2000 `
  --max-hops 2 `
  --include-history
```

The same operation is exposed by the focused repository MCP server as
`repository_verified_context`.

For an architectural overview rather than one task neighborhood, the `map`
command and `repository_map` MCP tool run deterministic personalized PageRank
over typed file, containment, import, and resolved-call edges:

```powershell
python -m entroly.repository_intelligence --root . map `
  --query "payment authorization" `
  --token-budget 2000 `
  --max-entries 100
```

With no query, the restart distribution is uniform and dependency/call hubs
rise globally. With a query, 85% of restart weight follows exact lexical
relevance while 15% preserves the repository backbone. Every returned anchor
is an exact source signature with a fresh file digest, byte-span digest,
stable weak-component identity, transparent score, and token estimate. A
stale or non-unique signature is omitted rather than guessed. The receipt can
be checked offline with `verify_repository_map_commitment()`.

For a known symbol, the `graph` command and `repository_symbol_graph` MCP tool
trace bounded static callers, callees, or both:

```powershell
python -m entroly.repository_intelligence --root . graph `
  --symbol "payments.service.charge_card" `
  --direction both `
  --max-depth 3
```

Short names are accepted only when they identify exactly one indexed symbol.
Ambiguous matches return candidates and no graph. Before traversal, Entroly
rechecks source-file hashes and every traversed call-site span; stale or
hash-mismatched evidence is omitted. The graph receipt can be checked with
`verify_symbol_graph_commitment()`.

Python functions and methods also expose a verified intraprocedural program
graph:

```powershell
python -m entroly.repository_intelligence --root . program `
  --symbol "payments.service.charge_card"
```

The graph models branches, loops, jumps, returns, raises, `with`, `match`, and
`try`/handler paths. A fixed-point reaching-definition analysis labels an edge
`must-reach` only when the variable is definitely defined on every modeled
predecessor and has one reaching definition; branch-dependent definitions are
`may-reach`. Every source node and variable occurrence has an exact byte span
and digest. Unsupported bindings remain diagnostics or unresolved uses.

External tracing and coverage tools can contribute value-free runtime events:

```powershell
python -m entroly.repository_intelligence --root . runtime `
  --events-json trace-events.json `
  --producer pytest
```

Only workspace-relative path, line, event kind, and count are accepted. Values
and exception payloads are discarded. Events are aggregated and attached to
the narrowest enclosing symbol only after the source-file and line-span hashes
verify. Entroly does not execute the project to obtain these observations.

Large repositories may opt into content-addressed persistence with the global
`--cache-dir` option. Per-file parse keys commit to workspace-relative path,
source digest, parser environment, Python version, and schema. An immutable
whole-index snapshot additionally commits to the complete source manifest,
parser environment, and resource limits; it is checkout-root independent.
Deterministic repository-map and health results are cached by stable index
digest plus all request parameters, and both their native receipt and an outer
cache commitment are verified before reuse. Corrupt entries fail open and are
atomically rebuilt.

Any source change produces a new manifest and rebuilds global import/call
resolution. This is exact persistent graph and derived-analysis reuse, not a
claim of fine-grained incremental name resolution. File discovery and hashing
also still run so deleted or changed files cannot hide behind cached state.

Structural health is available from the same immutable repository snapshot:

```powershell
python -m entroly.repository_intelligence --root . health `
  --max-findings 500 `
  --max-symbols 2000
```

The `repository_code_health` MCP tool exposes the same bounded report. Python
metrics come from the standard AST; other languages are profiled only when a
local tree-sitter grammar produced an exact declaration span. The report
includes cyclomatic decision count, a language-neutral cognitive-complexity
approximation, control nesting, parameter and symbol-size thresholds, import
strongly connected components, coupling, parser coverage, and unresolved-call
reasons. Every symbol profile and finding carries its source and evidence
digests. Files changed since indexing are omitted as `stale-index`.

The grade is intentionally reproducible policy, not an oracle. The response
publishes its thresholds and full formula, labels findings as review aids, and
commits the entire report with `verify_code_health_commitment()`. A hash proves
which bytes were analyzed; it does not prove that a threshold violation is a
bug or that an unreferenced symbol is dead.

## What is verified

Each selected fragment records:

- a workspace-relative path and qualified symbol name;
- exact UTF-8 byte and line boundaries;
- the indexed source-file SHA-256;
- the emitted fragment SHA-256;
- the parser backend and selected resolution (`full` or `signature`);
- the query score and graph path that caused selection.

Each resolved call edge records its binding policy and the exact byte range and
digest of the call-site evidence. If multiple repository definitions are
plausible, Entroly records an unresolved call with candidate IDs instead of
inventing an edge. Immediately before returning source, the file is re-read and
checked against the indexed hash. Changed files are omitted with a visible
`stale-index` reason.

For Python member calls, annotations, constructor assignments, local type
propagation, and `self` can select a concrete class member. An untyped receiver
is never bound merely because one same-named method exists; same-file member
candidates are returned as `untyped-receiver-member` negative evidence.

The receipt commits to the query, fragments, relationships, unresolved
evidence, retrieval policy, token estimate, and omissions. Operational
`generation` and CLI `command` fields are intentionally outside that
commitment. `verify_context_commitment()` detects payload tampering without
needing the workspace.

Source code and commit text are always untrusted input. A hash proves identity,
not safety or correctness.

## Retrieval design

The selector combines lexical entry-point scoring with a bounded, query-time
partial graph. It expands exact containment, call/caller, and resolved file
dependency relationships under a token budget. Oversized symbols degrade to an
exact signature slice; unverifiable or stale slices fail closed. The
whole-repository map complements this task graph with typed personalized
PageRank; it ranks only relationships the index actually resolved and never
promotes unresolved calls.

History retrieval is explicit opt-in. When requested, Entroly runs a bounded
local `git log` over selected workspace-relative paths with optional Git locks
disabled. It never fetches. Commit IDs, timestamps, and subjects are committed
into the receipt and labeled untrusted Git metadata.

This design incorporates several independently demonstrated directions:

- [Repoformer (ICML 2024)](https://proceedings.mlr.press/v235/wu24a.html):
  retrieval should be selective because irrelevant context can harm generation.
- [RepoGraph (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/4a4a3c197deac042461c677219efd36c-Paper-Conference.pdf):
  line-level definition/reference graphs improve repository reasoning.
- [LocAgent (ACL 2025)](https://aclanthology.org/2025.acl-long.426/):
  heterogeneous multi-hop graph navigation improves code localization.
- [Code Graph Model (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/178ae4ba29022eb7bf509c2e27bc8ab8-Abstract-Conference.html):
  semantic and structural dependencies materially improve repository tasks.
- [Repository Memory (ICLR 2026)](https://openreview.net/forum?id=8yjWLJy2eX):
  historical repository knowledge improves localization over stateless search.
- [DyRetriever (ASE 2026)](https://arxiv.org/abs/2608.01927): query-time
  partial dependency graphs can outperform costly static-graph retrieval.
- [Stack graphs](https://arxiv.org/abs/2211.01224): file-incremental graph
  construction can support precise name binding without executing untrusted
  project builds.

Entroly's new contribution is not another graph representation. It is the
evidence contract around graph-assisted context: freshness, exact source
identity, explicit ambiguity, budgeted omissions, and a deterministic receipt.

## Evidence and limitations

The preregistered local benchmark is documented in
`benchmarks/VERIFIED_CODE_CONTEXT_PREREGISTRATION.md`. It currently exercises
Python, Rust, TypeScript, Go, and Java relationships, ambiguity, deterministic
receipts, fragment and graph-edge evidence, exact-name graph ambiguity, and
stale-source failure. It does not measure LLM answer quality or competitor
superiority.

Parser grammars remain optional. Python uses its standard AST; other supported
languages use cached tree-sitter grammars when available and conservative
fallbacks otherwise. Python receiver annotations, constructor assignments,
local propagation, and `self` dispatch are type-informed static inference, not
compiler/LSP proof. Verified control/data flow is currently Python-only;
runtime evidence must be supplied by an external tracer. Persistence reuses an
unchanged whole graph and exact derived results, but any source change rebuilds
global relationships and affected derived analyses. PageRank is not maintained
with a fine-grained dynamic-graph algorithm.
Interprocedural data flow,
broader-language program graphs, and broad external task-quality benchmarks
remain future work and must not be claimed as implemented.

Structural health does not yet implement compiler-specific cognitive
complexity standards, whole-program dead-code proof, semantic clone detection,
or automatic refactoring. Import cycles are computed only from resolved index
edges, and unresolved-call rate is shown separately so missing semantic edges
cannot masquerade as a clean graph. On an August 8, 2026 local dogfood run of
this checkout (930 files; 12,778 symbols; query `persistent verified graph
architecture benchmark`; 2,000-token/100-entry map; 500-finding/2,000-profile
health limits), initial derived computation took 1.20 seconds for the map and
7.28 seconds for health. Across the next four unchanged runs, median verified
cache reuse took 0.0023 and 0.0840 seconds respectively; warm index-load median
was 2.69 seconds. These are environment-specific engineering measurements,
not universal performance claims.

External language servers and compiler indexers can supply definition,
declaration, implementation, reference, type-definition, and override
relationships through the `semantic` command or `repository_semantic_overlay`
MCP tool. Entroly interprets positions using the LSP UTF-16 convention,
converts them to exact UTF-8 byte ranges, and verifies both endpoints against
fresh indexed source. The provider remains explicitly untrusted: this proves
which source spans it connected, not that its semantic conclusion is correct.

## Two-phase verified rename

Rename preview is no-write by construction:

```powershell
python -m entroly.repository_intelligence --root . rename-preview `
  --symbol "payments.service.charge_card" `
  --new-name authorize_card > rename-plan.json
```

The plan resolves exactly one symbol, rechecks source freshness, and emits exact
identifier byte ranges for its definition, resolved calls, and Python direct
import bindings. Optional `--semantic-json` relationships can add external
LSP/compiler reference ranges only after the semantic overlay verifies both
endpoints. The plan commits every preimage and edit, performs zero writes, and
reports unresolved same-name calls plus remaining lexical occurrences for
review. Reference completeness remains `not-proven` even when a provider is
present.

Apply is a separate, explicit operation:

```powershell
python -m entroly.repository_intelligence --root . rename-apply `
  --plan-json rename-plan.json `
  --expected-plan-sha <receipt.plan_sha256> `
  --acknowledge-incomplete
```

Before writing, Entroly verifies the plan commitment, current index identity,
every file digest, exact identifier preimage, non-overlapping ranges, and staged
Python or available tree-sitter syntax. New files are staged beside their
targets. If a replacement fails, Entroly attempts to restore every completed
file from a same-directory backup and reports the failed transaction. A
successful service/MCP apply immediately rebuilds the repository snapshot.

This is not equivalent to compiler-complete refactoring. Dynamic lookup,
reflection, strings, generated code, macro expansion, non-call references, and
external consumers may remain. The mandatory acknowledgement exists because a
hash can make an incomplete plan tamper-evident but cannot make it complete.
