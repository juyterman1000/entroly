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
exact signature slice; unverifiable or stale slices fail closed.

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
fallbacks otherwise. Compiler/LSP-grade type-directed dispatch, control/data
flow, runtime traces, incremental persistence, and broad external benchmarks
remain future work and must not be claimed as implemented.
