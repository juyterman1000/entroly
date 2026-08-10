# Verified code-context benchmark preregistration

This benchmark measures the narrow properties introduced by
`entroly.repository_intelligence` schema v2. It is not a benchmark of model
answer quality and does not claim superiority over another product.

## Frozen workload

- Languages: Python, Rust, TypeScript, Go, and Java.
- Each language fixture contains a caller and callee with a known relationship.
- The Python fixture also contains two same-named definitions and one unbound
  call; a correct graph must not invent either definition as the callee.
- Context queries use only public symbol names and task words, never function
  bodies.
- Context budget: 512 estimated tokens; graph depth: 2.

## Metrics

1. `gold_edge_recall`: fraction of declared caller/callee pairs found.
2. `edge_evidence_validity`: fraction of returned gold edges whose byte slice
   hashes to the edge's evidence digest.
3. `ambiguous_call_truthfulness`: 1 only when the ambiguous call has no
   resolved edge and is present in the unresolved evidence set.
4. `query_symbol_recall`: fraction of public-name queries that return the named
   symbol in their verified context slice.
5. `fragment_evidence_validity`: fraction of emitted fragments whose content,
   source hash, and fragment hash all verify against the fixture bytes.
6. `deterministic_receipt`: 1 only when identical requests over an unchanged
   snapshot return identical payloads.
7. `stale_source_fail_closed`: 1 only when a post-index source mutation is not
   emitted and is recorded as a `stale-index` omission.
8. `symbol_graph_evidence_validity`: 1 only when the exact-name graph resolves,
   every returned call span verifies, and its deterministic receipt verifies.
9. `symbol_graph_ambiguity_truthfulness`: 1 only when a duplicate short name
   returns candidates but no invented graph.
10. `symbol_graph_stale_source_fail_closed`: 1 only when a post-index mutation
    prevents the stale root symbol from producing a graph.

## Failure accounting

All cases remain in the denominator. Exceptions score the affected metric as
zero and are written to the result artifact. Token estimates are explicitly
not provider billing records. Competitor comparisons require running their
unchanged public implementations on an equivalent workload and are outside
this benchmark.
