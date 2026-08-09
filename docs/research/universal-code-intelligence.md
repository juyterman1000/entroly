# Universal code intelligence research contract

This branch treats programming-language breadth as an architectural property,
not a fixed language-count claim.

## Core invariants

1. **One semantic IR.** Grammar-specific node names stop at the frontend.
2. **Progressive evidence.** Exact source is always available; parser, static,
   runtime, inferred, and learned evidence remain distinguishable.
3. **Registry-driven breadth.** The installed grammar registry is authoritative;
   Entroly's suffix table is only an offline fallback.
4. **Unknown-language survival.** Code-like files that predate registry support
   receive a bounded exact-source structural skeleton instead of disappearing.
5. **No partial-tree promotion.** Hitting traversal bounds invalidates parser
   completeness; partial AST results are not silently promoted to truth.
6. **Zero-friction default.** Repository parsing is part of the base install and
   missing grammars may be acquired lazily by default.
7. **Air-gap dominance.** `ENTROLY_AIR_GAP=1` always disables parser acquisition.
   `ENTROLY_TREE_SITTER_ALLOW_DOWNLOAD=0` is an explicit local-only override.
8. **Bindings are stronger than syntax.** A parsed call expression proves the
   spelling and source span, not the semantic callee. Binding requires stronger
   static/compiler/LSP evidence.
9. **Language count is not the metric.** Conformance is measured by semantic
   levels: source, syntax, structure, binding, flow, and transformation.
10. **Claims follow execution.** A language is not described as verified at a
    semantic level until its frozen conformance fixture passes at that level.

## Semantic levels

- **L0 source:** exact bytes, ranges, digest, bounded fallback.
- **L1 syntax:** grammar parse and syntax diagnostics.
- **L2 structure:** normalized declarations/containment/call-expression spans.
- **L3 semantics:** verified definitions/references/types/dispatch.
- **L4 flow:** CFG, definition-use, argument/return and interprocedural flow.
- **L5 transformation:** committed plans with source preimages, proof obligations,
  staged validation, and transaction-safe application.

The long-term architecture must let compiler, LSP, build-system, runtime,
history, coverage, and learned proposal adapters strengthen this same graph
without changing the agent interface.
