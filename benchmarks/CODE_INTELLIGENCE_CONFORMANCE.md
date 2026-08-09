# Code-intelligence conformance protocol

This protocol compares public code-intelligence behavior without converting
README breadth into an execution pass. It separates executed evidence from
source inspection and does not produce a single universal "best" score.

## Frozen implementations

Every run records the exact commit of Entroly and each unchanged upstream
checkout. An adapter may set environment variables or choose documented CLI,
SDK, or MCP arguments, but it may not patch competitor source. A build failure,
timeout, unavailable optional dependency, or missing public operation remains
in the report with its exact status.

Allowed statuses are:

- `executed-pass`: the public operation returned the required evidence;
- `executed-fail`: it ran but did not satisfy the criterion;
- `source-verified-not-executed`: the capability is present in inspected source
  but its public operation was not successfully run in this environment;
- `absent-public-surface`: no equivalent public operation was found;
- `not-comparable`: the product intentionally addresses a different operation.

Only `executed-pass` is an execution proof. Source inspection is never silently
promoted to a pass.

## Frozen fixture families

1. Python same-name methods on two classes, with annotated, constructor-bound,
   `self`, and untyped receivers.
2. Python branch and loop definitions with known may-reach and must-reach uses.
3. A five-file dependency hub plus one lexically rare isolated symbol.
4. Python, Rust, TypeScript, Go, and Java caller/callee pairs.
5. UTF-16 LSP locations containing a non-BMP character.
6. Value-bearing trace input whose value must not appear in output.
7. Cold, unchanged warm, one-file-changed, corrupt-cache, and stale-source runs.

The shared repository-map/type/flow fixture and machine-readable gold labels
live in `benchmarks/fixtures/code_intelligence_conformance/`. Additional
five-language, runtime, semantic-location, cache, and stale-source fixtures are
defined by the focused tests named in `docs/capability-coverage.json`; all
failures remain in the denominator.

## Conformance dimensions

### Structural and semantic correctness

1. Syntax-backed declarations across the five-language fixture.
2. Exact declaration and call-site byte spans.
3. Explicit refusal of ambiguous same-name calls.
4. Correct typed dispatch among same-name methods.
5. Static caller/callee traversal from an unambiguous symbol.
6. Branch/loop control-flow edges.
7. May-reach versus must-reach definition edges.
8. External definition/reference/override range intake using LSP UTF-16 rules.

### Repository understanding

9. Global dependency/call hub ranking.
10. Query-personalized rare-symbol ranking.
11. Bounded call/import/containment context expansion.
12. Explicit unresolved and omitted evidence.
13. Content-addressed parse, whole-index, and derived-analysis reuse; one-file
    changes invalidate global graph results rather than returning stale edges.

### Evidence and operational truthfulness

14. Fresh source revalidation before output.
15. Tamper-evident commitment over returned context or graph.
16. Exact evidence digest for every claimed source span.
17. Value-free runtime/coverage observation binding.
18. Bounded token estimate and visible budget omissions.
19. Parser-derived health findings carry exact source-span evidence.
20. Stale files are omitted from health reports rather than graded.
21. Import cycles and unresolved-call risk remain separately inspectable.
22. Health-policy thresholds, score formula, coverage, and report commitment
    are machine-verifiable.
23. Rename preview performs zero writes and commits exact identifier preimages.
24. Ambiguous symbols, stale source, overlapping edits, and tampered plans fail
    before mutation.
25. Apply requires the exact plan hash plus explicit incompleteness
    acknowledgement and validates staged syntax.
26. A multi-file replacement failure attempts rollback and remains visible.
27. MCP callers cannot choose or modify the operator-configured LSP executable.
28. LSP framing, timeout, output, message, relationship, environment, and
    workspace-URI bounds are exercised against an unchanged fake server.
29. External UTF-16 reference ranges pass through source verification before
    entering a committed rename plan.
30. External-process network behavior is labeled uncontrolled rather than
    silently counted as local-only execution.

## Reporting rule

Reports show each dimension and evidence pointer separately. An overall lead may
be claimed only for a named scope when one implementation has more
`executed-pass` results in that scope and no correctness-critical regression
(ambiguity, stale source, or evidence tampering). Language breadth, native LSP
execution/refactoring, and persistent global graph maintenance are reported as
separate capabilities rather than hidden in the 30-point score. Code-health
conformance covers evidence and policy honesty; it does not award a pass merely
for producing more warnings.
