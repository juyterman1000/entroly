# Entroly Context Engineering Positioning

Use this as neutral public positioning for posts, discussions, and registry
submissions. Keep the message Entroly-first: local proof, recoverable context,
verification, and developer trust.

## 1. Core philosophy

Entroly is a local context OS for AI coding agents. It does not only shrink
text. It selects evidence, compresses supporting material, preserves exact
recovery handles, emits receipts, verifies answers, and tracks value locally.

## 2. Architectural anchors

### A. Cache alignment

Standard context pruning can change the byte prefix of a prompt on every loop.
That can reduce provider cache effectiveness. Entroly's cache alignment
surfaces keep stable context prefixes where provider terms and request shapes
allow it.

### B. Recoverable compression

When context is compressed, Entroly keeps recovery handles and receipts so
omitted material is not silently lost. This matters for code, stack traces,
contracts, policies, and logs where one omitted line can change the answer.

### C. Local verification

Entroly includes WITNESS, EICV, STAVE, provenance, and receipt-proof surfaces
so developers can check whether an answer is grounded in retained evidence
instead of trusting compressed context blindly.

### D. First-run proof

Developers should be able to run:

```bash
entroly verify-claims
entroly simulate
```

before connecting a paid model key. Those commands are bounded local smoke
tests, not finance-grade ROI guarantees.

## 3. Public message

Use this short form:

```text
Entroly is a local context OS for AI coding agents. It helps teams spend fewer
input tokens by selecting the right evidence first, compressing safely,
preserving exact recovery, and verifying answers against local context.
```

## 4. Claims discipline

- Tie savings claims to committed benchmark JSON or provider-backed reports.
- Say "no outbound analytics by default," not "zero telemetry."
- Say "works with" integrations unless certification exists.
- Do not imply every small repo benefits.
- Do not claim quality retention without pointing to the benchmark or report.
