# Frequently Asked Questions

## What is Entroly?

Entroly is a local context-control plane for AI agents and applications. It can
select evidence under a token budget, compact selected material, keep omitted
content recoverable on supported paths, emit auditable receipts, and verify an
answer against supplied evidence.

## Is Entroly an agent or chat client?

No. Claude Code, Codex, OpenClaw, Cursor, Aider, or your application continues
to run the conversation, tools, provider authentication, and model. Entroly
operates on the context boundary you explicitly connect.

## Does Entroly send my code to its own service?

No outbound analytics uploader is enabled by default. Local selection,
receipts, and diagnostics run on the machine. A proxy or provider integration
still sends the selected prompt to the model provider configured by the user.
Remote model-registry discovery is opt-in.

## Is the Rust engine required?

No. The base Python package has a pure-Python path. Install `entroly[native]` or
`entroly[full]` for the published native extension. Use `entroly doctor` and
`entroly verify-claims` to see which capabilities are active; a successful
`import entroly_core` alone is not a sufficient capability check.

## Which install should I use?

- Start with `pip install entroly` for the CLI, SDK, and MCP control plane.
- Add `[proxy]` when your application controls provider base URLs.
- Add `[native]` for optional Rust acceleration.
- Add `[full]` for proxy, native, and receipt-proof dependencies.
- Use `npm install -g entroly` for the Node/WASM distribution.

See [QUICKSTART.md](QUICKSTART.md) and
[SUPPORTED_VERSIONS.md](SUPPORTED_VERSIONS.md).

## Does Entroly guarantee a fixed token reduction?

No. Savings depend on the workload, budget, duplication, content type, selected
integration, and whether the input is already compact. A trustworthy run may
pass through unchanged. Use `entroly simulate`, `entroly perf`, and the
versioned benchmark protocols to measure your workload.

## Does compression preserve every possible future answer?

Query-agnostic compression cannot know which fact a future question will need.
Use task-conditioned `optimize(..., query=...)` or receipt-producing selection
for high-ratio reductions, and retain recovery state when exact omitted content
may be needed later.

## What is a Context Receipt?

A Context Receipt records what was selected, what relevant material was
omitted, ranking and dependency reasons, fingerprints, warnings, token ratios,
and reproducibility metadata. A receipt explains a decision; exact recovery is
available only when the corresponding verified recovery store is retained.

## What is a Context Commit?

A Context Commit is a portable, content-addressed artifact binding the ordered
selected context, omitted evidence, recovery data, engine identity, and optional
parent lineage. Content addressing detects mutation. Signer identity requires
the optional attestation path.

## Does WITNESS eliminate hallucinations?

No. It is an evidence-based local verifier with measured performance on scoped
benchmarks. False positives and false negatives remain possible. Use audit mode
and independent validation for consequential workflows; do not treat it as a
medical, legal, financial, or safety certification.

## Does Entroly support every provider?

The core selection and SDK paths are provider-independent. Proxy and agent
integrations depend on protocol compatibility and tested adapters. The model
registry includes trust-labelled metadata, but unknown or untrusted context
limits must not authorize compression. Check the current compatibility section
in [README.md](README.md) and the integration-specific guide.

## Can I use Entroly with OpenClaw?

Yes. Install the published ClawHub plugin or use scoped attachment. OpenClaw
continues to own conversations, provider routing, authentication, and failover;
Entroly supplies context selection, receipts, recovery, and verification at the
context-engine boundary.

## How is Entroly different from Headroom, LeanCTX, or llmtrim?

All can reduce context cost. Entroly's primary boundary is auditable context
control: evidence selection before compression, explicit omitted-evidence
receipts, recoverability, Context Commits, and answer verification. Choose based
on the integration, latency, fidelity, and operational evidence that match your
workload. Run versioned tools on the same held-out tasks rather than relying on a
universal comparison claim.

## When should I skip Entroly?

Skip or pass through when the prompt is already small, every token is legally or
operationally required verbatim, added latency exceeds the value, the target
integration has no maintained adapter, or you cannot retain receipts/recovery
state safely. See [docs/limitations.md](docs/limitations.md).

## How do I get help?

Use [SUPPORT.md](SUPPORT.md). Security vulnerabilities must be reported
privately according to [SECURITY.md](SECURITY.md).

