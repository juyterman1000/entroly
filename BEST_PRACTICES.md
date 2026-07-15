# Entroly Best Practices

## Prove the local path first

Run `entroly doctor`, `entroly verify-claims`, and `entroly simulate` before
connecting a paid model. This separates installation problems from provider,
agent, and billing behavior.

## Choose one integration boundary

Use scoped attachment or MCP for subscription-based agents, the proxy when your
application controls provider API keys and base URLs, and the SDK when your code
owns context assembly. Avoid stacking wrapper, proxy, and SDK compression until
each layer has been measured independently.

## Set a task and budget explicitly

For aggressive reduction, provide the real query and an explicit budget.
Task-conditioned selection is more defensible than query-agnostic truncation.
Unknown or untrusted model limits should cause conservative fallback, not an
invented budget.

## Preserve high-value evidence

Keep errors, test failures, user requirements, changed files, security findings,
API contracts, and dependency links at high fidelity. Compress repetitive logs,
boilerplate, and already-represented structure first. Review omitted nearby
evidence on consequential tasks.

## Retain and protect receipts

Store receipts and recovery bundles only as long as needed, under the same
access controls as the source. Back up them together if exact replay matters.
Deleting the recovery store deletes Entroly's exact recovery path.

## Roll out conservatively

1. Establish an exact-passthrough baseline.
2. Run a held-out workload with fixed model settings.
3. Compare task success, token use, latency, cache behavior, and recovery.
4. Enable audit-only verification.
5. Canary a conservative budget for a small user group.
6. Expand only when regressions have a clear rollback.

## Keep provider behavior observable

Test streaming, tool calls, images, beta headers, retries, timeouts, and upstream
status codes—not only one buffered text request. Provider failover or model
routing must be explicit and receipt-recorded.

## Separate savings tiers

- **Estimated:** local tokenizer or heuristic calculation.
- **Provider observed:** usage returned by the provider for the request.
- **Realized:** reconciled production usage or invoice impact.
- **Opportunity:** forecast under a proposed policy.

Do not add these tiers together or present a forecast as money already saved.

## Treat verification as evidence, not certainty

WITNESS and related checks reduce risk on scoped workloads; they do not prove an
answer universally correct. Preserve citations and residual-risk warnings, and
require human review for high-consequence decisions.

## Benchmark the same workload

When comparing Entroly with raw context or another tool, freeze versions,
inputs, prompts, model settings, token budgets, seeds, and scoring. Publish
per-case results and disagreement cases. A lower token count is not a win if
task success, recoverability, or latency falls outside the acceptance criteria.

## Plan upgrades and removal

Pin versions in production, read [CHANGELOG.md](CHANGELOG.md), test the upgrade
against retained receipts, and keep a documented exact-passthrough path. Scoped
attachment should be revocable; wrappers and base-URL changes should have a
tested uninstall or reset procedure.

