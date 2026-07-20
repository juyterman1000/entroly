# Entroly evidence-first launch playbook

This file is a publication gate, not a library of prewritten promotional claims. Public communication should help a developer reproduce value on their own workload.

## Non-negotiable trust rules

- Lead with the user problem: agents repeatedly receive noisy, incomplete, or unauditable context.
- Describe Entroly as a budgeted, auditable context layer. Do not promise a universal token, cost, quality, or latency improvement.
- Bind every numeric product claim to a committed artifact, named protocol, repository revision, environment, and limitations.
- Treat token reduction, answer quality, latency, provider cache behavior, and billing as separate measurements.
- Never claim that WITNESS prevents all hallucinations. It emits an evidence-grounding signal and can be wrong.
- “No extra provider call” does not mean free: local compute and operational costs still apply.
- “Local-first” does not mean a proxied cloud-model request stays local; selected prompt content still goes to the configured provider.
- Do not imply vendor certification or competitor inferiority.
- Do not publish simulated screenshots or example counters as benchmark evidence.

The canonical evidence and limitations are in [public-evidence.md](../public-evidence.md).

## Before publishing anything

1. Run the relevant benchmark from a clean checkout.
2. Verify the result artifact and its hashes.
3. Confirm the documented command matches the released package.
4. State what was measured and what was not measured.
5. Link directly to the artifact and reproduction command.
6. Have a maintainer review the wording for overgeneralization.

If any step fails, publish the capability without a number or defer publication.

## Human, problem-first message

> Coding agents are good at generating code, but they can still receive the wrong repository evidence or lose earlier decisions between sessions. Entroly is an open-source context layer that selects evidence under an explicit token budget and produces receipts showing what was included and omitted. You can run a bounded local smoke test before connecting a model:
>
> `pip install entroly && cd /your/repo && entroly verify-claims`
>
> The smoke is not a billing or answer-quality guarantee. For reproducible measurements and limitations, see the public evidence ledger.

Use this as a factual structure, not as text to post repeatedly.

## Demonstrating value

For a public demo, record a real run and show:

- repository revision and clean/dirty state;
- the exact command and Entroly version;
- source and selected token counts;
- selected and omitted evidence;
- a context receipt and recovery boundary;
- answer-quality evaluation, if claimed, as a separate step;
- provider pricing source, if a modeled cost is shown.

Redact secrets, personal paths, repository content, and access tokens. Prefer a small public repository whose license permits demonstration.

## Claim templates

Safe:

> On `<repository>@<revision>`, query `<query-id>`, budget `<tokens>`, and Entroly `<version>`, selected input was `<selected>` tokens from `<source>` source tokens. This measures context selection only; it does not establish answer-quality or billing impact. Reproduce: `<command>`.

Unsafe:

> Entroly cuts every AI bill by X%, improves accuracy, and has zero cost.

## Community participation

Answer the question being asked before mentioning Entroly. Disclose maintainer affiliation. Link once, only when the project directly helps. Invite falsification and specific feedback instead of asking for stars.

## Release announcement checklist

- [ ] Release artifacts and package registries agree on the version.
- [ ] Installation commands pass in a clean environment.
- [ ] Public badges and marketplace links resolve to the intended destinations.
- [ ] Numeric claims pass `python scripts/verify_public_trust.py`.
- [ ] Demo assets are marked illustrative unless generated from an attached evidence artifact.
- [ ] Security and migration implications are explicit.
- [ ] Rollback and recovery steps are documented.

## Success metrics

Track outcomes that reflect durable adoption: time-to-first-success, verified installs, repeat users, documentation completion, issue response time, contributor conversion, and workload-specific benchmark reproductions. Stars are useful discovery feedback, but they are not proof of product value.
