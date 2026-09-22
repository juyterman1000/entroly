# Frontier primitives: implementation and evidence boundaries

These mechanisms are experimental Entroly integrations. Their value comes from
the way Entroly joins budget evidence, reversible context decisions, memory
source policy, benchmark binding, and execution-time revalidation. The current
evidence supports those implementation properties; it does not establish a
novelty claim or measured downstream product advantage.

## What the implementation establishes

| Mechanism | Concrete behavior | Boundary |
| --- | --- | --- |
| Budget feasibility | Weighted set-cover dynamic programming returns an exact optimum and witness when search completes. Bounded search reports a valid lower bound, a feasible upper bound, and uncertainty. | Established optimization technique. Correct only for the supplied coverage graph. Lexical ID overlap is not semantic evidence. |
| Temporal compression diagnostic | Records codec measurements and input hashes before and after an append. Nontrivial changes require re-verification. | Codec similarity cannot certify semantic stability, future recoverability, or causal information loss. |
| Reference coverage | Missing references stay missing, including references absent from both selected and omitted inventories. Matching uses identifier boundaries. | Caller-supplied inventories and regex recognition; mentions do not verify surrounding assertions or establish what a model saw. The legacy `honest` field refers only to reference coverage. |
| Vault provenance policy | An explicit `VaultConfig.trusted_source_root` checks existing files and resolved containment on writes and reads. Unknown, mixed, and external provenance cannot enter the coupling, flow, or task-dream selection paths under this policy. | Root membership is not content safety. Dependency provenance that cannot be established remains unknown. Legacy unlabelled records retain historical behavior when no root policy is configured. Direct vault queries remain inspectable. |
| Skill benchmark binding | Code, test-case, and evaluator hashes bind local results to a candidate; stale results block promotion and the MCP execution path. | Local consistency evidence, not signatures, independent test design, or protection against an attacker rewriting both code and metrics. Generated tests can still be inadequate. |

## Entroly Assurance Envelope

`build_assurance_envelope` composes the five surfaces into one operation-bound
decision. A concrete evidence contradiction blocks execution. Missing or stale
evidence holds execution for re-verification. Autonomous execution is allowed
only when every applicable surface passes. The envelope hashes its canonical
inputs and does not contain provider or model names, so the same contract can
sit above hosted APIs, local models, coding agents, and future adapters.

The numeric dashboard score cannot override the categorical decision. An
envelope authorizes only the exact task and evidence hash it records; it never
creates global trust in an agent, model, skill, or repository.

## Reproduced defects in the supplied implementation

The supplied 25-test suite passed before these repairs. Additional examples showed:

- Costs 6 for A, 6 for B, and 10 for a shared AB candidate were incorrectly
  reported as a minimum of 12. Per-obligation cheapest choices form a feasible
  upper bound, not a lower bound. The corrected solver is checked against
  exhaustive subset enumeration on small graphs.
- An invented entity absent from both context inventories received 100% coverage.
- A folder-name check classified `entroly/../../outside.py` as trusted.
- A nonempty omission with no retained evidence was marked temporally stable.
- Promotion provenance was a hostname plus a mutable trust label; it did not
  bind the benchmark to code or tests.

For bounded set cover, `max_i min_cost(i)` is a valid lower bound: every full
cover must pay at least the cost of its cheapest candidate for each obligation.
The sum of independently cheapest choices can overcount a shared candidate's
alternative and is only an upper bound. Search exhaustion therefore returns
`unknown` when the budget lies between the two bounds.

## Validation and rollback

Run `tests/test_frontier_evidence_regressions.py` for counterexamples, exhaustive
oracles, filesystem escapes, actual vault projection, and disk-backed skill
promotion followed by code/test/metrics mutations. Neighboring sufficiency,
vault, coupling, task-dream, security, and skill tests cover compatibility.

The first three mechanisms remain callable primitives rather than automatically
enforced SDK policies. There is no measured downstream answer-quality, token
savings, novelty, or public production-readiness claim. Root policy is opt-in;
automatic promoted-skill execution retains its separate existing opt-in. Revert
the scoped patch to roll back behavior; no stored data migration is required.
