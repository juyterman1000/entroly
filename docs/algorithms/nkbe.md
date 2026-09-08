# NKBE — Nash-KKT Budgetary Equilibrium

**Implementation:** [`entroly-core/src/nkbe.rs`](../entroly-core/src/nkbe.rs)
**Math reference:** [`../RESEARCH.md`](../RESEARCH.md)

## One-paragraph spec

Given N AI agents sharing a global context budget B, NKBE finds the
optimal split. It maximizes weighted utility `Σᵢ wᵢ · Uᵢ(Bᵢ)` subject to
`Σᵢ Bᵢ ≤ B` and `Bᵢ ≥ Bᵢ_min` for each agent. The optimum is found via
**two-phase KKT bisection**, then refined by **Nash bargaining** for
Pareto-optimal fairness. **REINFORCE** updates the agent priorities
online as outcomes come in.

## When you need it

Multi-agent coding workflows: Claude Code with subagents, Aider with
multiple panes, a CI agent + a review agent, etc. Every other context
engine flat-splits the budget. NKBE makes sure heavy lifters get budget
proportional to their utility, not their count.

## Algorithm sketch

1. **Global:** bisect Lagrange multiplier `λ*` until `Σᵢ Bᵢ(λ*) = B`.
2. **Per-agent:** each agent runs knapsack with its share `Bᵢ(λ*)`.
3. **Nash refinement:** gradient ascent on the log Nash product ensures
   Pareto optimality — no agent can do strictly better without another
   doing strictly worse.
4. **Online learning:** REINFORCE updates `wᵢ` from outcome rewards.

## Why it's novel

The application of **Arrow-Debreu Walrasian economics** + **Nash
bargaining** to AI context budget allocation is, to our knowledge,
unique. Standard RAG systems don't formalize the multi-agent case at
all; they treat each agent as an independent caller. NKBE makes the
shared budget a first-class allocation problem and solves it
optimally.

## References

Arrow & Debreu 1954; Nash 1950; Karush-Kuhn-Tucker conditions.
