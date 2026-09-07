# Correction: Awesome-LLMOps entry overclaims

Status: prepared, not submitted.
Target: https://github.com/tensorchord/Awesome-LLMOps
File: `README.md`
Priority: **highest** — this is a live false claim, which is worse than an
absent listing. It is the first thing a skeptical reader checks.

## The line as currently published

```markdown
| [Entroly](https://github.com/juyterman1000/entroly)                               | Information-theoretic context optimization proxy. Cuts LLM token costs by 70–95% with zero accuracy loss using greedy submodular knapsack maximization. | ![GitHub Badge](https://img.shields.io/github/stars/juyterman1000/entroly.svg?style=flat-square) |
```

## The replacement

```markdown
| [Entroly](https://github.com/juyterman1000/entroly)                               | Local-first context selection for AI agents. Emits a receipt naming every omitted fragment and a content-addressed handle that recovers its exact original bytes. | ![GitHub Badge](https://img.shields.io/github/stars/juyterman1000/entroly.svg?style=flat-square) |
```

Column count, link, and badge are unchanged. Only the description changes.

## Why this must be corrected

**"Cuts LLM token costs by 70–95%"** is not a measurement. In
`entroly/cli.py`, `saved = max(0, baseline - selected_tokens)` with
`baseline = min(total_tokens, 32_000)`. At a token budget of 8,000 the figure
is pinned at or above 75% *before selection has run*. It describes the
configured budget, not anything the tool achieved. A reader who reproduces it
and then reads that arithmetic has every reason to distrust the rest of the
project.

**"with zero accuracy loss"** is unqualified and contradicts the architecture.
Verification in Entroly fails closed precisely because retention is not
guaranteed; the project's own limitations document states that some tasks do
not benefit from compression and some budgets are too aggressive. An entry
promising zero loss claims something the codebase is built to deny.

**"greedy submodular knapsack maximization"** is also imprecise. The objective
in `knapsack.rs` is modular, so density-greedy gives a Dantzig-style ½ bound,
not the (1−1/e) that "submodular" implies to a reader who knows the term. The
replacement drops the mechanism claim rather than restating it wrongly.

The replacement text makes only claims that a stranger can check: run
`entroly simulate`, take any omitted fragment, recover it, compare bytes.

## Submission

Fork, single-line edit, PR titled:

> Fix Entroly entry: replace unverifiable savings claim with a checkable one

PR body:

> I maintain Entroly and I am correcting my own entry.
>
> The current description claims "Cuts LLM token costs by 70–95% with zero
> accuracy loss." That percentage is arithmetic on the configured token budget
> rather than a measured result — at a budget of 8,000 it is pinned above 75%
> before selection runs — and "zero accuracy loss" is unqualified in a way the
> project's own limitations document contradicts.
>
> The replacement describes only what a reader can verify themselves. Same
> column count, same link and badge; description only.
>
> Sorry for the noise, and thanks for maintaining the list.

## Related

`marketing/POSITIONING.md` records the frozen positioning and the rule that
produced this correction. Any future listing copy is subject to the same rule:
no savings percentage, anywhere.
