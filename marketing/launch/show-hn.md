# Show HN launch draft

Status: prepared, not submitted.

## Title

Show HN: Entroly – recoverable context selection and verification for AI agents

## Post

I maintain Entroly, an Apache-2.0 context-assurance layer for AI agents.

The project started from a concern with context-compression demos: a smaller
prompt is easy to show, but that number alone does not tell you whether the
answer-bearing evidence was removed, whether omitted material can be recovered,
or whether the model's answer remains supported.

Entroly therefore treats compression as one part of a broader control loop:

- select evidence under an explicit token budget;
- retain omitted originals behind content-addressed handles;
- emit a Context Receipt describing selections, omissions, risks, and recovery;
- let the model or operator retrieve exact originals when needed;
- check generated claims against the evidence supplied to the model;
- expose the contract through CLI, SDK, MCP, proxy, agent wrappers, Rust, WASM,
  npm, Docker, and Homebrew.

The local path does not need an API key:

```bash
pip install -U entroly
cd /path/to/repository
entroly verify-claims
entroly simulate
```

One disclosure, because it will be asked: if the native engine is missing,
`simulate` installs it from PyPI before measuring. Without it, selection never
reads the query, so the percentage would be decided by the token budget rather
than by anything the tool did. It is a package install — no code or prompts
leave the machine — and `ENTROLY_NO_SELF_HEAL=1` disables it, in which case the
figure is reported labelled as unearned rather than withheld.

`verify-claims` performs bounded installation, compression, receipt, recovery,
and routing checks. `simulate` estimates context reduction on the repository
without calling a model.

Repository: https://github.com/juyterman1000/entroly
Docs: https://juyterman1000.github.io/entroly/docs/index.html
Limitations: https://github.com/juyterman1000/entroly/blob/main/docs/limitations.md
Benchmarks: https://github.com/juyterman1000/entroly/blob/main/docs/BENCHMARKS.md

A few design choices that may be useful to discuss:

1. Recovery handles are content-addressed rather than query-based, so retrieval
   cannot silently substitute a newer or merely similar source.
2. Receipts separate what was selected from what was recoverable and what was
   actually sent to a provider.
3. Compression is allowed to pass through when context already fits the budget.
4. Benchmark harnesses include null/control checks because tasks that can be
   solved without context do not measure context quality.
5. Provider-bound savings are not treated as equivalent to a local tokenizer
   estimate when the provider's observed usage is available.

The project does not claim universal savings or universal quality retention.
Some tasks do not benefit from compression; some budgets are too aggressive;
and proxy mode still sends the selected prompt to the user's configured model
provider.

I would especially value criticism of the receipt and recovery contract,
benchmark design, integration ergonomics, and cases where pass-through should be
more conservative. Reproductions are most useful when they include the exact
version, repository revision, task, token budget, model/provider, commands, and
raw artifacts.

## Reply guidelines

- Answer technical objections directly; do not redirect criticism to marketing
  copy.
- Link code, tests, limitations, or raw artifacts whenever possible.
- Correct inaccurate claims publicly.
- Do not ask for votes, coordinated comments, or stars.
- Do not describe a result as independent when a maintainer configured or tuned
  the evaluation.
- Record the final discussion URL in the distribution registry.
