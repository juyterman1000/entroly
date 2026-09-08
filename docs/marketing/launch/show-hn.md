# Show HN launch draft

Status: prepared, not submitted.

Do not add a savings percentage to this draft. The figure is decided by the configured token budget
before selection runs, so it measures the budget rather than the tool.

## Title

Show HN: Entroly – context compression that hands back a receipt you can verify

## Post

Every context-compression tool shows you a smaller prompt. None of them tell you
what they threw away.

Entroly emits a receipt for each selection: what was kept, what was omitted, and
a content-addressed handle that recovers the exact original bytes of anything it
dropped. You can check it on your own repository without an API key:

```bash
pip install -U entroly
cd /path/to/your/repo
entroly simulate
```

That prints the fragments it selected, the fragments it omitted, and the handles
that recover them. Pick any omitted fragment and recover it — the bytes come
back identical, or the receipt is wrong and you have caught me.

I maintain the project. It is Apache-2.0, local-first, and the selection path
makes no outbound calls.

**Why receipts rather than a compression ratio.** A ratio tells you the prompt
got smaller. It cannot tell you whether the answer-bearing evidence survived,
and a tool that drops the one function you needed scores well on it. So the
contract Entroly tries to hold is narrower and checkable: nothing is discarded
irrecoverably, and the receipt says what happened. Current measurements:
5,117/5,117 native source fragments verified byte-exact, 13/13 public SDK
recovery probes matching their source spans. Both link to raw artifacts in the
repo.

**One disclosure, because it will be asked.** If the native engine is missing,
`simulate` installs `entroly-core` from PyPI before measuring. Without it,
selection never reads the query at all, and any figure reported would be
arithmetic on the token budget rather than a result. It is a package install —
no code, prompts, or telemetry leave the machine — and `ENTROLY_NO_SELF_HEAL=1`
disables it, in which case the figure is reported labelled unearned rather than
quietly presented as earned.

**What it does not claim.** No universal savings number, and no universal
quality retention. Some tasks do not benefit from compression at all; some
budgets are too aggressive; compression passes through untouched when context
already fits. Proxy mode still sends the selected prompt to whichever provider
you configured — it is not a privacy layer.

Repository: https://github.com/juyterman1000/entroly
Docs: https://juyterman1000.github.io/entroly/docs/index.html
Limitations: https://github.com/juyterman1000/entroly/blob/main/docs/limitations.md
Benchmarks: https://github.com/juyterman1000/entroly/blob/main/docs/BENCHMARKS.md

Design choices I would most like argued with:

1. Recovery handles are content-addressed rather than query-based, so retrieval
   cannot silently substitute a newer or merely similar source.
2. Receipts separate what was selected from what was recoverable and what was
   actually sent to a provider — three different things that usually get
   reported as one.
3. Benchmark harnesses include null controls, because a task solvable with no
   context at all measures nothing about context quality.
4. Provider-bound savings are never treated as equivalent to a local tokenizer
   estimate when the provider's observed usage is available.

Criticism of the receipt and recovery contract is the most useful thing I could
get from this thread. Reproductions help most when they include the exact
version, repository revision, task, token budget, model, commands, and raw
artifacts.

## Reply guidelines

- Answer technical objections directly; do not redirect criticism to marketing
  copy.
- Link code, tests, limitations, or raw artifacts whenever possible.
- Correct inaccurate claims publicly, including inaccurate claims in this post.
- Do not ask for votes, coordinated comments, or stars.
- Do not describe a result as independent when a maintainer configured or tuned
  the evaluation.
- Record the final discussion URL in the distribution registry only after
  posting.
