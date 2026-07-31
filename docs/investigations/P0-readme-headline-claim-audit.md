# P0 — README headline cites a benchmark that declares itself not headline-eligible

**Status:** open — audit only, no README or product code changed by this commit
**Class:** claim-integrity defect
**Artifact audited:** `benchmarks/results/neural_evidence_frontier.json`
**Claim audited:** "Entroly selected **1.02 of 16 passages** on average while keeping the answer-bearing passage in **298 of 300** held-out questions" (`README.md:96`, `:104-105`, `:656`)

## The finding in one line

The benchmark behind the README's first visual sets `headline_eligible: false`
in its own artifact, because its own statistical gate failed — and the README
uses it as the headline anyway.

## Required determinations

The directive asked six specific questions. Answers, each independently verified:

| Question | Answer |
|---|---|
| Does the benchmark exercise the installed product? | **No.** It imports `entroly.neural_evidence_selector` directly. |
| Which entry point invokes the selector? | **None.** |
| Can normal users reach it? | **No.** Unreachable from every shipped entry point. |
| Is its configuration enabled by default? | **No.** No flag, env var or config enables it. |
| Does the test use production wiring? | **No.** It calls the module's internals directly. |
| Is its result reproducible? | **Partly.** It has a `verify` mode, but the artifact's implementation commit is not in `main`'s history. |

### Reachability

`scripts/codebase_graph.py` computes reachability from the entry points declared
in `[project.scripts]` plus `python -m entroly` and `import entroly`.
`entroly.neural_evidence_selector` is in the unreachable set — imported only by
`benchmarks/` and `tests/`.

The single mention inside product code is not an import:

```python
# entroly/compression_retrieval_store_secure.py:247-259
for _module_name in (..., "entroly.neural_evidence_selector", ...):
    _module = sys.modules.get(_module_name)      # only patches if ALREADY loaded
    if _module is not None and hasattr(_module, "CompressionRetrievalStore"):
```

`sys.modules.get()` returns `None` unless something else already imported the
module, so this never causes a load. Confirmed at runtime: after `import
entroly` and after exercising the public SDK receipt path, the module is absent
from `sys.modules`.

Every other occurrence of "neural" in product code is a *negation* — `sdk.py`,
`server.py`, `proxy.py` and `universal_compress.py` all advertise "no neural
model, no LLM calls". The product's own documentation of itself contradicts the
headline.

### The statistical gate the artifact failed

`headline_eligible` is computed at `benchmarks/neural_evidence_frontier.py:405`:

```python
headline_eligible = (
    neural_metrics["top1_recall"] > lexical_metrics["top1_recall"]
    and p_value < 0.05
)
```

Neither condition holds:

| Method | top-1 correct | top-1 recall |
|---|---:|---:|
| Lexical BM25 | 297 / 300 | **99.0%** |
| Local transformer | 293 / 300 | **97.7%** |
| Dual-channel guard (headline) | 298 / 300 | 99.3% |

The transformer is **worse than BM25**, and McNemar exact p = **0.21875**
(`test_metrics.paired.mcnemar_exact_p`), far above 0.05. The artifact records
`calibration.reason: "no threshold passed the finite-sample risk and
non-inferiority gates"`.

The headline 298/300 comes from running BM25 *and* the transformer and keeping
both candidates when they disagree. Against BM25 alone at 297/300, the entire
neural apparatus buys **one additional trial out of 300** — the difference the
p-value says is indistinguishable from noise.

### Scope mismatch

The artifact states its own scope: `"Answer-bearing paragraph retrieval under
fixed one-of-N selection on a frozen SQuAD v2 validation subset."` The module
docstring adds: *"It does not call an LLM and therefore does not measure
downstream answer quality."*

This is Wikipedia paragraph retrieval, not a coding-agent task, and not token
savings or answer quality — the things the README's first screen is selling.

### Reproducibility gap

`implementation.git_commit` is `0dc83f1f7759d7ace58cfc2d7ae19380473452f1`, which
is **not an ancestor of `origin/main`**. The artifact was produced by code that
is no longer in the mainline history, so re-running `verify` at `main` does not
re-execute the implementation that generated it.

## What the README does and does not do wrong

To be fair to the current text: the README **does** disclose the weakness. Lines
110-113 state the p-value, say the difference "was not statistically
conclusive", and note the experiment "measures retrieval, not generated answers,
token savings, or production cost". That disclosure is honest and unusually
candid.

The defect is *placement and prominence*, not concealment. A result its own
artifact marks `headline_eligible: false` is the first visual on the page, above
everything the product actually ships. A reader who looks at the banner and
stops — which is what a first screen is for — takes away a validated headline
number that the benchmark itself refuses to certify.

## Why no gate caught this

The repository already has this exact discipline elsewhere. Two other pilots are
pinned to stay non-headline, and CI enforces it:

- `benchmarks/neural_query_shift.py:445` — `raise ValueError("query-shift pilot must remain headline_eligible=false")`
- `scripts/verify_context_assurance_public.py:206` — `"PRISM-R pilot must remain headline_eligible=false"`

Both assert the *artifact's* flag stays false. Neither checks the inverse and
more important property: **that a `headline_eligible: false` artifact is not
being used as a README headline.** The flag is computed, stored, and enforced
for internal consistency, but nothing connects it to the marketing surface.

That is the missing Phase 10 claim gate.

## Recommended action

Per the directive, and not to be actioned until independently reviewed:

1. **Do not** wire `neural_evidence_selector` into production to legitimize the
   claim. Product wiring must be justified by user value, not by preserving
   marketing copy.
2. **Do not** strengthen the claim.
3. Remove the result from the first screen, or relabel it explicitly
   experimental where it stands.
4. Add a claim gate that fails CI when a README first-screen claim cites an
   artifact with `headline_eligible: false`, or cites a benchmark whose measured
   module is unreachable from a shipped entry point.
5. Re-run the benchmark from a commit that exists in `main` so the artifact is
   reproducible from mainline history.

The honest replacement headline is not this benchmark. It is whatever survives
the fidelity repair in
[`P0-receipt-chunk-fidelity.md`](P0-receipt-chunk-fidelity.md) — a property that
is unique to Entroly, reaches users through shipped entry points, and can be
verified byte-for-byte by a third party.

## Reproduction

```bash
# the artifact declares itself ineligible
python -c "import json;d=json.load(open('benchmarks/results/neural_evidence_frontier.json'));print('headline_eligible:',d['headline_eligible']);print('mcnemar_p:',d['test_metrics']['paired']['mcnemar_exact_p']);print('bm25:',d['test_metrics']['lexical_bm25']['top1_correct'],'transformer:',d['test_metrics']['local_transformer']['top1_correct'])"

# the measured module is unreachable from every shipped entry point
python scripts/codebase_graph.py --json /tmp/g.json >/dev/null
python -c "import json;print('unreachable:', 'entroly.neural_evidence_selector' in json.load(open('/tmp/g.json'))['unreachable'])"

# the implementation commit is not in mainline history
git merge-base --is-ancestor 0dc83f1f7759d7ace58cfc2d7ae19380473452f1 origin/main \
  && echo ancestor || echo "NOT an ancestor of origin/main"
```
