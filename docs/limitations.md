# Entroly Limitations

Entroly is a local context control plane. It can reduce what you send, preserve
more useful context under a budget, and surface verification signals. It does
not make provider calls free, guarantee perfect answers, or override provider
terms.

## What Entroly Does Not Guarantee

- No universal savings guarantee. Savings depend on repo size, prompt shape,
  model context window, tool-output volume, and how often provider prefix caches
  hit.
- No zero quality-loss guarantee. Benchmarks measure retention on specific
  datasets and budgets; some workloads lose accuracy when the budget is too
  tight.
- No provider certification. Entroly works through documented local proxy,
  SDK, MCP, and wrapper paths, but each provider account remains subject to
  that provider's terms, data policy, model limits, and rate limits.
- No fully local cloud inference. Entroly indexes and selects locally, but when
  you use a cloud LLM API, the optimized prompt still goes to that provider.
- No automatic legal/compliance decision. `docs/provider-compliance.md` is an
  engineering checklist, not legal advice.
- No perfect hallucination detection. WITNESS/STAVE/EICV are local verifiers
  with measured false positives and false negatives. Use strict mode only when
  conservative suppression is acceptable.
- No exact provider image token guarantee for every model. Entroly estimates
  image tokens from published provider formulas where available; provider usage
  metadata or token-count APIs remain the source of truth.

## Best-Fit Workloads

- Large or medium repos where raw context does not fit.
- Chatty coding agents where repeated prefixes and old tool outputs dominate
  cost.
- Teams that want local preflight measurement before changing provider traffic.
- Workflows where evidence-grounded verification is useful even when it is not
  perfect.

## Poor-Fit Workloads

- Tiny prompts that already fit comfortably.
- One-off creative tasks where all context is intentionally unstructured.
- Tasks where any compression, selection, or image downscaling is unacceptable.
- Provider/tool configurations whose terms do not permit local proxies or
  custom endpoints.

## How To Measure Honestly

Use local commands first:

```bash
entroly simulate --budget 4096
entroly perf --budget 4096 --json
entroly verify-claims
```

These commands do not call an LLM. They estimate local context reduction and
optimizer latency. They are not a bill guarantee because they exclude output
tokens, provider cache hit rates, retries, tool-use overhead, and model-specific
pricing changes.

For live traffic, inspect response headers such as:

- `X-Entroly-Action`
- `X-Entroly-Outcome`
- `X-Entroly-Transform-Compliant`
- `X-Entroly-Tokens-Saved-Pct`
- `X-Entroly-Cache-Hit-Rate`

Provider invoices and provider usage metadata remain the billing source of
truth.

## Image Inputs

Image optimization is opt-in. The default behavior is to preserve image bytes
and only report estimates/recommendations. When enabled, Entroly gates any image
rewrite on estimated token savings and a quality floor.

References for current formulas and token-count behavior:

- OpenAI image token accounting: https://platform.openai.com/docs/guides/images-vision
- Anthropic vision image sizing and rough token estimate: https://docs.anthropic.com/en/docs/build-with-claude/vision
- Gemini multimodal token counting: https://ai.google.dev/gemini-api/docs/tokens

## Context Receipts

Context Receipts are an audit trail for local context selection. They improve
inspectability, but they do not prove that the selected context is complete or
that an answer is legally, financially, medically, or operationally correct.
Each receipt includes a deterministic `risk_summary` so these boundaries are
visible in the artifact instead of hidden in marketing copy.

- Dependency detection is heuristic. The MVP catches obvious defined terms and
  references such as `as defined in`, `subject to`, `pursuant to`, `see
  section`, exhibits, schedules, addenda, and clauses. It can miss implicit
  dependencies, unusual drafting styles, scanned/OCR errors, tables, footnotes,
  and jurisdiction-specific language.
- BM25-style retrieval is lexical. The semantic/vector scorer and reranker are
  extension points, but the local default does not claim embedding-level recall.
- Page numbers are preserved only when the input text exposes page markers. PDF
  layout reconstruction and OCR are outside this MVP path.
- Fingerprints make a receipt reproducible for the ingested text bytes. They do
  not certify that the source corpus was complete, authorized, or unchanged
  outside the files Entroly saw.
- Omitted-context warnings are conservative signals, not exhaustive proof of all
  missing evidence.
- `risk_summary.coverage_score` and `review_level` are local heuristics derived
  from chunk coverage, token coverage, unresolved dependencies, and omitted
  relevant chunks. They are triage signals, not correctness probabilities.
- Human review is required for contracts, compliance, policy, and audit use
  cases. Use receipts to inspect evidence and risk, not as a substitute for
  professional review.

## Compliance Checklist

Before production use:

1. Confirm your provider and tool terms permit the chosen proxy, wrapper, or SDK
   path.
2. Run `entroly doctor --privacy`.
3. Run provider-shape tests for the APIs you use.
4. Start in audit/observe mode, inspect headers, then enable stronger mutation
   paths deliberately.
5. Keep benchmark claims tied to committed artifacts and dated reproduction
   commands.
