# Exact-source evidence location

Entroly can locate passages that address a natural-language query while
returning only text that already exists in the supplied source. This helps when
literal search misses wording such as `cancel` versus `termination of service`,
but an agent still needs source offsets, hashes, and a bounded claim.

## CLI

The deterministic lexical ranker is available in the base installation:

```bash
entroly find terms.txt --query "what happens if I cancel?" --json
```

An existing local sentence-transformer can provide paraphrase matching:

```bash
entroly find terms.txt \
  --query "what happens if I cancel?" \
  --semantic-model /models/local-encoder \
  --threshold 0.62 \
  --calibration-id cancellation-eval-v1 \
  --json
```

Entroly refuses remote model identifiers and does not download a model. The
optional local encoder requires the `neural` extra. A threshold becomes a
calibrated operating point only when the caller supplies the identity of the
external calibration artifact used to choose it.

Each match includes exact source text, character offsets, SHA-256 hashes,
ranker identity and fingerprint, source-coverage status, and exact-recovery
metadata. The command exits `0` when evidence is selected, `1` for abstention
or no match, and `2` for invalid input or configuration.

## Python

```python
from entroly import locate_evidence, verify_evidence_location

result = locate_evidence(source_text, "workspace rate limits")
for match in result.matches:
    print(match.focus_text, match.focus_start_char, match.focus_end_char)

assert verify_evidence_location(result, current_source_text)["valid"]
```

Custom rankers return exactly one supplied passage ID, score, and supplied
sentence index per candidate. Unknown IDs, duplicates, missing judgments,
invalid scores, and invented sentence indices cause abstention. Entroly does
not silently fall back after a configured ranker fails.

## Browser and MCP

`entroly browser` accepts `--semantic-model`, `--threshold`,
`--calibration-id`, and `--max-matches`. The public MCP profile exposes
`locate_evidence(text, query, ...)`. It uses the lexical ranker unless
`ENTROLY_SEMANTIC_MODEL_PATH` names an existing local model directory, keeping
the capability provider-neutral.

## Claim boundary

A selected span proves only that the returned text and offsets came from the
supplied source and that the named ranker assigned its recorded score. It does
not prove relevance, completeness, factual correctness, independence, or task
success. Exact recovery preserves omitted text; it does not repair a bad
ranking decision.
