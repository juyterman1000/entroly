# BIPT — Byte-level Information Provenance Tracing

BIPT is a deterministic heuristic for a narrow, auditable question:

> Which identifier bytes in generated output have matching substrings in the supplied context?

It does **not** determine whether output is correct, whether a novel identifier exists elsewhere, or whether a response is hallucination-free.

## Why provenance is useful

Generated code can mention an identifier that was present in the supplied repository evidence, a valid new identifier, an API outside the supplied context, or an invented API. Correctness requires tests and semantic checks. BIPT contributes one signal: byte-level traceability to the exact context the model received.

Given context `C` and output `O`, BIPT builds a suffix automaton over `C`, matches output substrings, identifies language-supported identifier spans, and reports unmatched bytes. The aggregate is the Identifier Provenance Deficit:

> **IPD** = Σ novel identifier bytes / Σ identifier bytes

`IPD` is in `[0, 1]`. Lower values mean more identifier bytes matched the supplied context. This is a provenance ratio, not a calibrated probability of hallucination.

## Algorithm

```text
1. Build a suffix automaton from context C.
2. Find the longest C-substring beginning at each output position.
3. Extract identifier byte spans from supported output syntax.
4. Compute matched and unmatched bytes for each identifier.
5. Emit IPD and a per-identifier receipt.
```

The matching core is linear in the context and output lengths; end-to-end time also includes parsing and receipt construction. Benchmark latency on the actual language, file size, runtime, and hardware.

## Example receipt

```text
provenance trace (IPD = 0.073)
  parse_request_headers  -> matching source span
  validate_session_token -> matching source span
  magic_encode_v2        -> no qualifying match; review
```

“No qualifying match” means only that the configured matcher did not trace the identifier to the supplied context. A caller may warn, request more evidence, run symbol resolution, or require tests. It should not automatically label the identifier false.

## Integration

| Surface | How |
|---|---|
| MCP server | `verify_provenance` tool |
| SDK | `from entroly.verifiers import trace_provenance` |
| Verification flow | Optional signal alongside semantic and execution checks |

```bash
entroly verify-provenance --context path/to/source.py --output "your AI response"
```

```python
from entroly.verifiers import trace_provenance

result = trace_provenance(output_text, context_text)
print(result.ipd)
print(result.invented_identifiers)  # compatibility name: unmatched identifiers
```

## Limitations

- A matched identifier can still be used incorrectly.
- A novel identifier can be valid and intentional.
- Similar names can create partial matches.
- Identifier extraction is language-dependent.
- The supplied context can itself be stale, incomplete, or malicious.
- A provenance receipt is not a compliance certification or a substitute for review and tests.

## References

- Kolmogorov, A. N. (1965). *Three approaches to the quantitative definition of information.*
- Solomonoff, R. J. (1964). *A formal theory of inductive inference.*
- Lempel, A. & Ziv, J. (1976). *On the complexity of finite sequences.*
- Blumer et al. (1985). *The smallest automaton recognizing the subwords of a text.*

For benchmark scope and current public evidence, see [docs/public-evidence.md](docs/public-evidence.md).
