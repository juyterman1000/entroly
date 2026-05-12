# BIPT — Byte-level Information Provenance Tracing

> *The first hallucination detector that asks a different question.*

## The wrong question

Every previous hallucination detector asks:

> *"Is this output **correct**?"*

That question is unanswerable in general. A function name your AI wrote could be:

- A real function in your codebase (correct)
- A real function in a *different* library you don't use (incorrect)
- A function you renamed last week (the AI doesn't know)
- A function that doesn't exist anywhere (the most common case — hallucination)

You can't tell the difference without running the code or having a perfect oracle of your codebase. Off-the-shelf hallucination detectors keep missing the mark because they're trying to solve correctness, which is hard, instead of *provenance*, which is tractable.

## The right question

BIPT asks something different:

> *"Can every byte of this output be **explained** by the input context?"*

If the LLM was given context **C** and produced output **O**, then every identifier, every API call, every constant in **O** must have a *provenance trail* back to **C**. A byte either:

1. **Comes from C** — meaningful, traceable, defensible
2. **Doesn't come from C** — invented, hallucinated, dangerous

This is a *much simpler* question. It's not asking "is this correct?" It's asking "can the AI justify this byte by pointing to a place in C?" That's a question we can answer mathematically, in linear time, with zero training data, zero external knowledge, and zero reliance on another LLM.

## The math (one paragraph)

This is **conditional Kolmogorov complexity** in disguise. From algorithmic information theory (Kolmogorov 1965; Solomonoff 1964), the conditional complexity K(O|C) measures the length of the shortest program that produces O given C as input. If K(O|C) is small, O is "explainable" by C. If K(O|C) is large, O contains information not derivable from C — i.e., invention.

True K is uncomputable (Rice's theorem). But the **Lempel-Ziv factorization** of O *against* C is computable in O(|O|) time and serves as a tight, defensible upper bound:

> LZ(O\|C) decomposes O into factors {f₁, f₂, …, fₖ}. Each factor is either:
> - a **COPY** from C (longest match in C at that position), or
> - a **NOVEL byte** (no match).

A long COPY means "the AI is quoting your code — grounded." A NOVEL byte means "the AI is inventing." Sum the novel bytes inside *identifier positions* (function names, constants, types, imports — the bytes that matter) and you get the **Identifier Provenance Deficit (IPD)** — the formal hallucination rate:

> **IPD** = Σ_{i ∈ identifiers} novel_bytes(i) / Σ_{i ∈ identifiers} len(i)
>
> IPD ∈ [0, 1] — 0 = fully grounded; 1 = fully invented.

## The algorithm

```
1. Build a Suffix Automaton (DAWG) from context C.     O(|C|) time and space.
2. For each position j in O, walk the automaton to find
   match_len[j] = length of longest C-substring matching O[j..].
                                                       O(|O|) total.
3. Parse O's AST to identify identifier byte spans
   (function names, types, constants, imports).
4. For each identifier span [start, end], compute
   novel_fraction = (end - start - max_match_within_span) / (end - start).
5. IPD = sum(novel_bytes) / sum(identifier_bytes).
6. Flag any identifier with novel_fraction > 0.5 as INVENTED.
```

That's it. Linear in |C| + |O|. ~80 lines of Python in [`entroly/verifiers/provenance_tracer.py`](entroly/verifiers/provenance_tracer.py). Adds ~5ms to a typical context-injection request.

## What you see

When BIPT runs, every model output gets a **receipt**:

```
provenance trace (IPD = 0.073)
─────────────────────────────────────────────────────────────
  identifier 'parse_request_headers' ──→ matched at C[14523..14541]   GROUNDED
  identifier 'validate_session_token' ──→ matched at C[8911..8932]    GROUNDED
  identifier 'magic_encode_v2'       ──→ NO MATCH                     INVENTED ⚠
  identifier 'decode_response'       ──→ matched at C[19302..19315]   GROUNDED

ACTION: flag 'magic_encode_v2' as hallucinated → enqueue FORGE repair
```

You see exactly which identifiers came from your codebase and which were invented. The receipt is machine-parseable, log-able, attestable, and citation-quality if you ever need to defend the AI's output in front of an auditor.

## Why nobody else has this

Every commercial hallucination detector we've found wraps the LLM as a black box: output → "fact-check" → score 0–1. None ground the check in the **specific input context**. They use external general knowledge (doesn't help for your private code), training-set similarity (doesn't help for your private code), or LLM-as-judge (which has the same hallucination problem you're trying to detect).

BIPT is grounded in the input bytes themselves. The check is:

> *did the bytes come from the bytes I gave you?*

That's an *information-theoretic invariant*. It works without knowing what's "true" in the world — only what's in your context. It cannot be gamed by a longer training run, a bigger model, or fancier prompts. The hallucination is detected by an arithmetic operation on byte sequences.

## Practical implications

| What this means | For you |
|---|---|
| You can **prove** to your team's compliance officer which functions came from your code | Required in regulated industries (finance, health, defense) |
| You can **audit** an AI session retroactively | Open the BIPT trace, see every quoted identifier and its source |
| You can **alert** on hallucinated outputs before they ship | Pre-commit hook, CI gate, IDE warning |
| You can **measure** your AI's grounding rate over time | Track IPD trends across the team's sessions |
| You can **trust** the AI more selectively | High-IPD outputs get human review; low-IPD outputs ship faster |

## Integration

BIPT runs in three places:

| Surface | How |
|---|---|
| **MCP server** | `verify_provenance` tool — agents can call it explicitly |
| **SDK** | `from entroly.verifiers import trace_provenance` — 3 lines to integrate |
| **Flow orchestrator** | Fires automatically as step 2b of the verify-before-answer flow |

### Try it from the CLI

```bash
entroly verify-provenance --context path/to/source.py --output "your AI response"
```

### Or from your code

```python
from entroly.verifiers import trace_provenance

result = trace_provenance(output_text, context_text)
print(result.ipd)                    # 0.0 = grounded, 1.0 = fully invented
print(result.invented_identifiers)   # list of bytes flagged as hallucinated
```

## What BIPT is *not*

- **Not a correctness checker.** BIPT tells you the AI is quoting your code, not that the code is correct. A perfectly grounded answer can still be wrong if your source is wrong.
- **Not a semantics checker.** BIPT works on bytes, not meaning. `parseUrl` matching against `parseURL` will register as partial-novelty.
- **Not a substitute for tests.** Your code still needs to compile and pass tests. BIPT catches the *invention* class of failure, which is *upstream* of testing.

But it is the only system that can give you a **provable receipt** for what your AI quoted from your codebase, in real time, on every request.

## Citations

- Kolmogorov, A. N. (1965). *Three approaches to the quantitative definition of information.* Problems of Information Transmission 1(1).
- Solomonoff, R. J. (1964). *A formal theory of inductive inference.* Information and Control 7.
- Lempel, A. & Ziv, J. (1976). *On the complexity of finite sequences.* IEEE Transactions on Information Theory IT-22(1).
- Blumer, A., Blumer, J., Haussler, D., Ehrenfeucht, A., Chen, M. T., & Seiferas, J. (1985). *The smallest automaton recognizing the subwords of a text.* Theoretical Computer Science 40.

---

*BIPT is layer 1 of entroly's five-layer hallucination defense (BIPT, FORGE, TRIAD, PROVE, CAVE). See [RESEARCH.md](RESEARCH.md) for the full mathematical exposition of all five layers plus the rest of entroly's algorithmic substrate (NKBE Nash-equilibrium budgeting, Causal Context Graph, Cognitive Bus, Resonance Matrix).*
