# Language symbol extraction coverage

Entroly audits every language in its path-to-language map with one small,
reviewable source sample. This benchmark measures a narrow but important
contract: can the parser-backed extractor return the exact declarations in the
sample without inventing a parameter, return type, keyword, or body identifier?

## Measured result

Environment: `tree-sitter-language-pack==1.14.3`.

| Measure | Pre-change `main` (`2eeecb87`) | Current result |
|---|---:|---:|
| Mapped languages audited | 41 | 41 |
| Declaration-bearing samples | 37 | 37 |
| Exact declaration samples | 20 | **32** |
| Declaration coverage | 54.1% | **86.5%** |
| Verified non-declarative samples | 4/4 | **4/4** |
| Lost declaration coverage | — | **0** |

The 12 newly conforming language samples are Dart, Erlang, Go, Haskell,
Kotlin, Protobuf, R, Solidity, SQL, Svelte, Vue, and Zig.

- **Exact declaration samples (32):** Ada, Bash, C, C#, C++, Dart, Elixir,
  Erlang, Fish, Go, Haskell, Java, JavaScript, Julia, Kotlin, Lua, PHP,
  Protobuf, Python, R, Ruby, Rust, Scala, Solidity, SQL, Svelte, Swift, TSX,
  TypeScript, V, Vue, and Zig.
- **Measured declaration gaps (5):** C3, F#, Groovy, Nim, and OCaml.
- **Verified non-declarative samples (4):** assembly, CSS, HTML, and SCSS. These
  are audited for zero invented code declarations and are not placed in the
  37-language declaration denominator.

## Pass contract

A language sample passes only when all four conditions hold:

1. Tree-sitter reports valid syntax for the sample.
2. The bounded traversal completes.
3. The observed symbol set exactly equals the expected set; subset matching is
   not enough.
4. Every reported byte range decodes to the exact source stored on the span.

The exact-set rule caught a Dart false positive during this work: a generic
`function_body` heuristic emitted the returned identifier `x` as a declaration.
The public result was generated only after that defect was fixed and protected
by a regression test.

## Reproduce

From a full repository checkout:

```bash
python -m pip install "tree-sitter-language-pack==1.14.3"
python benchmarks/language_symbol_coverage.py \
  --baseline-ref 2eeecb8733103fe7234133f48b105f271662b219 \
  --check
```

The runner loads the baseline extractor directly from Git, so baseline and
current code use the same cases and installed grammar pack. CI runs the check
once with explicit grammar acquisition and again in air-gap mode. The checked-in
[JSON artifact](../../benchmarks/results/language_symbol_coverage.json) records
the case-manifest hash, extractor hashes, exact observed symbols, parser backend,
syntax status, traversal completeness, and byte-fidelity outcome.

## Limitations

- One representative sample per language is conformance evidence, not proof of
  complete grammar coverage.
- This benchmark measures declaration extraction, not call resolution, data
  flow, build semantics, answer quality, latency, or token savings.
- Coverage depends on the recorded optional parser-pack version. A later grammar
  release must regenerate and review the artifact rather than silently inheriting
  these numbers.
- The five gaps remain visible. They are not counted as supported and are not
  filled with regex-generated symbols.
