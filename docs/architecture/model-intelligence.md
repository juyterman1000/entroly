# Model Intelligence Control Plane

Entroly resolves model capabilities through one evidence-aware contract rather
than treating every model name as a hard-coded context window.

## Trust states

- `verified`: capability metadata checked against an authoritative source.
- `announced`: the model or capability has been announced, but some limits or
  prices remain incomplete.
- `discovered`: a local runtime reported the model during bounded loopback
  discovery.
- `user`: an explicit local override supplied by the operator.
- `fallback`: no model record could be resolved safely.

`unknown` is represented as `null`, never as a fabricated `false`, zero price,
or invented context limit.

## Deterministic precedence

The effective registry applies layers in this order:

1. bundled offline snapshot
2. discovered local models
3. user overrides

Later layers replace earlier records and remove stale aliases. Exact aliases are
matched first. Prefix behavior is opt-in through aliases ending in a delimiter,
so names such as `gpt-4xyz` cannot inherit the `gpt-4` budget accidentally.

## Reproducible identity

Two SHA-256 fingerprints are exposed:

- the immutable bundled snapshot digest
- the effective digest after discovery, overrides and fallback policy

A context receipt can therefore prove the exact model-knowledge state used to
compute a budget.

## Budget invariant

The raw context window is not a safe input budget because input and output share
the same capacity. Entroly reserves the requested or model output allowance and
a configurable safety margin before selecting context.

## Cross-language parity

Python, Rust and WASM implementations must consume the same JSON snapshot and
produce identical canonical capability and registry digests. A parity fixture
must fail CI whenever resolution, trust, context limits or digest serialization
diverge across implementations.
