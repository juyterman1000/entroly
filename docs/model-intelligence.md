# Model Intelligence Registry

Entroly's model intelligence registry is the single budgeting truth for model
context limits and capability metadata. Provider transport policy remains in
`proxy_config.py`; model facts live in `entroly/models/registry.json`.

## Why this exists

A silent generic context-window fallback can produce incorrect budgets. The
registry therefore distinguishes:

- exact, alias, and longest-prefix matches;
- verified, inferred, discovered, user-supplied, and unknown metadata;
- known model identifiers whose context limit is still unverified;
- bundled offline facts and explicit local overrides.

Unknown models are **fail-visible**. The default policy logs a warning and uses
a conservative 32,768-token window. Strict deployments can fail closed:

```bash
export ENTROLY_UNKNOWN_MODEL_POLICY=error
```

Compatibility mode restores the historical 128,000-token fallback:

```bash
export ENTROLY_UNKNOWN_MODEL_POLICY=legacy
```

## Inspecting the active registry

```bash
entroly-models resolve gpt-4o-2024-08-06
entroly-models resolve gpt-5.6-sol --json
entroly-models list --provider anthropic
entroly-models fingerprint
```

The fingerprint is deterministic and changes whenever effective model facts
change, making model-budget decisions reproducible in receipts and CI.

## Local overrides

Set `ENTROLY_MODEL_REGISTRY` to one or more JSON files separated by the
platform path separator. Later overlays win by trust and confidence.

```bash
export ENTROLY_MODEL_REGISTRY="$HOME/.config/entroly/models.json"
```

Override files use `entroly/models/schema.json`. They may contain a SHA-256
payload digest and an optional Ed25519 signature. Enforce signed overlays with:

```bash
export ENTROLY_REQUIRE_SIGNED_MODEL_REGISTRY=1
```

User overrides have higher precedence than bundled facts. They never modify the
installed package.

## Ollama discovery

Discovery is opt-in and loopback-only by default:

```bash
export ENTROLY_OLLAMA_DISCOVERY=1
entroly-models discover-ollama
```

Entroly reads `/api/tags`, probes `/api/show`, extracts architecture-specific
`context_length` metadata, and records local capabilities. Responses and model
counts are bounded. A non-loopback endpoint is rejected unless the operator
explicitly sets `ENTROLY_OLLAMA_ALLOW_REMOTE=1`.

## Trust model

| Trust | Meaning |
|---|---|
| bundled | Shipped with the Entroly package |
| signed | External registry with a verified Ed25519 signature |
| discovered | Runtime metadata from a local model server |
| user | Explicit operator override |
| inferred | Compatibility profile rather than provider-verified fact |
| unknown | No matching registry fact |

A record can exist while its context window remains unknown. Entroly will not
turn partial knowledge into a fabricated number; it applies the configured
unknown-model policy and exposes the warning through the resolution result.

## Python API

```python
from entroly.model_registry import resolve_model

resolution = resolve_model("gemini-2.5-pro-preview")
print(resolution.effective_context_window)
print(resolution.match_kind)
print(resolution.provenance)
```

This design lets future provider feeds, OpenRouter catalogs, signed community
updates, and local runtimes merge into one deterministic model graph without
hard-coding transport behavior into budgeting logic.
