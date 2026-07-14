# Entroly for OpenClaw

Entroly provides budget-aware, auditable context assembly for OpenClaw while
leaving OpenClaw's persisted transcript untouched.

**Every OpenClaw provider benefits from the same Entroly context engine.**
OpenClaw continues to own provider selection, authentication, model failover,
and wire formats. Entroly operates once on OpenClaw's normalized messages
before each model run, whether the route is OpenAI, Anthropic, Gemini,
Nemotron, OpenRouter, Ollama, or a private/custom provider.

Unlike uniform history summarization, Entroly scores older messages against the
current request, reserves a bounded part of the context budget for matching
evidence, and keeps evidence messages verbatim when they fit. Lower-value
history is compressed around those evidence pins. The receipt records every
score, query-term match count, allocation, and transformation without storing
the request text or matched terms.

## Install from ClawHub

Install the Python engine, then install the plugin from OpenClaw's official
ClawHub registry:

```bash
pip install "entroly>=1.0.57"
openclaw plugins install clawhub:entroly-openclaw
openclaw plugins enable entroly
```

The npm-only fallback remains available as:

```bash
openclaw plugins install npm:entroly-openclaw
openclaw plugins enable entroly
```

From an Entroly source checkout:

```bash
pip install "entroly>=1.0.57"
openclaw plugins install ./integrations/openclaw
openclaw plugins enable entroly
```

Select the engine in `~/.openclaw/openclaw.json`:

```json5
{
  plugins: {
    slots: {
      contextEngine: "entroly"
    }
  }
}
```

Restart the Gateway and verify the loaded plugin with:

```bash
openclaw plugins inspect entroly --runtime --json
openclaw plugins doctor
```

After the first agent turn, run `/entroly-context` in any connected channel to
see the estimated before/after context size, reduction, warnings, and receipt.
Run `/entroly-context doctor` to verify the configured Python executable and
local JSONL bridge before inviting users onto the Gateway. The plugin requires
the Entroly 1.0.57 bridge v2 protocol; doctor reports an actionable upgrade
instead of accepting an older, incompatible Python installation.

OpenClaw's resolved prompt token budget is authoritative. Entroly never guesses
a context window from the provider name. For a custom model whose limit
OpenClaw cannot resolve, configure that model's context window in OpenClaw or
set an explicit operator-approved fallback:

```json5
{
  plugins: {
    entries: {
      entroly: { config: { fallbackTokenBudget: 32768 } }
    }
  }
}
```

Without either budget, Entroly returns the exact original context with an
actionable warning instead of risking a provider overflow.

## Reproduce the evidence-pinning control

```bash
python -m benchmarks.openclaw_evidence_pinning
```

The committed synthetic workload compares query-aware evidence pinning with
uniform budget compression at the same estimated token budget. See the
[result JSON](../../benchmarks/results/openclaw_evidence_pinning.json). It uses
no model calls and does not claim downstream task accuracy.

Receipts are written under `<workspace>/.entroly/receipts/openclaw/` unless
`receiptDir` is configured. They record keyed per-message digests and decisions,
estimated tokens, reduction, warnings, and whether context changed. Query text
and matched terms are not persisted; only lengths and counts remain. Each
append-only proposal has a cryptographic nonce and immutable payload digest. It
is first marked `proposed`, then atomically marked `accepted` only after the
plugin validates the returned message structure and reveals its precommitted
256-bit challenge. Entroly stores only a host-commit digest plus an HMAC
signature from a private key outside the workspace receipt; it never persists
the challenge secret. Acceptance proves plugin validation, not provider
delivery. Content digests are HMAC-keyed as well, preventing offline guesses of
low-entropy user prompts from a receipt alone.
The original content remains recoverable from OpenClaw's unchanged transcript.
Entroly makes no remote calls in this path.

The signing key is initialized atomically outside the workspace receipt store
under the user state directory (`%LOCALAPPDATA%/Entroly` on Windows or
`$XDG_STATE_HOME/entroly` on Unix). Controlled deployments may set
`ENTROLY_OPENCLAW_RECEIPT_KEY_FILE` to an absolute protected path. Entroly
quarantines an interrupted empty initialization artifact; a non-empty corrupt
key fails open with its exact recovery path instead of silently rotating and
invalidating earlier receipt signatures. `/entroly-context doctor` checks this
key lifecycle as part of bridge readiness.

The receipt directory defaults to private `0700` permissions and receipt files
to `0600` on Unix; Windows uses the workspace ACL. Entroly never auto-deletes
audit history. The default append-only quota is 512 files or 64 MiB, whichever
comes first. At the limit, assembly fails open with an actionable message;
archive receipts or raise `receiptMaxFiles` / `receiptMaxBytes` explicitly.

## Safety contract

- System and developer messages are never modified.
- Signed text, thinking/reasoning, images, tool calls, tool-result metadata,
  opaque provider signatures, and unknown content blocks are never modified.
- Only unsigned text fields in older normalized messages are compressible;
  provider/model/usage metadata cannot change selection or allocation.
- Recent messages are preserved verbatim.
- Bridge errors return the exact original message list.
- A malformed host message envelope is surfaced as an error; it is never
  coerced into an empty prompt.
- If the minimum safe normalized context cannot fit the host budget, Entroly
  returns the exact original context and requests OpenClaw recovery.
- Entroly does not rewrite the OpenClaw transcript or claim persistent
  compaction.
- `/compact` and provider-overflow recovery delegate to OpenClaw's native
  compaction runtime, preserving its provider-aware retry behavior.
- Active OpenClaw memory guidance remains present when Entroly is selected as
  the context engine.
