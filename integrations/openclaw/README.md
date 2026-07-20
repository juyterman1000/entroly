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
pip install "entroly>=1.0.64"
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
pip install "entroly>=1.0.64"
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
the Entroly 1.0.64 bridge v2 protocol; doctor reports an actionable upgrade
instead of accepting an older, incompatible Python installation.

OpenClaw's resolved prompt token budget is authoritative. When an older or
degraded host cannot provide one, Entroly automatically resolves a conservative
input ceiling from its model registry. Only `verified`, operator-supplied
`user`, or explicitly `discovered` metadata is accepted; announced and unknown
records are rejected. The receipt records the model id, trust state, registry
digest, native window, output reserve, safety reserve, and source.

Context-budget discovery is local and enabled by default. It does not enable
remote provider discovery or send credentials anywhere. Local Ollama/LM Studio
and remote OpenRouter discovery remain separate operator opt-ins through
Entroly's model-registry environment controls. Disable this fallback with
`autoDiscoverContextBudget: false`.

For a custom model that neither OpenClaw nor trusted registry metadata can
resolve, configure the model in OpenClaw, provide an `ENTROLY_MODEL_REGISTRY`
override, or set an explicit last-resort fallback:

```json5
{
  plugins: {
    entries: {
      entroly: { config: { fallbackTokenBudget: 32768 } }
    }
  }
}
```

Without any trusted budget, Entroly returns the exact original context with an
actionable warning instead of risking a provider overflow. Explicit OpenClaw
budgets always win, followed by the operator's `fallbackTokenBudget`; automatic
registry discovery is used only when neither is available.

## Optional proof-guided exact recovery

Default behavior performs local context assembly only and makes no additional
model call. To let Entroly verify a draft, recover exact omitted messages, and
ask OpenClaw for a bounded revision, opt in explicitly:

```json5
{
  plugins: {
    slots: { contextEngine: "entroly" },
    entries: {
      entroly: {
        hooks: { allowConversationAccess: true },
        config: {
          proofGuidedRecovery: true,
          proofGuidedMaxRounds: 2,
          proofGuidedRecoveryTokens: 1200,
          proofGuidedMaxMessages: 3
        }
      }
    }
  }
}
```

`proofGuidedMaxRounds` includes the first response, so the default maximum of
two allows at most one additional model call. Entroly never receives provider
credentials and never calls the provider itself: it returns an idempotent
revision request to OpenClaw, which keeps its normal provider routing and
billing behavior. The retry contains exact recovered message text and SHA-256
commitments, not a generated summary.

The local bridge checks the draft with EICV and the context firewall. If a
claim is unsupported, it either identifies exact omitted evidence for the
bounded revision or supplies a safer verified output. If bridge verification
fails or model output is unsafe, the delivery hook substitutes a clear withheld
response instead of silently passing unverified text. `/entroly-context` shows
the proof status, attempt count, and local audit artifact ID.

This feature requires a current OpenClaw build that exposes `llm_output`,
`before_agent_finalize`, and `reply_payload_sending`, plus the explicit
`allowConversationAccess` grant shown above. Verify registration after restart:

```bash
openclaw plugins inspect entroly --runtime --json
openclaw plugins doctor
```

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
