# Entroly for OpenClaw

Entroly provides budget-aware, auditable context assembly for OpenClaw while
leaving OpenClaw's persisted transcript untouched.

Unlike uniform history summarization, Entroly scores older messages against the
current request, reserves a bounded part of the context budget for matching
evidence, and keeps evidence messages verbatim when they fit. Lower-value
history is compressed around those evidence pins. The receipt records every
score, matched query term, allocation, and transformation.

## Install

After the package is published:

```bash
pip install entroly
openclaw plugins install entroly-openclaw
openclaw plugins enable entroly
```

From an Entroly source checkout:

```bash
pip install entroly
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

## Reproduce the evidence-pinning control

```bash
python -m benchmarks.openclaw_evidence_pinning
```

The committed synthetic workload compares query-aware evidence pinning with
uniform budget compression at the same estimated token budget. See the
[result JSON](../../benchmarks/results/openclaw_evidence_pinning.json). It uses
no model calls and does not claim downstream task accuracy.

Receipts are written under `<workspace>/.entroly/receipts/openclaw/` unless
`receiptDir` is configured. They record per-message hashes and decisions,
estimated tokens, reduction, warnings, and whether context changed. The
original content remains recoverable from OpenClaw's unchanged transcript.
Entroly makes no remote calls in this path.

## Safety contract

- System and developer messages are never modified.
- Structured message content is never modified.
- Recent messages are preserved verbatim.
- Bridge errors return the exact original message list.
- Entroly does not rewrite the OpenClaw transcript or claim persistent
  compaction.
