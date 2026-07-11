# Entroly for OpenClaw

Entroly provides budget-aware, auditable context assembly for OpenClaw while
leaving OpenClaw's persisted transcript untouched.

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
