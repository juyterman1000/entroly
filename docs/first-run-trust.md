# First-Run Trust Guide

Entroly is useful only if a new developer can install it, see value, and understand the safety boundary in minutes.

This guide is the first-run contract for CLI, SDK, MCP, proxy, and package-manager users.

## 1. Install

```bash
pip install -U entroly
```

Alternative surfaces:

```bash
npm install -g entroly
docker pull ghcr.io/juyterman1000/entroly:latest
```

## 2. Verify the local package

```bash
entroly verify-claims
```

This checks:

- SDK import path
- local repo indexing
- context optimization under a budget
- exact recovery for compressed fragments
- native Rust engine or pure-Python fallback mode
- no provider API key requirement for the smoke test

## 3. See savings before using a paid model

```bash
entroly simulate
```

This is a local, no-LLM estimate. It should show:

- files indexed
- tokens indexed
- selected fragments
- selected tokens
- estimated reduction against the stated baseline
- latency
- limitations of the estimate

The simulation is not a quality benchmark and does not claim provider-bill parity. It is a fast confidence check before wiring Entroly into a paid model path.

## 4. Choose the right integration

| User type | Recommended path | Why |
|---|---|---|
| Claude Code subscription user | `claude mcp add entroly -- entroly` | Keeps Claude Code as the client and adds Entroly tools without proxy billing assumptions |
| Cursor, VS Code, Windsurf, or MCP-native IDE user | `entroly init`, or register `entroly` with no arguments | Adds the installed Python MCP runtime without a Docker requirement |
| Pay-as-you-go API user | `entroly proxy` | Transparent optimization for Anthropic/OpenAI-compatible clients |
| Python SDK user | `from entroly import compress, compress_messages, optimize` | Direct library control |
| Node/npm user | `npm install -g entroly` | WASM runtime path without Python-first setup |
| CI user | `entroly batch --budget 8000 --fail-over-budget` | Enforce prompt budgets before merge |

Do not mix paths on the first run. Run `verify-claims` and `simulate`, choose the
one integration that matches your workflow, then expand later.

## 5. Safety expectations

Entroly should not pretend every workload benefits.

Expected behavior:

- Large repos, noisy logs, repeated tool outputs, and multi-turn coding sessions should show savings.
- Tiny repos or short prompts should pass through.
- Any compressed fragment should be recoverable by handle.
- Receipts should show what was selected, what was omitted, and what risk remains.
- Verification should flag unsupported claims instead of silently trusting compressed context.

## 6. What success should feel like

A first-time developer should leave the first run with three concrete signals:

| Signal | How they see it |
|---|---|
| Local value | `verify-claims` and `simulate` run without an API key and produce a JSON report |
| Integration clarity | MCP, proxy, SDK, npm, Docker, and CI paths are clearly separated |
| Trust boundary | Receipts, exact recovery, verification, and pass-through behavior are visible instead of hidden |

Power users can then explore Memory OS, session intelligence, value tracking, gateway accounting, multimodal intake, and self-improvement. Those are additive layers; the basic first-run path must stay simple.

## 7. What to send in bug reports

Please include:

```bash
entroly --version
entroly verify-claims
entroly simulate
```

Also include:

- operating system
- Python version
- install method
- integration path: CLI, MCP, proxy, SDK, npm, or Docker
- whether the Rust engine or Python fallback was active

Do not include private source code or secrets in public issues.
