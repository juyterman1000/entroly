# Entroly Plugin for Google Antigravity

This plugin integrates **Entroly** with Google Antigravity (AGY, Antigravity IDE, and Antigravity 2.0).

## Capabilities

- 💰 **Knapsack Token Optimization**: Automatically extracts precisely the answer-bearing code fragments within your token budget, reducing token payload by up to 95%.
- 🔍 **Reversible Compression & Merkle Proofs**: Anything omitted is indexed locally and recoverable byte-for-byte using cryptographic SHA-256 receipts.
- 🛡️ **WITNESS Grounding Verification**: Local fact-checking to detect hallucinations and ensure code claims match indexed evidence.
- 🧠 **Cross-Agent Shared Memory**: Shared knowledge base between Claude Code, Cursor, Codex, and Antigravity.

## Structure

- `plugin.json` — Antigravity plugin manifest.
- `mcp_config.json` — MCP server definition running `python -m entroly.server`.
- `rules/entroly-context.md` — Agent rules enforcing token budgeting and receipt preservation.
- `skills/entroly-context-control/SKILL.md` — Procedural workflow instructions for Antigravity agents.

## Installation & Activation

When cloned or opened in Antigravity, the plugin is discovered from `.agents/plugins/entroly` and activated via `.agents/plugins.json`.
