# Entroly — VS Code Extension

Select and compress repository context for AI tools with auditable receipts inside VS Code and Cursor.

---

## Key Features

1. **⚡ Status Bar Live Token Counter**: See a local token estimate for the active file or highlighted code block. Provider billing may differ.
2. **One-Click Context Selection (`Ctrl+Alt+E` / `Cmd+Alt+E`)**: Select code or a file and ask Entroly to assemble context under a chosen budget for Claude Code, Cursor, or ChatGPT.
3. **Auditable Merkle Receipts**: Inspect SHA-256 fragment hashes and source offsets for supported recovery paths.
4. **Inspectable Omissions**: See exactly why low-utility spans were pruned to protect your token budget.

---

## Commands

- `Entroly: Compress Selection into Optimal Context` (`Ctrl+Alt+E` / `Cmd+Alt+E`)
- `Entroly: Compress Current File for Prompt`
- `Entroly: Inspect Cryptographic Merkle Receipt`
- `Entroly: Open Interactive Web Playground`

---

## Extension Settings

- `entroly.tokenBudget`: Default target token budget (default: `450` tokens).
- `entroly.showStatusBarTokens`: Toggle real-time token counter in the status bar (default: `true`).
- `entroly.useCliIfAvailable`: Use local Rust/Python `entroly` CLI if present on path.
- `entroly.syncUnsavedBuffers`: Opt in to sending dirty file buffers to the configured proxy so selection can see edits before save (default: `false`). Save, close, or undo to the saved state retracts an installed overlay. Avoid pointing `entroly.proxyUrl` at an untrusted service when this is enabled.

---

## Building Locally

```bash
cd extensions/vscode
npm install
npm run compile
```

To package as `.vsix`:
```bash
npx @vscode/vsce package
```
