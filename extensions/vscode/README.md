# Entroly — VS Code Extension

Select and compress repository context for AI tools with auditable receipts inside VS Code and Cursor.

---

## Key Features

1. **⚡ Status Bar Live Token Counter**: See a local token estimate for the active file or highlighted code block. Provider billing may differ.
2. **Compress a selection or file (`Ctrl+Alt+E` / `Cmd+Alt+E`)**: Runs the local `entroly compress` CLI on what the editor shows, including unsaved edits. The panel shows the token counts from that run's receipt and copies the compressed text under a one-line header naming the `entroly recover` command that restores the omitted parts.
3. **Compress receipts**: View the last run's source SHA-256, token counts, codec, recovery digest and recovery store. Nothing is shown that the CLI did not report; with no run yet, the view says so.

Requires the Python CLI on the PATH VS Code starts with (`pip install -U entroly`). Without it the compress commands report that and show no figures.

---

## Commands

- `Entroly: Compress Selection` (`Ctrl+Alt+E` / `Cmd+Alt+E`)
- `Entroly: Compress Current File for Prompt`
- `Entroly: View Last Compress Receipt`
- `Entroly: Open Interactive Web Playground`

---

## Extension Settings

- `entroly.tokenBudget`: Token budget shown beside the status bar count (default: `450` tokens).
- `entroly.showStatusBarTokens`: Toggle real-time token counter in the status bar (default: `true`).
- `entroly.useCliIfAvailable`: Run the local `entroly compress` CLI for the compress commands (default: `true`). They have no other implementation.
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
