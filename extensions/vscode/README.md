# Entroly — VS Code Extension

Cut AI context token costs by 70%+ with verifiable Merkle receipts right inside VS Code and Cursor.

---

## Key Features

1. **⚡ Status Bar Live Token Counter**: Always know the exact token cost of the active file or highlighted code block in real time.
2. **One-Click Knapsack Compression (`Ctrl+Alt+E` / `Cmd+Alt+E`)**: Select any code or file and run Entroly's 0-1 Knapsack selection engine to produce budget-optimal context for Claude Code, Cursor, or ChatGPT.
3. **Auditable Merkle Receipts**: Inspect cryptographic proofs (SHA-256 fragment hashes and exact byte offsets) verifying that context compression is 100% bit-exact and reversible.
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
