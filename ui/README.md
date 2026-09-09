# Entroly Desktop & Web UI Application

The `ui/` directory contains the standalone, desktop-ready control plane interface for Entroly.

## Design Philosophy

- **Zero Clutter**: Pure HTML5, Vanilla CSS, and modular ES6 JavaScript. No heavy Node runtime, webpack, or `node_modules` required.
- **Desktop First**: Features an ambient desktop topbar HUD, keyboard command palette (`Cmd+K` / `Ctrl+K`), installable PWA manifest, and offline service worker caching.
- **Dual Connection**: Connects in real-time to the Entroly daemon (`localhost:9377` / `:9378`), falling back gracefully to an interactive high-fidelity simulation engine when offline.

## Surfaces & Capabilities

1. **Context Compression & Diff Workbench**:
   - Live side-by-side context optimization viewer (pruned boilerplate highlighted vs preserved semantic interfaces).
   - Real-time token budget slider with instant cost savings calculator across frontier models (Claude 3.7 Sonnet, GPT-4o, DeepSeek R1).
2. **Cryptographic WITNESS Receipts Chain**:
   - Auditable record of every context-selection decision with verifiable SHA-256 signatures.
3. **PRISM Reinforcement Learning Radar**:
   - Pure HTML5 Canvas radar chart visualizing real-time weights across Recency, Frequency, Semantic, Entropy, and Centrality.
4. **Security & Health Matrix**:
   - Tripwires for prompt injection attempts, context poisoning, and god-file bloat.

## Running the UI

### Method 1: Via the Entroly CLI
```bash
entroly dashboard
# Or launch the full supervisor:
entroly go
```

### Method 2: Direct Desktop / Browser Launch
Simply open `ui/index.html` in any modern web browser or install it as a native standalone window via Chrome/Edge ("Install Entroly Control Plane").
