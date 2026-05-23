#!/usr/bin/env node
// Entroly CLI — JS port of cli.py
// Zero-friction onboarding for AI coding agents.
//
// Commands:
//   entroly optimize   Generate optimized context snapshot
//   entroly serve      Start MCP server with auto-indexing
//   entroly health     Analyze codebase health (grade A-F)
//   entroly status     Check if server is running
//   entroly stats      Show session statistics
//   entroly init       Auto-detect project + AI tool, generate MCP config
//   entroly demo       Before/after demo showing token savings
//   entroly clean      Clear cached state

const { WasmEntrolyEngine } = require('../pkg/entroly_wasm');
const { EntrolyConfig } = require('./config');
const { autoIndex } = require('./auto_index');
const { persistIndex, loadIndex } = require('./checkpoint');
const { EntrolyMCPServer } = require('./server');
const { runAutotune } = require('./autotune');
const path = require('path');
const fs = require('fs');
const os = require('os');

const VERSION = require('../package.json').version;

// ANSI colors
const C = {
  BOLD:   '\x1b[1m', GREEN:  '\x1b[38;5;82m', CYAN:   '\x1b[38;5;45m',
  YELLOW: '\x1b[38;5;220m', RED: '\x1b[38;5;196m', GRAY: '\x1b[38;5;240m',
  RESET:  '\x1b[0m',
};

function banner() {
  return `${C.CYAN}${C.BOLD}  Entroly${C.RESET} v${VERSION} — information-theoretic context optimization`;
}

// ── Commands ──

function cmdServe() {
  const server = new EntrolyMCPServer();
  server.run();
}

function cmdOptimize(args) {
  const config = new EntrolyConfig();
  const engine = new WasmEntrolyEngine();
  const indexPath = path.join(config.checkpointDir, 'index.json.gz');

  // Load persistent index
  if (loadIndex(engine, indexPath)) {
    console.error(`${C.GREEN}✓${C.RESET} Loaded persistent index (${engine.fragment_count()} fragments)`);
  }

  // Auto-index if no fragments
  if (engine.fragment_count() === 0) {
    console.error(`${C.CYAN}⏳${C.RESET} Auto-indexing codebase...`);
    const result = autoIndex(engine);
    console.error(`${C.GREEN}✓${C.RESET} Indexed ${result.files_indexed} files (${result.total_tokens.toLocaleString()} tokens) in ${result.duration_s}s`);
    persistIndex(engine, indexPath);
  }

  const budget = parseInt(args[0] || '128000', 10);
  const query = args.slice(1).join(' ') || '';

  if (!query) {
    console.error(`${C.YELLOW || ''}! Missing task. Usage: entroly-wasm optimize <budget> "<task>"${C.RESET || ''}`);
    console.error(`  Example: entroly-wasm optimize 8000 "fix the auth bug"`);
    process.exit(2);
  }
  if (engine.fragment_count() < 2) {
    console.error(`${C.YELLOW || ''}! Only ${engine.fragment_count()} fragment indexed — cd into a real repo before optimizing.${C.RESET || ''}`);
    process.exit(2);
  }

  engine.advance_turn();
  const result = engine.optimize(budget, query);

  // Persist after optimization
  persistIndex(engine, indexPath);

  console.log(JSON.stringify(result, null, 2));
}

function cmdHealth() {
  const config = new EntrolyConfig();
  const engine = new WasmEntrolyEngine();
  const indexPath = path.join(config.checkpointDir, 'index.json.gz');

  loadIndex(engine, indexPath);
  if (engine.fragment_count() === 0) {
    console.error(`${C.CYAN}⏳${C.RESET} Auto-indexing codebase...`);
    autoIndex(engine);
    persistIndex(engine, indexPath);
  }

  // Don't declare a grade on an empty scan — guards against false 'A' on blank dirs.
  const n = engine.fragment_count();
  if (n < 5) {
    console.error(`${C.YELLOW || ''}! Only ${n} fragment${n===1?'':'s'} indexed — scan something with real code (cd into your repo, then retry).${C.RESET || ''}`);
    process.exit(2);
  }
  const health = engine.analyze_health();
  console.log(typeof health === 'string' ? health : JSON.stringify(health, null, 2));
}

function cmdStats() {
  const config = new EntrolyConfig();
  const engine = new WasmEntrolyEngine();
  loadIndex(engine, path.join(config.checkpointDir, 'index.json.gz'));

  const stats = engine.stats();
  console.log(typeof stats === 'string' ? stats : JSON.stringify(stats, null, 2));
}

function cmdInit() {
  console.log(banner());
  console.log();

  // Detect IDE
  const cwd = process.cwd();
  const cursorDir = path.join(cwd, '.cursor');
  const vscodeDir = path.join(cwd, '.vscode');

  if (fs.existsSync(cursorDir) || fs.existsSync(path.join(os.homedir(), '.cursor'))) {
    console.log(`${C.GREEN}✓${C.RESET} Detected: ${C.BOLD}Cursor${C.RESET}`);
    const mcpConfig = {
      mcpServers: {
        entroly: {
          command: 'npx',
          args: ['-y', 'entroly-wasm', '--serve'],
        },
      },
    };
    const configDir = fs.existsSync(cursorDir) ? cursorDir : path.join(os.homedir(), '.cursor');
    const configPath = path.join(configDir, 'mcp.json');

    // Merge with existing config; backup before overwrite so we never lose user data.
    let existing = {};
    let parseFailed = false;
    if (fs.existsSync(configPath)) {
      try { existing = JSON.parse(fs.readFileSync(configPath, 'utf-8')); }
      catch { parseFailed = true; }
      try { fs.copyFileSync(configPath, configPath + '.entroly-backup'); } catch {}
    }
    existing.mcpServers = { ...existing.mcpServers, ...mcpConfig.mcpServers };

    fs.mkdirSync(configDir, { recursive: true });
    fs.writeFileSync(configPath, JSON.stringify(existing, null, 2));
    if (parseFailed) {
      console.log(`${C.YELLOW || ''}! Existing config was unparseable; original kept at ${configPath}.entroly-backup${C.RESET}`);
    }
    console.log(`${C.GREEN}✓${C.RESET} Written MCP config to ${C.CYAN}${configPath}${C.RESET}`);
  } else if (fs.existsSync(vscodeDir)) {
    console.log(`${C.GREEN}✓${C.RESET} Detected: ${C.BOLD}VS Code${C.RESET}`);
    console.log(`  Add to your MCP settings:`);
    console.log(`  ${C.CYAN}"entroly": { "command": "npx", "args": ["-y", "entroly-wasm", "--serve"] }${C.RESET}`);
  } else {
    console.log(`  No IDE detected. To use Entroly as an MCP server, run:`);
    console.log(`  ${C.CYAN}npx entroly-wasm --serve${C.RESET}`);
  }

  console.log();
  console.log(`  ${C.BOLD}Quick start:${C.RESET}`);
  console.log(`    ${C.CYAN}npx entroly-wasm optimize 8000 "fix the auth bug"${C.RESET}`);
  console.log(`    ${C.CYAN}npx entroly-wasm health${C.RESET}`);
  console.log(`    ${C.CYAN}npx entroly-wasm demo${C.RESET}`);
}

function cmdDemo() {
  console.log(banner());
  console.log();
  console.log(`  ${C.BOLD}Before/After Demo${C.RESET}`);
  console.log();

  const engine = new WasmEntrolyEngine();

  // Create sample fragments
  const samples = [
    { content: 'function authenticate(user, pass) {\n  const hash = bcrypt.hash(pass, 10);\n  return db.users.findOne({ user, hash });\n}', source: 'file:auth.js', tokens: 45, pinned: true },
    { content: 'export const API_KEY = process.env.API_KEY;\nexport const DB_URL = process.env.DATABASE_URL;\nexport const PORT = 3000;', source: 'file:config.js', tokens: 30, pinned: false },
    { content: 'class PaymentProcessor {\n  constructor(stripe) { this.stripe = stripe; }\n  async charge(amount, currency, token) {\n    return this.stripe.charges.create({ amount, currency, source: token });\n  }\n}', source: 'file:payments.js', tokens: 55, pinned: false },
    { content: 'import { test, expect } from "vitest";\ntest("auth works", () => {\n  expect(authenticate("admin", "pass")).toBeTruthy();\n});', source: 'file:auth.test.js', tokens: 35, pinned: false },
    { content: '# API Documentation\n\n## Endpoints\n\n### POST /api/auth\nBody: { username, password }\nResponse: { token, expiresIn }\n\n### POST /api/pay\nBody: { amount, currency, token }', source: 'file:README.md', tokens: 40, pinned: false },
  ];

  for (const s of samples) engine.ingest(s.content, s.source, s.tokens, s.pinned);
  const totalTokens = samples.reduce((sum, s) => sum + s.tokens, 0);

  console.log(`  ${C.GRAY}Ingested ${samples.length} fragments (${totalTokens} tokens total)${C.RESET}`);
  console.log();

  // Without optimization (all fragments)
  console.log(`  ${C.RED}Without Entroly:${C.RESET} ${totalTokens} tokens → send everything`);
  console.log(`  ${C.GRAY}  Cost: $${(totalTokens * 0.000015).toFixed(6)}/call${C.RESET}`);
  console.log();

  // With optimization
  engine.advance_turn();
  const result = engine.optimize(80, 'fix the authentication bug');
  const optimized = result.total_tokens || 0;

  console.log(`  ${C.GREEN}With Entroly:${C.RESET} ${optimized} tokens → mathematically optimal subset`);
  console.log(`  ${C.GRAY}  Cost: $${(optimized * 0.000015).toFixed(6)}/call${C.RESET}`);
  console.log(`  ${C.GREEN}  Saved: ${totalTokens - optimized} tokens (${Math.round((1 - optimized / totalTokens) * 100)}% reduction)${C.RESET}`);
  console.log();

  // Show which fragments were selected
  const selected = result.selected || [];
  if (selected.length) {
    console.log(`  ${C.BOLD}Selected fragments:${C.RESET}`);
    for (const frag of selected) {
      if (frag.variant === 'full') {
        console.log(`    ${C.GREEN}✓${C.RESET} ${frag.source} (${frag.token_count} tokens, relevance=${frag.relevance})`);
      }
    }
  }
}

function cmdGo() {
  console.log(`\n${C.CYAN}${C.BOLD}  Entroly Daemon${C.RESET} — self-evolving context engine\n`);
  console.log(`  ${C.GRAY}Auto-detecting IDE, indexing codebase, starting MCP server...${C.RESET}\n`);

  // Step 1: Init (detect IDE + write config)
  cmdInit();

  // Step 2: Serve (start MCP daemon)
  console.log(`\n  ${C.GREEN}${C.BOLD}Starting MCP server...${C.RESET}\n`);
  cmdServe();
}

function cmdGateways() {
  console.log(banner());
  console.log();
  console.log(`  ${C.BOLD}Chat Gateways${C.RESET} — stream evolution events to Telegram/Discord/Slack\n`);

  // Delegate to the standalone entrypoint in gateways.js
  const gwPath = path.join(__dirname, 'gateways.js');
  require(gwPath);
}

function cmdClean() {
  const config = new EntrolyConfig();
  const dir = config.checkpointDir;
  try {
    fs.rmSync(dir, { recursive: true, force: true });
    console.log(`${C.GREEN}✓${C.RESET} Cleaned: ${dir}`);
  } catch (e) {
    console.error(`${C.RED}✗${C.RESET} Failed to clean: ${e.message}`);
  }
}

function cmdStatus() {
  const config = new EntrolyConfig();
  const indexPath = path.join(config.checkpointDir, 'index.json.gz');
  const hasIndex = fs.existsSync(indexPath);
  console.log(`${C.BOLD}Entroly Status${C.RESET}`);
  console.log(`  Version:    ${VERSION}`);
  console.log(`  Engine:     Wasm (standalone)`);
  console.log(`  Index:      ${hasIndex ? C.GREEN + '✓ persistent index found' + C.RESET : C.YELLOW + '⚠ no index (run optimize first)' + C.RESET}`);
  console.log(`  Checkpoint: ${config.checkpointDir}`);
}

function cmdAutotune(args) {
  const iterations = parseInt(args[0] || '100', 10);
  const benchOnly = args.includes('--bench-only');
  console.log(banner());
  console.log();
  console.log(`  ${C.BOLD}Autotune${C.RESET} — autonomous self-tuning loop`);
  console.log(`  Iterations: ${iterations}${benchOnly ? ' (bench-only)' : ''}`);
  console.log();
  runAutotune(iterations, 5000, benchOnly);
}

function cmdHelp() {
  console.log(banner());
  console.log();
  console.log(`  ${C.BOLD}Usage:${C.RESET} entroly-wasm [command] [options]`);
  console.log();
  console.log(`  ${C.BOLD}Commands:${C.RESET}`);
  console.log(`    ${C.CYAN}go${C.RESET}         ${C.BOLD}One command: detect IDE → index → start daemon${C.RESET} (default)`);
  console.log(`    ${C.CYAN}serve${C.RESET}      Start MCP server (stdio JSON-RPC)`);
  console.log(`    ${C.CYAN}optimize${C.RESET}   Generate optimized context (args: [budget] [query...])`);
  console.log(`    ${C.CYAN}health${C.RESET}     Analyze codebase health (grade A-F)`);
  console.log(`    ${C.CYAN}demo${C.RESET}       Before/after demo showing token savings`);
  console.log(`    ${C.CYAN}gateways${C.RESET}   Stream evolution events to Telegram/Discord/Slack`);
  console.log(`    ${C.CYAN}init${C.RESET}       Auto-detect IDE and generate MCP config`);
  console.log(`    ${C.CYAN}stats${C.RESET}      Show session statistics`);
  console.log(`    ${C.CYAN}status${C.RESET}     Check environment status`);
  console.log(`    ${C.CYAN}autotune${C.RESET}   Run autonomous self-tuning (args: [iterations] [--bench-only])`);
  console.log(`    ${C.CYAN}clean${C.RESET}      Clear cached state`);
  console.log();
  console.log(`  ${C.BOLD}Examples:${C.RESET}`);
  console.log(`    ${C.GRAY}npx entroly-wasm${C.RESET}                              ${C.GRAY}# auto-detect + start${C.RESET}`);
  console.log(`    ${C.GRAY}npx entroly-wasm optimize 8000 "fix the auth bug"${C.RESET}`);
  console.log(`    ${C.GRAY}npx entroly-wasm health${C.RESET}`);
  console.log(`    ${C.GRAY}npx entroly-wasm gateways${C.RESET}`);
}

// ── Main ──
const [,, cmd, ...args] = process.argv;

switch (cmd) {
  case 'go': case undefined: cmdGo(); break;
  case 'serve': case '--serve': case 'server': cmdServe(); break;
  case 'optimize': case 'opt': cmdOptimize(args); break;
  case 'health': cmdHealth(); break;
  case 'stats': cmdStats(); break;
  case 'init': cmdInit(); break;
  case 'demo': cmdDemo(); break;
  case 'gateways': case 'gateway': cmdGateways(); break;
  case 'clean': cmdClean(); break;
  case 'status': cmdStatus(); break;
  case 'autotune': case 'tune': cmdAutotune(args); break;
  case '--version': case '-v': console.log(VERSION); break;
  case '--help': case '-h': cmdHelp(); break;
  default:
    console.error(`Unknown command: ${cmd}`);
    cmdHelp();
    process.exit(1);
}
