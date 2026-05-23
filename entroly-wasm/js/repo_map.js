/**
 * Repo File Map — pure-JS port of entroly/repo_map.py
 * Builds grouped inventory with ownership roles.
 */

'use strict';

const fs = require('fs');
const path = require('path');

const SKIP_DIRS = new Set(['.git', '.venv', '__pycache__', '.pytest_cache', '.ruff_cache', 'target', 'node_modules', '.tmp']);

const PY_ROLES = {
  'server.py': ['primary Python MCP server and product shell', 'python-runtime'],
  'cli.py': ['primary CLI and operator surface', 'python-runtime'],
  'proxy.py': ['HTTP proxy prompt compiler runtime', 'python-runtime'],
  'dashboard.py': ['developer-facing runtime dashboard', 'python-runtime'],
  'vault.py': ['vault persistence and artifact schema', 'python-cogops'],
  'epistemic_router.py': ['ingress routing policy engine', 'python-cogops'],
  'belief_compiler.py': ['Truth to Belief compiler', 'python-cogops'],
  'verification_engine.py': ['belief verification and confidence engine', 'python-cogops'],
  'change_pipeline.py': ['change-driven PR and review pipeline', 'python-cogops'],
  'flow_orchestrator.py': ['canonical flow executor', 'python-cogops'],
  'skill_engine.py': ['dynamic skill synthesis and lifecycle', 'python-cogops'],
  'auto_index.py': ['workspace discovery and raw ingest indexing', 'python-support'],
  'autotune.py': ['parameter tuning and feedback journal', 'python-support'],
  'multimodal.py': ['image, diff, diagram, and voice ingestion', 'python-support'],
};

const RUST_ROLES = {
  'lib.rs': ['core Rust engine and PyO3 export surface', 'rust-runtime'],
  'cogops.rs': ['Rust epistemic engine and CogOps data plane', 'rust-cogops'],
  'knapsack.rs': ['budgeted context selection optimizer', 'rust-core'],
  'entropy.rs': ['information density scoring', 'rust-core'],
  'dedup.rs': ['SimHash deduplication', 'rust-core'],
  'prism.rs': ['reinforcement and spectral optimizer', 'rust-learning'],
  'sast.rs': ['static security analysis engine', 'rust-verification'],
  'health.rs': ['codebase health analysis', 'rust-verification'],
  'cache.rs': ['EGSC cache and retrieval economics', 'rust-core'],
};

const WASM_ROLES = {
  'server.js': ['Node MCP server over WASM engine', 'wasm-runtime'],
  'cli.js': ['Node CLI over WASM engine', 'wasm-runtime'],
  'vault.js': ['pure-JS vault persistence', 'wasm-cogops'],
  'cogops.js': ['pure-JS epistemic engine', 'wasm-cogops'],
  'skills.js': ['pure-JS skill lifecycle', 'wasm-cogops'],
  'workspace.js': ['Node workspace change listener', 'wasm-cogops'],
  'config.js': ['Node configuration wrapper', 'wasm-support'],
  'auto_index.js': ['Node workspace indexing wrapper', 'wasm-support'],
  'checkpoint.js': ['Node checkpoint wrapper', 'wasm-support'],
  'autotune.js': ['Node autotune wrapper', 'wasm-support'],
};

function buildRepoMap(rootDir) {
  const root = path.resolve(rootDir);
  const grouped = { root: [], python: [], 'rust-core': [], wasm: [], tests: [], other: [] };
  const walk = (dir) => {
    let entries;
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
    for (const e of entries) {
      if (SKIP_DIRS.has(e.name)) continue;
      const full = path.join(dir, e.name);
      if (e.isDirectory()) { walk(full); continue; }
      const rel = path.relative(root, full).replace(/\\/g, '/');
      const parts = rel.split('/');
      const name = path.basename(rel);
      let repo = 'other', role = 'support file', category = 'support';
      if (parts.length === 1) {
        repo = 'root';
        role = 'top-level support artifact';
        category = 'root-support';
      } else if (parts[0] === 'entroly' && !parts[1]?.startsWith('entroly')) {
        repo = 'python';
        const r = PY_ROLES[name];
        if (r) { role = r[0]; category = r[1]; }
        else { role = 'Python package support file'; category = 'python-support'; }
      } else if (parts[0] === 'entroly-core') {
        repo = 'rust-core';
        const r = RUST_ROLES[name];
        if (r) { role = r[0]; category = r[1]; }
        else { role = parts[1] === 'src' ? 'Rust core module' : 'Rust core metadata'; category = 'rust-core'; }
      } else if (parts[0] === 'entroly-wasm') {
        repo = 'wasm';
        const r = WASM_ROLES[name];
        if (r) { role = r[0]; category = r[1]; }
        else { role = 'WASM package support file'; category = 'wasm-support'; }
      } else if (parts[0] === 'tests') {
        repo = 'tests'; role = 'integration test'; category = 'python-test';
      } else if (['docs', 'examples', 'bench'].includes(parts[0])) {
        repo = 'other'; role = 'documentation, example, or benchmark asset'; category = 'support';
      }
      grouped[repo].push({ repo, path: rel, role, category });
    }
  };
  walk(root);
  return grouped;
}

function renderRepoMapMarkdown(grouped) {
  const lines = ['# Entroly Repo File Map', '', 'Canonical ownership map across the Python product shell, Rust core, and WASM/JS surface.', ''];
  for (const section of ['root', 'python', 'rust-core', 'wasm', 'tests', 'other']) {
    const entries = grouped[section] || [];
    if (!entries.length) continue;
    lines.push(`## ${section}`, '', '| Path | Role | Category |', '|---|---|---|');
    for (const e of entries) lines.push(`| \`${e.path}\` | ${e.role} | \`${e.category}\` |`);
    lines.push('');
  }
  return lines.join('\n') + '\n';
}

module.exports = { buildRepoMap, renderRepoMapMarkdown };
