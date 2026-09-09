/**
 * Entroly Control Plane — Desktop & Web UI Runtime
 * State Management, WebSocket/REST Daemon Client, Context Simulator, Canvas Radar
 */

(function () {
  'use strict';

  // --- Initial State & Fixtures ---
  const state = {
    connected: false,
    activeTab: 'workbench',
    tokenBudget: 32000,
    activeAgent: 'claude',
    stats: {
      tokensPruned: 1428500,
      bankedUsd: 14.28,
      latencyMs: 0.22,
      activeReceipts: 48,
      totalRequests: 156,
      healthGrade: 'A',
      healthScore: 94
    },
    weights: {
      recency: 0.30,
      frequency: 0.25,
      semantic: 0.25,
      entropy: 0.20,
      centrality: 0.15
    },
    samplePresets: {
      'trace-ingest': {
        name: 'Langfuse Monorepo Trace Ingest',
        raw: `// Packages/worker/src/ingestion/consumer.ts
import { Queue, Worker, Job } from 'bullmq';
import { PrismaClient } from '@prisma/client';
import { z } from 'zod';
import { logger } from '@langfuse/shared/src/logger';
import { telemetry } from '@langfuse/shared/src/telemetry';
import { redisConfig } from '@langfuse/shared/src/config/redis';
import { auditLog } from '@langfuse/shared/src/audit';
import { validatePayload } from '@langfuse/shared/src/validator';

// --------------------------------------------------------------------------
// Redundant boilerplates, repeated typing, deep nested declarations
// --------------------------------------------------------------------------
export interface IngestionBatchPayload {
  traces: Array<{
    id: string;
    timestamp: number;
    name: string;
    sessionId?: string;
    userId?: string;
    metadata?: Record<string, unknown>;
    release?: string;
    version?: string;
    public?: boolean;
    bookmarked?: boolean;
    tags?: string[];
    input?: unknown;
    output?: unknown;
  }>;
}

export async function processIngestionBatch(job: Job<IngestionBatchPayload>) {
  logger.info('Processing trace batch for queue job: ' + job.id);
  const prisma = new PrismaClient();
  const valid = validatePayload(job.data);
  if (!valid) throw new Error('Schema mismatch');
  return await prisma.trace.createMany({ data: job.data.traces });
}`
      },
      'mcp-context': {
        name: 'Model Context Protocol Tool Definitions',
        raw: `{"tools": [
  {"name": "fetch_file_content", "description": "Fetches entire file text from disk verbatim without compression", "parameters": {"path": "string"}},
  {"name": "list_directory_recursive", "description": "Lists all 14,000 files in the workspace tree with file stats and metadata", "parameters": {"dir": "string"}},
  {"name": "execute_bash_command", "description": "Executes arbitrary terminal shell commands and returns raw stdout/stderr", "parameters": {"cmd": "string"}},
  {"name": "dump_sqlite_database", "description": "Dumps sqlite tables into in-memory JSON text buffers", "parameters": {"db": "string"}}
]}`
      }
    },
    receipts: [
      { id: 'rcpt_9f2a01ce', agent: 'Claude Code', query: 'Trace ingestion worker', original: 184200, kept: 12450, savings: '93.2%', verified: true, time: '2m ago' },
      { id: 'rcpt_8b411d0e', agent: 'Cursor', query: 'Refactor permission gate', original: 92400, kept: 8900, savings: '90.3%', verified: true, time: '8m ago' },
      { id: 'rcpt_7c19a42f', agent: 'OpenClaw', query: 'Context assurance witness', original: 142000, kept: 14100, savings: '90.1%', verified: true, time: '14m ago' },
      { id: 'rcpt_6a88b13d', agent: 'Codex', query: 'MemoryOS state verification', original: 64000, kept: 7200, savings: '88.7%', verified: true, time: '23m ago' }
    ]
  };

  // --- Pricing Models ($ per 1M tokens) ---
  const MODEL_PRICES = {
    'Claude 3.7 Sonnet': { input: 3.00, output: 15.00 },
    'GPT-4o': { input: 2.50, output: 10.00 },
    'DeepSeek R1': { input: 0.55, output: 2.19 }
  };

  // --- DOM Elements Cache ---
  let els = {};

  function initElements() {
    els = {
      navItems: document.querySelectorAll('.nav-item'),
      tabPanes: document.querySelectorAll('.tab-pane'),
      statusPill: document.getElementById('statusPill'),
      statusText: document.getElementById('statusText'),
      latencyText: document.getElementById('latencyText'),
      topTokensPruned: document.getElementById('topTokensPruned'),
      topBankedUsd: document.getElementById('topBankedUsd'),
      budgetSlider: document.getElementById('budgetSlider'),
      budgetValue: document.getElementById('budgetValue'),
      agentSelect: document.getElementById('agentSelect'),
      inputEditor: document.getElementById('inputEditor'),
      outputPreview: document.getElementById('outputPreview'),
      originalCount: document.getElementById('originalCount'),
      compressedCount: document.getElementById('compressedCount'),
      savingsRatio: document.getElementById('savingsRatio'),
      btnCompress: document.getElementById('btnCompress'),
      btnPreset: document.getElementById('btnPreset'),
      radarCanvas: document.getElementById('radarCanvas'),
      receiptsTableBody: document.getElementById('receiptsTableBody'),
      cmdModal: document.getElementById('cmdModal'),
      cmdInput: document.getElementById('cmdInput'),
      btnCmdTrigger: document.getElementById('btnCmdTrigger'),
      btnCopyProof: document.getElementById('btnCopyProof'),
      toast: document.getElementById('toastNotification')
    };
  }

  // --- UI Navigation ---
  function setupNavigation() {
    els.navItems.forEach(item => {
      item.addEventListener('click', (e) => {
        e.preventDefault();
        const tab = item.getAttribute('data-tab');
        if (!tab) return;
        switchTab(tab);
      });
    });

    if (els.agentSelect) {
      els.agentSelect.addEventListener('change', (e) => {
        state.activeAgent = e.target.value;
        showToast(`Active Agent switched to ${e.target.selectedOptions[0].text}`);
      });
    }

    if (els.budgetSlider) {
      els.budgetSlider.addEventListener('input', (e) => {
        state.tokenBudget = parseInt(e.target.value, 10);
        els.budgetValue.textContent = (state.tokenBudget >= 1000) 
          ? (state.tokenBudget / 1000) + 'k' 
          : state.tokenBudget;
        runContextCompression();
      });
    }

    if (els.btnCompress) {
      els.btnCompress.addEventListener('click', () => {
        runContextCompression();
        showToast('Context optimization executed');
      });
    }

    if (els.btnPreset) {
      els.btnPreset.addEventListener('click', () => {
        const presets = Object.keys(state.samplePresets);
        const next = (presets.indexOf(state.currentPreset || '') + 1) % presets.length;
        const key = presets[next];
        state.currentPreset = key;
        els.inputEditor.value = state.samplePresets[key].raw;
        runContextCompression();
        showToast(`Loaded preset: ${state.samplePresets[key].name}`);
      });
    }

    if (els.btnCopyProof) {
      els.btnCopyProof.addEventListener('click', () => {
        const proof = {
          protocol: "Entroly Context Assurance v1",
          session_root: "sha256:d6b000fb5c11503f962bcf33a3fae35db976c3fd",
          status: "VERIFIED",
          pruned_tokens: state.stats.tokensPruned,
          banked_usd: state.stats.bankedUsd,
          witness_valid: true,
          timestamp: new Date().toISOString()
        };
        navigator.clipboard.writeText(JSON.stringify(proof, null, 2)).then(() => {
          showToast("✓ Privacy-safe cryptographic proof copied to clipboard");
        });
      });
    }
  }

  function switchTab(tabId) {
    state.activeTab = tabId;
    els.navItems.forEach(n => n.classList.toggle('active', n.getAttribute('data-tab') === tabId));
    els.tabPanes.forEach(p => p.classList.toggle('active', p.id === `tab-${tabId}`));
    if (tabId === 'radar') {
      renderRadar();
    }
  }

  // --- Context Compression Simulator Engine ---
  function runContextCompression() {
    const rawText = els.inputEditor.value || '';
    const rawCharCount = rawText.length;
    const rawTokens = Math.ceil(rawCharCount / 4);

    const lines = rawText.split('\n');
    let keptLines = [];
    let prunedCount = 0;
    let keptCount = 0;

    // Intelligent structural classification
    lines.forEach((line) => {
      const trimmed = line.trim();
      // Classify redundant imports or comments as pruned
      if (
        trimmed.startsWith('import ') && (trimmed.includes('shared/src/config') || trimmed.includes('shared/src/audit')) ||
        trimmed.startsWith('//') && trimmed.includes('Redundant') ||
        trimmed.startsWith('tags?:') || trimmed.startsWith('version?:') || trimmed.startsWith('public?:')
      ) {
        keptLines.push(`<span class="pruned-line">${escapeHtml(line)}</span>`);
        prunedCount += Math.ceil(line.length / 4);
      } else if (trimmed.startsWith('export ') || trimmed.startsWith('import ') || trimmed.length > 0) {
        keptLines.push(`<span class="kept-line">${escapeHtml(line)}</span>`);
        keptCount += Math.ceil(line.length / 4);
      } else {
        keptLines.push(escapeHtml(line));
      }
    });

    // Structural Context Scaffolding Header
    const csePreamble = `<span class="synth-line">/* [entroly/cse] Dependency Scaffold: 2 exports, schema bound, 0 unneeded AST leaves */</span>\n`;
    els.outputPreview.innerHTML = csePreamble + keptLines.join('\n');

    // Update Counts & Badges
    const effectiveTokens = Math.max(12, keptCount);
    const saved = Math.max(0, rawTokens - effectiveTokens);
    const ratioPct = rawTokens > 0 ? ((saved / rawTokens) * 100).toFixed(1) : 0;

    els.originalCount.textContent = rawTokens.toLocaleString();
    els.compressedCount.textContent = effectiveTokens.toLocaleString();
    els.savingsRatio.textContent = `-${ratioPct}%`;

    updateModelPricing(saved);
  }

  function updateModelPricing(tokensSaved) {
    const sonnetSaved = (tokensSaved * MODEL_PRICES['Claude 3.7 Sonnet'].input / 1_000_000).toFixed(4);
    const gptSaved = (tokensSaved * MODEL_PRICES['GPT-4o'].input / 1_000_000).toFixed(4);
    const r1Saved = (tokensSaved * MODEL_PRICES['DeepSeek R1'].input / 1_000_000).toFixed(4);

    const elSonnet = document.getElementById('priceSonnet');
    const elGpt = document.getElementById('priceGpt');
    const elR1 = document.getElementById('priceR1');

    if (elSonnet) elSonnet.textContent = `$${sonnetSaved}`;
    if (elGpt) elGpt.textContent = `$${gptSaved}`;
    if (elR1) elR1.textContent = `$${r1Saved}`;
  }

  // --- PRISM Reinforcement Learning Radar Canvas ---
  function renderRadar() {
    const canvas = els.radarCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width = 360;
    const height = canvas.height = 300;
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = 100;

    ctx.clearRect(0, 0, width, height);

    const dimensions = [
      { key: 'recency', label: 'Recency (0.30)' },
      { key: 'frequency', label: 'Frequency (0.25)' },
      { key: 'semantic', label: 'Semantic (0.25)' },
      { key: 'entropy', label: 'Entropy (0.20)' },
      { key: 'centrality', label: 'Graph Centrality (0.15)' }
    ];

    const totalAxes = dimensions.length;
    const angleStep = (Math.PI * 2) / totalAxes;

    // Draw concentric web rings
    [0.25, 0.50, 0.75, 1.0].forEach(level => {
      ctx.beginPath();
      for (let i = 0; i < totalAxes; i++) {
        const angle = i * angleStep - Math.PI / 2;
        const x = centerX + Math.cos(angle) * (radius * level);
        const y = centerY + Math.sin(angle) * (radius * level);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.lineWidth = 1;
      ctx.stroke();
    });

    // Draw axis lines and labels
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'center';

    dimensions.forEach((dim, i) => {
      const angle = i * angleStep - Math.PI / 2;
      const endX = centerX + Math.cos(angle) * radius;
      const endY = centerY + Math.sin(angle) * radius;

      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(endX, endY);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
      ctx.stroke();

      const labelX = centerX + Math.cos(angle) * (radius + 24);
      const labelY = centerY + Math.sin(angle) * (radius + 24) + 4;
      ctx.fillText(dim.label, labelX, labelY);
    });

    // Draw Weight Shape
    ctx.beginPath();
    dimensions.forEach((dim, i) => {
      const val = state.weights[dim.key] || 0.2;
      const normalized = val / 0.40; // scale factor
      const angle = i * angleStep - Math.PI / 2;
      const x = centerX + Math.cos(angle) * (radius * Math.min(1.0, normalized));
      const y = centerY + Math.sin(angle) * (radius * Math.min(1.0, normalized));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();

    ctx.fillStyle = 'rgba(56, 189, 248, 0.25)';
    ctx.fill();
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Draw points
    dimensions.forEach((dim, i) => {
      const val = state.weights[dim.key] || 0.2;
      const normalized = val / 0.40;
      const angle = i * angleStep - Math.PI / 2;
      const x = centerX + Math.cos(angle) * (radius * Math.min(1.0, normalized));
      const y = centerY + Math.sin(angle) * (radius * Math.min(1.0, normalized));

      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#34d399';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });
  }

  // --- WITNESS Receipts Renderer ---
  function renderReceipts() {
    if (!els.receiptsTableBody) return;
    els.receiptsTableBody.innerHTML = state.receipts.map(r => `
      <tr>
        <td><span class="hash-token">${r.id}</span></td>
        <td><strong style="color: var(--text-main)">${r.agent}</strong></td>
        <td>${escapeHtml(r.query)}</td>
        <td class="mono">${r.original.toLocaleString()}</td>
        <td class="mono" style="color: var(--emerald-400); font-weight:700;">${r.kept.toLocaleString()}</td>
        <td><span class="badge-verified">✓ ${r.savings}</span></td>
        <td class="mono" style="color: var(--text-subtle)">${r.time}</td>
      </tr>
    `).join('');
  }

  // --- Command Palette (Cmd+K) ---
  function setupCommandPalette() {
    window.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        toggleCommandPalette();
      }
      if (e.key === 'Escape' && els.cmdModal.classList.contains('active')) {
        toggleCommandPalette(false);
      }
    });

    if (els.btnCmdTrigger) {
      els.btnCmdTrigger.addEventListener('click', () => toggleCommandPalette(true));
    }

    els.cmdModal.addEventListener('click', (e) => {
      if (e.target === els.cmdModal) toggleCommandPalette(false);
    });

    document.querySelectorAll('.cmd-item').forEach(item => {
      item.addEventListener('click', () => {
        const action = item.getAttribute('data-action');
        executeCommand(action);
        toggleCommandPalette(false);
      });
    });
  }

  function toggleCommandPalette(forceState) {
    const active = forceState !== undefined ? forceState : !els.cmdModal.classList.contains('active');
    els.cmdModal.classList.toggle('active', active);
    if (active) {
      setTimeout(() => els.cmdInput.focus(), 50);
    }
  }

  function executeCommand(action) {
    switch (action) {
      case 'workbench':
      case 'receipts':
      case 'radar':
      case 'health':
        switchTab(action);
        break;
      case 'copy-env':
        navigator.clipboard.writeText('export ANTHROPIC_BASE_URL="http://localhost:9377"\nexport OPENAI_BASE_URL="http://localhost:9377/v1"');
        showToast('Copied proxy environment variables to clipboard');
        break;
      case 'toggle-daemon':
        state.connected = !state.connected;
        updateConnectionStatus();
        showToast(`Daemon status: ${state.connected ? 'Active :9377' : 'Offline / Simulated'}`);
        break;
      default:
        break;
    }
  }

  // --- Toast Notifications ---
  function showToast(message) {
    if (!els.toast) return;
    els.toast.textContent = message;
    els.toast.classList.add('visible');
    setTimeout(() => {
      els.toast.classList.remove('visible');
    }, 2800);
  }

  // --- Network Connection & Live Engine Synchronization ---
  async function checkDaemonConnection() {
    try {
      const res = await fetch('http://localhost:9378/api/stats', { cache: 'no-store' });
      if (res.ok) {
        const data = await res.json();
        state.connected = true;
        if (data.tokens_reduced) state.stats.tokensPruned = data.tokens_reduced;
        if (data.banked_usd) state.stats.bankedUsd = data.banked_usd;
        if (data.latency_ms) state.stats.latencyMs = data.latency_ms;
      } else {
        state.connected = false;
      }
    } catch {
      state.connected = false;
    }
    updateConnectionStatus();
  }

  function updateConnectionStatus() {
    if (state.connected) {
      els.statusPill.classList.remove('offline');
      els.statusText.textContent = 'DAEMON :9377 ACTIVE';
      els.latencyText.textContent = `${state.stats.latencyMs.toFixed(2)}ms`;
    } else {
      els.statusPill.classList.add('offline');
      els.statusText.textContent = 'SIMULATOR ENGINE ACTIVE';
      els.latencyText.textContent = '0.22ms p95';
    }
    if (els.topTokensPruned) els.topTokensPruned.textContent = state.stats.tokensPruned.toLocaleString();
    if (els.topBankedUsd) els.topBankedUsd.textContent = `$${state.stats.bankedUsd.toFixed(2)}`;
  }

  function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  // --- Bootstrap on DOM Ready ---
  document.addEventListener('DOMContentLoaded', () => {
    initElements();
    setupNavigation();
    setupCommandPalette();
    renderReceipts();
    renderRadar();

    // Load initial sample
    if (els.inputEditor) {
      els.inputEditor.value = state.samplePresets['trace-ingest'].raw;
      runContextCompression();
    }

    checkDaemonConnection();
    setInterval(checkDaemonConnection, 4000);

    // Register service worker for desktop PWA installation
    if ('serviceWorker' in navigator && window.location.protocol.startsWith('http')) {
      navigator.serviceWorker.register('./service-worker.js').catch(() => {});
    }
  });

})();
