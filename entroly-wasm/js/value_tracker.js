/**
 * ValueTracker — JS port of entroly/value_tracker.py (schema v4)
 *
 * Persistent, evidence-classified value accounting. Provider-bound reductions
 * may support modeled cost avoidance; local-only and legacy reductions do not.
 * The bounded evolution budget uses only provider-classified value:
 *
 *     C_spent(t)  ≤  τ · S_provider(t)         (τ = 5%)
 *
 * CROSS-RUNTIME CONTRACT: this writes the SAME file, SAME directory and
 * SAME JSON schema as the Python tracker so that one shared dashboard
 * shows value no matter whether the user installed via pip (MCP/proxy),
 * the SDK, or npm (this wasm runtime). The directory resolution and the
 * UTC date-key formats below MUST stay byte-identical to
 * entroly/value_tracker.py — diverging them silently re-creates the
 * "blank dashboard for npm users" bug.
 *
 * Atomic-write safe; survives process restarts.
 */

'use strict';

const fs = require('fs');
const path = require('path');
const os = require('os');

const EVOLUTION_TAX_RATE = 0.05;
const FILE_NAME = 'value_tracker.json';
const ACTIVITY_NAME = 'activity.jsonl';
const SCHEMA_VERSION = 4;
const PRICING_AS_OF = '2026-05';
const MAX_DAILY = 90, MAX_WEEKLY = 52, MAX_MONTHLY = 24, MAX_ACTIVITY = 200;

// Per-model $/1M tokens — kept in sync with entroly/value_tracker.py (×1000).
const COST_PER_M = {
  default: 3.0,
  'gpt-4o': 2.5, 'gpt-4o-mini': 0.15, 'gpt-4-turbo': 10.0, 'gpt-4': 30.0,
  'gpt-3.5-turbo': 0.5, 'o1': 15.0, 'o1-mini': 3.0, 'o3': 10.0,
  'o3-mini': 1.1, 'o4-mini': 1.1,
  'claude-opus-4': 15.0, 'claude-sonnet-4': 3.0, 'claude-haiku-4': 0.8,
  'claude-3-5-sonnet': 3.0, 'claude-3-5-haiku': 0.8,
  'gemini-2.5-pro': 1.25, 'gemini-2.5-flash': 0.075,
  'gemini-1.5-pro': 1.25, 'gemini-1.5-flash': 0.075,
};

const MODEL_ALIASES = {
  'claude-3-opus': 'claude-opus-4',
  'claude-3-sonnet': 'claude-sonnet-4',
  'claude-3-haiku': 'claude-haiku-4',
  'claude-3.5-sonnet': 'claude-3-5-sonnet',
  'claude-3.5-haiku': 'claude-3-5-haiku',
};

function estimateCost(tokens, model = '') {
  let m = (model || '').toLowerCase();
  for (const [alias, canonical] of Object.entries(MODEL_ALIASES)) {
    if (m.startsWith(alias)) { m = canonical + m.slice(alias.length); break; }
  }
  const keys = Object.keys(COST_PER_M).filter(k => k !== 'default')
    .sort((a, b) => b.length - a.length);
  const key = (m && keys.find(k => m.startsWith(k)));
  if (!key && model) {
    if (!estimateCost._warned) estimateCost._warned = new Set();
    if (!estimateCost._warned.has(model)) {
      estimateCost._warned.add(model);
      try { console.warn(`[entroly] unknown model '${model}'; using default $${COST_PER_M.default}/M`); } catch (_) {}
    }
  }
  return (tokens / 1_000_000) * COST_PER_M[key || 'default'];
}

function hasPricedModel(model = '') {
  if (!model) return false;
  let normalized = String(model).toLowerCase();
  for (const [alias, canonical] of Object.entries(MODEL_ALIASES)) {
    if (normalized.startsWith(alias)) {
      normalized = canonical + normalized.slice(alias.length);
      break;
    }
  }
  return Object.keys(COST_PER_M)
    .filter(key => key !== 'default')
    .some(key => normalized.startsWith(key));
}

// ── UTC date keys — MUST match Python time.gmtime()-based keys ──────────

function _pad(n) { return String(n).padStart(2, '0'); }

function _dayKey(d = new Date()) {
  return `${d.getUTCFullYear()}-${_pad(d.getUTCMonth() + 1)}-${_pad(d.getUTCDate())}`;
}

function _monthKey(d = new Date()) {
  return `${d.getUTCFullYear()}-${_pad(d.getUTCMonth() + 1)}`;
}

function _weekKey(d = new Date()) {
  // ISO-8601 week, matching Python datetime.date.isocalendar().
  const t = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()));
  const day = t.getUTCDay() || 7;          // Mon=1..Sun=7
  t.setUTCDate(t.getUTCDate() + 4 - day);  // nearest Thursday
  const isoYear = t.getUTCFullYear();
  const yearStart = new Date(Date.UTC(isoYear, 0, 1));
  const week = Math.ceil((((t - yearStart) / 86400000) + 1) / 7);
  return `${isoYear}-W${_pad(week)}`;
}

function atomicWrite(filePath, content) {
  const tmp = filePath + '.tmp-' + process.pid;
  fs.writeFileSync(tmp, content, 'utf8');
  fs.renameSync(tmp, filePath);
}

class ValueTracker {
  constructor(dataDir = null) {
    // Honor ENTROLY_DIR exactly like Python's _default_dir(); else
    // ~/.entroly. NOT cwd/.entroly — that was the npm/py path split.
    this._dir = dataDir
      || process.env.ENTROLY_DIR
      || path.join(os.homedir(), '.entroly');
    fs.mkdirSync(this._dir, { recursive: true });
    this._path = path.join(this._dir, FILE_NAME);
    this._activityPath = path.join(this._dir, ACTIVITY_NAME);
    this._data = this._migrate(this._load());
    this._activity = this._loadActivity();
  }

  _load() {
    if (fs.existsSync(this._path)) {
      try {
        const raw = JSON.parse(fs.readFileSync(this._path, 'utf8'));
        if (raw && typeof raw === 'object' && 'version' in raw) return raw;
      } catch (_) { /* fall through */ }
    }
    return this._defaults();
  }

  _defaults() {
    const now = Date.now() / 1000;
    return {
      version: SCHEMA_VERSION,
      lifetime: {
        tokens_saved: 0, cost_saved_usd: 0.0,
        requests_optimized: 0, requests_total: 0, duplicates_caught: 0,
        provider_tokens_saved: 0, provider_cost_avoided_usd: 0.0,
        provider_requests: 0, provider_requests_optimized: 0,
        provider_unpriced_tokens: 0, provider_unpriced_requests: 0,
        local_tokens_reduced: 0, local_operations: 0,
        unclassified_tokens_reduced: 0,
        unclassified_cost_estimate_usd: 0.0,
        unclassified_operations: 0,
        first_seen: now, last_seen: now,
        evolution_spent_usd: 0.0, evolution_attempts: 0,
        evolution_successes: 0,
        hallucinations_blocked: 0, routing_saved_usd: 0.0,
        routing_decisions: 0,
      },
      daily: {}, weekly: {}, monthly: {},
    };
  }

  _migrate(data) {
    // Backward-compatible forward migration: preserve old mixed counters as
    // unclassified instead of presenting them as provider-bound savings.
    const base = this._defaults();
    data.lifetime = data.lifetime || {};
    const previousVersion = Number(data.version || 0);
    if (previousVersion < 4) {
      const lt = data.lifetime;
      if (!('unclassified_tokens_reduced' in lt)) {
        lt.unclassified_tokens_reduced = Number(lt.tokens_saved || 0);
      }
      if (!('unclassified_cost_estimate_usd' in lt)) {
        lt.unclassified_cost_estimate_usd = Number(lt.cost_saved_usd || 0);
      }
      if (!('unclassified_operations' in lt)) {
        lt.unclassified_operations = Number(lt.requests_optimized || 0);
      }
      lt.cost_saved_usd = Number(lt.provider_cost_avoided_usd || 0);
    }
    for (const [k, v] of Object.entries(base.lifetime)) {
      if (!(k in data.lifetime)) data.lifetime[k] = v;
    }
    for (const b of ['daily', 'weekly', 'monthly']) {
      if (!data[b] || typeof data[b] !== 'object') data[b] = {};
      if (previousVersion < 4) {
        for (const row of Object.values(data[b])) {
          if (!row || typeof row !== 'object') continue;
          if (!('unclassified_tokens_reduced' in row)) {
            row.unclassified_tokens_reduced = Number(row.tokens_saved || 0);
          }
          if (!('unclassified_cost_estimate_usd' in row)) {
            row.unclassified_cost_estimate_usd = Number(row.cost_saved || 0);
          }
          if (!('unclassified_operations' in row)) {
            row.unclassified_operations = Number(row.requests || 0);
          }
          row.cost_saved = Number(row.provider_cost_avoided_usd || 0);
          const periodDefaults = {
            provider_tokens_saved: 0,
            provider_cost_avoided_usd: 0.0,
            provider_requests: 0,
            provider_requests_optimized: 0,
            provider_unpriced_tokens: 0,
            provider_unpriced_requests: 0,
            local_tokens_reduced: 0,
            local_operations: 0,
          };
          for (const [field, value] of Object.entries(periodDefaults)) {
            if (!(field in row)) row[field] = value;
          }
        }
      }
    }
    data.version = SCHEMA_VERSION;
    return data;
  }

  _loadActivity() {
    if (!fs.existsSync(this._activityPath)) return [];
    try {
      const out = [];
      for (const line of fs.readFileSync(this._activityPath, 'utf8').split('\n')) {
        const s = line.trim();
        if (!s) continue;
        try { out.push(JSON.parse(s)); } catch (_) { /* skip */ }
      }
      return out.slice(-MAX_ACTIVITY);
    } catch (_) { return []; }
  }

  _save() {
    try { atomicWrite(this._path, JSON.stringify(this._data, null, 2)); }
    catch (_) { /* best-effort */ }
  }

  _saveActivity() {
    try {
      this._activity = this._activity.slice(-MAX_ACTIVITY);
      const content = this._activity.map(e => JSON.stringify(e)).join('\n')
        + (this._activity.length ? '\n' : '');
      atomicWrite(this._activityPath, content);
    } catch (_) { /* best-effort */ }
  }

  _bump(bucketName, key, tokens, cost, channel, optimized, providerPriced) {
    const bucket = this._data[bucketName] || (this._data[bucketName] = {});
    const defaults = {
      tokens_saved: 0, cost_saved: 0.0, requests: 0,
      provider_tokens_saved: 0, provider_cost_avoided_usd: 0.0,
      provider_requests: 0, provider_requests_optimized: 0,
      provider_unpriced_tokens: 0, provider_unpriced_requests: 0,
      local_tokens_reduced: 0, local_operations: 0,
      unclassified_tokens_reduced: 0,
      unclassified_cost_estimate_usd: 0.0,
      unclassified_operations: 0,
    };
    if (!bucket[key]) bucket[key] = { ...defaults };
    const row = bucket[key];
    for (const [field, value] of Object.entries(defaults)) {
      if (!(field in row)) row[field] = value;
    }
    row.tokens_saved += tokens;
    row.requests += 1;
    if (channel === 'provider') {
      row.cost_saved = +(row.cost_saved + cost).toFixed(6);
      row.provider_tokens_saved += tokens;
      row.provider_cost_avoided_usd = +(row.provider_cost_avoided_usd + cost).toFixed(6);
      row.provider_requests += 1;
      if (optimized) row.provider_requests_optimized += 1;
      if (!providerPriced) {
        row.provider_unpriced_tokens += tokens;
        row.provider_unpriced_requests += 1;
      }
    } else if (channel === 'local') {
      row.local_tokens_reduced += tokens;
      row.local_operations += 1;
    } else {
      row.unclassified_tokens_reduced += tokens;
      row.unclassified_cost_estimate_usd = +(row.unclassified_cost_estimate_usd + cost).toFixed(6);
      row.unclassified_operations += 1;
    }
    const limit = { daily: MAX_DAILY, weekly: MAX_WEEKLY, monthly: MAX_MONTHLY }[bucketName];
    const ks = Object.keys(bucket);
    if (ks.length > limit) {
      ks.sort();
      for (const old of ks.slice(0, ks.length - limit)) delete bucket[old];
    }
  }

  record({ tokensSaved = 0, model = '', duplicates = 0, optimized = true, source = 'npm' } = {}) {
    tokensSaved = Math.max(0, Number(tokensSaved) || 0);
    const normalizedSource = String(source || 'unclassified').toLowerCase();
    const channel = ['provider', 'proxy', 'gateway'].includes(normalizedSource)
      ? 'provider'
      : ['sdk', 'npm', 'mcp', 'local'].includes(normalizedSource)
        ? 'local' : 'unclassified';
    const estimatedCost = estimateCost(tokensSaved, model);
    const providerPriced = channel !== 'provider' || hasPricedModel(model);
    const cost = providerPriced ? estimatedCost : 0;
    const now = new Date();
    const lt = this._data.lifetime;
    lt.tokens_saved += tokensSaved;
    lt.requests_total = (lt.requests_total || 0) + 1;
    if (optimized) lt.requests_optimized += 1;
    lt.duplicates_caught = (lt.duplicates_caught || 0) + duplicates;
    lt.last_seen = Date.now() / 1000;
    if (channel === 'provider') {
      lt.cost_saved_usd = +(lt.cost_saved_usd + cost).toFixed(6);
      lt.provider_tokens_saved += tokensSaved;
      lt.provider_cost_avoided_usd = +(lt.provider_cost_avoided_usd + cost).toFixed(6);
      lt.provider_requests += 1;
      if (optimized) lt.provider_requests_optimized += 1;
      if (!providerPriced) {
        lt.provider_unpriced_tokens += tokensSaved;
        lt.provider_unpriced_requests += 1;
      }
    } else if (channel === 'local') {
      lt.local_tokens_reduced += tokensSaved;
      lt.local_operations += 1;
    } else {
      lt.unclassified_tokens_reduced += tokensSaved;
      lt.unclassified_cost_estimate_usd = +(lt.unclassified_cost_estimate_usd + estimatedCost).toFixed(6);
      lt.unclassified_operations += 1;
    }
    this._bump('daily', _dayKey(now), tokensSaved, cost, channel, optimized, providerPriced);
    this._bump('weekly', _weekKey(now), tokensSaved, cost, channel, optimized, providerPriced);
    this._bump('monthly', _monthKey(now), tokensSaved, cost, channel, optimized, providerPriced);
    this._save();
    this._activity.push({
      ts: +(Date.now() / 1000).toFixed(3),
      kind: 'optimize',
      summary: `Optimized request: saved ${tokensSaved.toLocaleString()} tokens`
        + (model ? ` (${model})` : ''),
      tokens_saved: tokensSaved,
      cost_saved_usd: +(channel === 'provider' ? cost : 0).toFixed(6),
      modeled_cost_avoided_usd: +(channel === 'provider' ? cost : 0).toFixed(6),
      model: model || '',
      duplicates,
      source: normalizedSource,
      measurement_channel: channel,
    });
    this._saveActivity();
    return { tokensSaved, costSaved: cost };
  }

  recordEvent(kind, summary, opts = {}) {
    try {
      const row = {
        ts: +(Date.now() / 1000).toFixed(3),
        kind: String(kind),
        summary: String(summary).slice(0, 240),
        source: opts.source || 'npm',
      };
      if (opts.tokensSaved) row.tokens_saved = opts.tokensSaved | 0;
      if (opts.costSavedUsd) row.cost_saved_usd = +Number(opts.costSavedUsd).toFixed(6);
      if (opts.model) row.model = opts.model;
      this._activity.push(row);
      this._saveActivity();
    } catch (_) { /* fail-open */ }
  }

  recordHallucinationBlocked(n = 1, detail = '') {
    try {
      this._data.lifetime.hallucinations_blocked =
        (this._data.lifetime.hallucinations_blocked || 0) + (n | 0);
      this._save();
      this.recordEvent('hallucination',
        detail || `Blocked ${n} unsupported claim(s)`, { source: 'witness' });
    } catch (_) { /* fail-open */ }
  }

  recordRoutingSaving(costSavedUsd, chosenModel = '', detail = '') {
    try {
      const lt = this._data.lifetime;
      lt.routing_saved_usd = +((lt.routing_saved_usd || 0) + Number(costSavedUsd)).toFixed(6);
      lt.routing_decisions = (lt.routing_decisions || 0) + 1;
      this._save();
      this.recordEvent('routing',
        detail || `Routed to ${chosenModel || 'cheaper model'}`,
        { source: 'ravs', costSavedUsd, model: chosenModel });
    } catch (_) { /* fail-open */ }
  }

  getEvolutionBudget() {
    const lt = this._data.lifetime;
    const lifetimeSaved = lt.provider_cost_avoided_usd || 0;
    const totalSpent = lt.evolution_spent_usd || 0;
    const totalEarned = lifetimeSaved * EVOLUTION_TAX_RATE;
    const available = Math.max(0, totalEarned - totalSpent);
    return {
      availableUsd: +available.toFixed(6),
      totalEarnedUsd: +totalEarned.toFixed(6),
      totalSpentUsd: +totalSpent.toFixed(6),
      canEvolve: available > 0.001,
      taxRate: EVOLUTION_TAX_RATE,
    };
  }

  recordEvolutionSpend(costUsd, success = false) {
    const lt = this._data.lifetime;
    const lifetimeSaved = lt.provider_cost_avoided_usd || 0;
    const currentSpent = lt.evolution_spent_usd || 0;
    const totalEarned = lifetimeSaved * EVOLUTION_TAX_RATE;
    const available = totalEarned - currentSpent;
    if (costUsd > available + 0.001) {
      return { status: 'rejected', remainingUsd: +Math.max(0, available).toFixed(6) };
    }
    lt.evolution_spent_usd = +(currentSpent + costUsd).toFixed(6);
    lt.evolution_attempts = (lt.evolution_attempts || 0) + 1;
    if (success) lt.evolution_successes = (lt.evolution_successes || 0) + 1;
    this._save();
    return {
      status: 'recorded',
      remainingUsd: +Math.max(0, available - costUsd).toFixed(6),
    };
  }

  // ── Read APIs for the entroly_dashboard MCP tool ───────────────────────

  _sortedBucket(name, lastN) {
    const b = this._data[name] || {};
    return Object.keys(b).sort().slice(-lastN).map(k => ({ key: k, ...b[k] }));
  }

  getTrends() {
    return {
      lifetime: { ...this._data.lifetime },
      daily: this._sortedBucket('daily', 30),
      weekly: this._sortedBucket('weekly', 12),
      monthly: this._sortedBucket('monthly', 12),
      activity: this._activity.slice(-50).reverse(),
    };
  }

  getValueReceipt() {
    const lifetime = { ...this._data.lifetime };
    const daily = this._sortedBucket('daily', 90);
    const providerDays = daily.filter(row => Number(row.provider_requests || 0) > 0).length;
    const localDays = daily.filter(row => Number(row.local_operations || 0) > 0).length;
    return {
      schema_version: 'entroly.value-receipt.v1',
      provider_path: {
        requests_observed: Number(lifetime.provider_requests || 0),
        requests_optimized: Number(lifetime.provider_requests_optimized || 0),
        input_tokens_reduced: Number(lifetime.provider_tokens_saved || 0),
        modeled_input_cost_avoided_usd: Number(
          Number(lifetime.provider_cost_avoided_usd || 0).toFixed(6)
        ),
        active_days: providerDays,
        unpriced_requests: Number(lifetime.provider_unpriced_requests || 0),
        unpriced_input_tokens: Number(lifetime.provider_unpriced_tokens || 0),
        evidence: 'Pre/post token counts on provider-bound requests; modeled dollars are not invoices. Requests without an explicit catalog match remain unpriced.',
      },
      local_operations: {
        operations: Number(lifetime.local_operations || 0),
        tokens_reduced: Number(lifetime.local_tokens_reduced || 0),
        active_days: localDays,
        dollar_claimed_usd: 0.0,
        evidence: 'npm, MCP, SDK, and local reductions; provider delivery is not observable.',
      },
      legacy_unclassified: {
        operations: Number(lifetime.unclassified_operations || 0),
        tokens_reduced: Number(lifetime.unclassified_tokens_reduced || 0),
        historical_cost_estimate_usd: Number(
          Number(lifetime.unclassified_cost_estimate_usd || 0).toFixed(6)
        ),
        dollar_claimed_usd: 0.0,
        evidence: 'Preserved unknown-source history; excluded from provider savings.',
      },
      trust_signals: {
        unsupported_claims_blocked: Number(lifetime.hallucinations_blocked || 0),
        routing_decisions: Number(lifetime.routing_decisions || 0),
        modeled_routing_cost_avoided_usd: Number(
          Number(lifetime.routing_saved_usd || 0).toFixed(6)
        ),
      },
      pricing: { source: 'bundled', as_of: PRICING_AS_OF },
      generated_at_unix: +(Date.now() / 1000).toFixed(3),
    };
  }

  getActivity(lastN = 50) {
    return this._activity.slice(-lastN).reverse();
  }

  stats() {
    return {
      lifetime: { ...this._data.lifetime },
      budget: this.getEvolutionBudget(),
      activity: this.getActivity(20),
    };
  }
}

// Module-level singleton, mirroring Python get_tracker().
let _tracker = null;
function getTracker() {
  if (!_tracker) _tracker = new ValueTracker();
  return _tracker;
}

module.exports = {
  ValueTracker, EVOLUTION_TAX_RATE, estimateCost, getTracker,
};
