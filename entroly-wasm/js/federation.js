/**
 * Federation — Privacy-Preserving Weight Sharing (JavaScript)
 * ============================================================
 *
 * Enables entroly-wasm users to share learned archetype weights
 * across installations via GitHub Issues API. Zero infrastructure cost.
 *
 * What gets shared:
 *   - Archetype label (e.g. "js_monorepo", "react_frontend")
 *   - 4 weight floats (recency, frequency, semantic, entropy)
 *   - Sample count + confidence
 *   - All noised via Gaussian DP (ε=1.0)
 *
 * What is NEVER shared:
 *   - Source code, file paths, or any identifiable data
 *   - Client identity (uses random UUID, not username-derived)
 *
 * @module federation
 */

'use strict';

const fs = require('fs');
const path = require('path');
const https = require('https');
const crypto = require('crypto');

// ── Constants ──
const FEDERATION_REPO = 'juyterman1000/entroly';
const FEDERATION_ISSUE_TITLE = '[federation] Weight Exchange';
const GITHUB_API = 'api.github.com';

const DEFAULT_EPSILON = 1.0;       // Privacy budget per contribution
const WEIGHT_CLIP = [0.0, 1.0];    // Clip before noising
const MIN_CONFIDENCE = 0.7;        // Don't share until locally confident
const MIN_SAMPLE_COUNT = 5;        // Need ≥5 episodes
const TRIM_FRACTION = 0.10;        // Trim top/bottom 10% in aggregation

// Weight keys that participate
const WEIGHT_KEYS = ['w_r', 'w_f', 'w_s', 'w_e'];


// ── Anonymous Client ID ──

/**
 * Generate or load a truly anonymous client identifier.
 * Uses random UUID persisted to disk — no PII derivation.
 */
function getClientId(dataDir) {
  const idPath = path.join(dataDir, '.federation_id');
  try {
    if (fs.existsSync(idPath)) {
      const stored = fs.readFileSync(idPath, 'utf-8').trim();
      if (stored.length >= 32) return stored;
    }
  } catch {}

  // Generate fresh random ID
  const freshId = crypto.createHash('sha256')
    .update(crypto.randomUUID())
    .digest('hex');
  try {
    fs.mkdirSync(path.dirname(idPath), { recursive: true });
    fs.writeFileSync(idPath, freshId);
  } catch {}
  return freshId;
}


// ── Differential Privacy ──

/**
 * Calibrate Gaussian noise sigma for (ε, δ=1e-5)-DP.
 * From Balle et al. (ICML 2018): σ = Δ · √(2ln(1.25/δ)) / ε
 */
function calibrateSigma(epsilon = DEFAULT_EPSILON, sensitivity = 1.0) {
  const delta = 1e-5;
  return sensitivity * Math.sqrt(2 * Math.log(1.25 / delta)) / epsilon;
}

/**
 * Add Gaussian noise to a weight value.
 */
function addNoise(value, sigma) {
  // Box-Muller transform for Gaussian
  const u1 = Math.random() || 1e-10;
  const u2 = Math.random();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return Math.max(WEIGHT_CLIP[0], Math.min(WEIGHT_CLIP[1], value + sigma * z));
}


// ── Contribution Packet ──

/**
 * Create a DP-noised contribution packet from current weights.
 */
function createContribution(clientId, archetype, weights, sampleCount, confidence) {
  if (confidence < MIN_CONFIDENCE || sampleCount < MIN_SAMPLE_COUNT) {
    return null; // Not confident enough to share
  }

  const sigma = calibrateSigma();
  const noisedWeights = {};
  for (const k of WEIGHT_KEYS) {
    noisedWeights[k] = addNoise(weights[k] ?? 0.25, sigma);
  }

  return {
    version: '1.0',
    client_id: clientId,
    archetype: archetype || 'unknown',
    weights: noisedWeights,
    sample_count: sampleCount,
    confidence: Math.round(confidence * 1000) / 1000,
    noise_sigma: Math.round(sigma * 10000) / 10000,
    dp_epsilon: DEFAULT_EPSILON,
    timestamp: Date.now() / 1000,
  };
}


// ── Trimmed Mean Aggregation ──

/**
 * Byzantine-resilient aggregation: trim top/bottom 10%, average rest.
 */
function trimmedMean(values, trimFrac = TRIM_FRACTION) {
  if (values.length < 3) {
    return values.reduce((a, b) => a + b, 0) / values.length;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const trimCount = Math.floor(sorted.length * trimFrac);
  const trimmed = sorted.slice(trimCount, sorted.length - trimCount);
  if (trimmed.length === 0) return sorted[Math.floor(sorted.length / 2)];
  return trimmed.reduce((a, b) => a + b, 0) / trimmed.length;
}


// ── GitHub Transport ──

class GitHubTransport {
  /**
   * @param {Object} options
   * @param {string} [options.repo] - GitHub repo (owner/name)
   * @param {string} [options.token] - PAT (prefer ENTROLY_FEDERATION_BOT)
   */
  constructor(options = {}) {
    this._repo = options.repo || FEDERATION_REPO;
    this._token = options.token
      || process.env.ENTROLY_FEDERATION_BOT
      || process.env.ENTROLY_GITHUB_TOKEN
      || null;
    this._issueNumber = null;
  }

  get canWrite() { return !!this._token; }
  get canRead() { return true; } // Public repos are readable without auth

  /**
   * Make an HTTPS request to GitHub API.
   * @private
   */
  _request(method, urlPath, body = null) {
    return new Promise((resolve, reject) => {
      const headers = {
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'entroly-wasm-federation/1.0',
      };
      if (this._token) headers['Authorization'] = `token ${this._token}`;
      if (body) headers['Content-Type'] = 'application/json';

      const options = {
        hostname: GITHUB_API,
        path: urlPath,
        method,
        headers,
        timeout: 15000,
      };

      const req = https.request(options, (res) => {
        let data = '';
        res.on('data', chunk => data += chunk);
        res.on('end', () => {
          try {
            resolve({ status: res.statusCode, data: JSON.parse(data) });
          } catch {
            resolve({ status: res.statusCode, data: null });
          }
        });
      });

      req.on('error', reject);
      req.on('timeout', () => { req.destroy(); reject(new Error('timeout')); });

      if (body) req.write(JSON.stringify(body));
      req.end();
    });
  }

  /**
   * Find or create the federation exchange issue.
   * @returns {Promise<number|null>} Issue number
   */
  async findOrCreateIssue() {
    if (this._issueNumber) return this._issueNumber;

    try {
      // Search for existing
      const searchPath = `/repos/${this._repo}/issues?labels=federation&state=open&per_page=1`;
      const search = await this._request('GET', searchPath);
      if (search.data && search.data.length > 0) {
        this._issueNumber = search.data[0].number;
        return this._issueNumber;
      }

      // Create if we have write access
      if (!this.canWrite) return null;

      const createPath = `/repos/${this._repo}/issues`;
      const result = await this._request('POST', createPath, {
        title: FEDERATION_ISSUE_TITLE,
        body: '# 🌐 Entroly Federated Weight Exchange\n\n' +
              'Transport layer for federated archetype learning.\n\n' +
              '**Each comment contains a DP-noised contribution packet.**\n' +
              '- All weights protected by Gaussian DP (ε=1.0)\n' +
              '- No source code, paths, or PII shared\n' +
              '- Auto-aggregated via trimmed-mean\n\n' +
              'Do not edit or delete comments — they are machine-managed.',
        labels: ['federation'],
      });

      if (result.data && result.data.number) {
        this._issueNumber = result.data.number;
        return this._issueNumber;
      }
    } catch (e) {
      // Non-fatal
    }
    return null;
  }

  /**
   * Push a contribution packet as an Issue comment.
   * @param {Object} packet - Contribution packet
   * @returns {Promise<boolean>}
   */
  async push(packet) {
    if (!this.canWrite) return false;

    const issue = await this.findOrCreateIssue();
    if (!issue) return false;

    try {
      const body = '```json\n' + JSON.stringify(packet, null, 2) + '\n```';
      const commentPath = `/repos/${this._repo}/issues/${issue}/comments`;
      const result = await this._request('POST', commentPath, { body });
      return result.status === 200 || result.status === 201;
    } catch {
      return false;
    }
  }

  /**
   * Pull contribution packets from Issue comments.
   * @param {number} [sinceHours=48] - Look back this many hours
   * @returns {Promise<Object[]>} Array of contribution packets
   */
  async pull(sinceHours = 48) {
    const issue = await this.findOrCreateIssue();
    if (!issue) return [];

    try {
      const cutoff = new Date(Date.now() - sinceHours * 3600000);
      const since = cutoff.toISOString();
      const commentPath = `/repos/${this._repo}/issues/${issue}/comments?since=${since}&per_page=100`;
      const result = await this._request('GET', commentPath);

      if (!result.data || !Array.isArray(result.data)) return [];

      const packets = [];
      for (const comment of result.data) {
        const body = comment.body || '';
        if (!body.includes('```json')) continue;
        try {
          const start = body.indexOf('```json') + 7;
          const end = body.indexOf('```', start);
          const raw = body.substring(start, end).trim();
          const packet = JSON.parse(raw);
          if (packet.version && packet.weights) {
            packets.push(packet);
          }
        } catch {}
      }
      return packets;
    } catch {
      return [];
    }
  }
}


// ── Federation Client ──

class FederationClient {
  /**
   * @param {Object} options
   * @param {string} options.dataDir - Directory for federation state
   */
  constructor(options = {}) {
    this._dataDir = options.dataDir || path.join(process.cwd(), '.entroly');
    this._enabled = process.env.ENTROLY_FEDERATION === '1';
    this._clientId = getClientId(this._dataDir);
    this._contribDir = path.join(this._dataDir, 'federation_contributions');
    this._globalPath = path.join(this._dataDir, 'federation_global.json');

    try { fs.mkdirSync(this._contribDir, { recursive: true }); } catch {}
  }

  get enabled() { return this._enabled; }
  get clientId() { return this._clientId; }

  /**
   * Create and save a local contribution from current weights.
   */
  contribute(archetype, weights, sampleCount, confidence) {
    const packet = createContribution(
      this._clientId, archetype, weights, sampleCount, confidence,
    );
    if (!packet) return null;

    // Save locally
    const filename = `contrib_${Date.now()}_${this._clientId.slice(0, 8)}.json`;
    const filepath = path.join(this._contribDir, filename);
    try {
      fs.writeFileSync(filepath, JSON.stringify(packet));
    } catch {}

    return packet;
  }

  /**
   * Save a remote contribution locally for merge.
   */
  saveContribution(packet) {
    if (packet.client_id === this._clientId) return false; // Anti-echo
    const filename = `remote_${Date.now()}_${(packet.client_id || '').slice(0, 8)}.json`;
    const filepath = path.join(this._contribDir, filename);
    try {
      fs.writeFileSync(filepath, JSON.stringify(packet));
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Load all local contribution files.
   */
  loadContributions() {
    try {
      const files = fs.readdirSync(this._contribDir).filter(f => f.endsWith('.json'));
      return files.map(f => {
        try {
          return JSON.parse(fs.readFileSync(path.join(this._contribDir, f), 'utf-8'));
        } catch { return null; }
      }).filter(Boolean);
    } catch {
      return [];
    }
  }

  /**
   * Merge contributions via trimmed-mean aggregation.
   * @returns {{ weights: Object, contributors: number } | null}
   */
  mergeGlobal() {
    const contributions = this.loadContributions();
    if (contributions.length < 3) return null;

    // Group by archetype
    const byArchetype = {};
    for (const c of contributions) {
      const arch = c.archetype || 'unknown';
      if (!byArchetype[arch]) byArchetype[arch] = [];
      byArchetype[arch].push(c);
    }

    // Aggregate each archetype
    const result = {};
    for (const [arch, packets] of Object.entries(byArchetype)) {
      if (packets.length < 3) continue;

      const merged = {};
      for (const k of WEIGHT_KEYS) {
        const values = packets.map(p => p.weights[k]).filter(v => typeof v === 'number');
        if (values.length >= 3) {
          merged[k] = trimmedMean(values);
        }
      }

      if (Object.keys(merged).length === WEIGHT_KEYS.length) {
        result[arch] = {
          weights: merged,
          contributors: packets.length,
          confidence: packets.reduce((s, p) => s + (p.confidence || 0), 0) / packets.length,
          updated_at: Date.now() / 1000,
        };
      }
    }

    if (Object.keys(result).length > 0) {
      try {
        fs.writeFileSync(this._globalPath, JSON.stringify(result, null, 2));
      } catch {}
    }

    return result;
  }

  /**
   * Get the global consensus weights for an archetype.
   */
  getGlobalWeights(archetype) {
    try {
      const data = JSON.parse(fs.readFileSync(this._globalPath, 'utf-8'));
      return data[archetype] || null;
    } catch {
      return null;
    }
  }
}


module.exports = {
  GitHubTransport,
  FederationClient,
  createContribution,
  trimmedMean,
  getClientId,
  calibrateSigma,
  addNoise,
  WEIGHT_KEYS,
};
