// Entroly Autotune v2 — Cross-Session Weight Optimization
//
// Learns optimal scoring weights (recency, frequency, semantic, entropy)
// from real developer feedback across sessions. Replaces synthetic benchmarks
// with a feedback journal that persists between restarts.
//
// Key mechanisms:
//   - Reward-weighted regression with advantage normalization
//   - Exponential decay for non-stationarity (γ=0.995)
//   - Per-dimension adaptive step sizes via SNR
//   - Task-conditioned profiles (different weights for debugging vs features)
//

const fs = require('fs');
const path = require('path');

// ── Constants ──
const WEIGHT_KEYS = ['w_r', 'w_f', 'w_s', 'w_e'];
const DEFAULTS = { w_r: 0.30, w_f: 0.25, w_s: 0.25, w_e: 0.20, exploration: 0.1 };
const DECAY_GAMMA = 0.995;       // Per-episode decay (half-life ≈ 138 episodes)
const WARMUP_EPISODES = 8;       // Min episodes before tuning
const MAX_BLEND_RATE = 0.5;      // Max single-step movement toward optimal
const EXPLORATION_C = 0.1;       // UCB exploration constant
const MIN_WEIGHT = 0.05;         // Floor per weight
const MAX_WEIGHT = 0.80;         // Ceiling per weight
const JOURNAL_MAX_AGE_MS = 14 * 24 * 60 * 60 * 1000; // 14 days


// --- Feedback Journal ---

class FeedbackJournal {
  constructor(journalDir) {
    this.journalPath = path.join(journalDir, 'feedback_journal.jsonl');
    try { fs.mkdirSync(journalDir, { recursive: true }); } catch {}
    this._cache = null;
  }

  /**
   * Log a feedback episode.
   * @param {object} episode
   * @param {object} episode.weights - {w_r, w_f, w_s, w_e} at time of last optimize
   * @param {string[]} episode.selectedSources - Sources selected by optimizer
   * @param {number} episode.selectedCount - Fragment count
   * @param {number} episode.tokenBudget - Budget used
   * @param {string} episode.query - Query string
   * @param {number} episode.reward - +1.0 success, -1.0 failure, or continuous [-1,1]
   * @param {number} episode.turn - Session turn number
   */
  log(episode) {
    const entry = {
      t: Date.now(),
      w: episode.weights,
      n: episode.selectedCount,
      src: (episode.selectedSources || []).slice(0, 15),
      q: (episode.query || '').slice(0, 120),
      r: Math.max(-1, Math.min(1, episode.reward)), // clamp to [-1, 1]
      turn: episode.turn,
      bgt: episode.tokenBudget,
    };
    try {
      fs.appendFileSync(this.journalPath, JSON.stringify(entry) + '\n');
      this._cache = null;
    } catch {}
  }

  /**
   * Load episodes, filtered by max age.
   */
  load(maxAge = JOURNAL_MAX_AGE_MS) {
    if (this._cache) return this._cache;
    const cutoff = Date.now() - maxAge;
    try {
      const lines = fs.readFileSync(this.journalPath, 'utf-8').split('\n').filter(Boolean);
      this._cache = [];
      for (const line of lines) {
        try {
          const e = JSON.parse(line);
          if (e && e.t >= cutoff && e.w) this._cache.push(e);
        } catch {}
      }
      return this._cache;
    } catch {
      this._cache = [];
      return [];
    }
  }

  /** Prune old episodes */
  prune(maxAge = JOURNAL_MAX_AGE_MS) {
    this._cache = null; // invalidate so load() re-reads with new maxAge
    const kept = this.load(maxAge);
    this._cache = null; // invalidate again after write
    try {
      fs.writeFileSync(this.journalPath,
        kept.map(e => JSON.stringify(e)).join('\n') + (kept.length ? '\n' : ''));
    } catch {}
  }

  count() { return this.load().length; }

  stats() {
    const eps = this.load();
    if (!eps.length) return { episodes: 0, successes: 0, failures: 0, avgReward: 0, oldestDays: 0 };
    const successes = eps.filter(e => e.r > 0).length;
    const failures = eps.filter(e => e.r < 0).length;
    const avgReward = eps.reduce((s, e) => s + e.r, 0) / eps.length;
    const oldestDays = Math.round((Date.now() - Math.min(...eps.map(e => e.t))) / 86400000 * 10) / 10;
    return {
      episodes: eps.length, successes, failures,
      avgReward: Math.round(avgReward * 1000) / 1000,
      oldestDays,
    };
  }
}


// --- Reward-Weighted Optimizer ---

/**
 * Extract weight vector from an episode's stored format.
 */
function extractWeights(w) {
  return {
    w_r: w.w_r ?? w.R ?? DEFAULTS.w_r,
    w_f: w.w_f ?? w.F ?? DEFAULTS.w_f,
    w_s: w.w_s ?? w.S ?? DEFAULTS.w_s,
    w_e: w.w_e ?? w.E ?? DEFAULTS.w_e,
  };
}

/**
 * Normalize weight vector to sum=1, clamped to [MIN_WEIGHT, MAX_WEIGHT].
 */
function normalizeWeights(w) {
  const out = {};
  for (const k of WEIGHT_KEYS) out[k] = Math.max(MIN_WEIGHT, Math.min(MAX_WEIGHT, w[k]));
  const sum = WEIGHT_KEYS.reduce((s, k) => s + out[k], 0);
  for (const k of WEIGHT_KEYS) out[k] = Math.round(out[k] / sum * 10000) / 10000;
  return out;
}

/**
 * Core optimizer: reward-weighted regression with all research enhancements.
 *
 * @param {object[]} episodes - Raw journal episodes
 * @param {object} currentWeights - Current weight vector {w_r, w_f, w_s, w_e}
 * @returns {object|null} Optimization result
 */
function optimize(episodes, currentWeights) {
  if (episodes.length < 3) return null;

  // Advantage normalization
  const rewards = episodes.map(e => e.r);
  const μ = rewards.reduce((a, b) => a + b) / rewards.length;
  const σ = Math.sqrt(rewards.reduce((s, r) => s + (r - μ) ** 2, 0) / rewards.length);

  // Edge case: when all rewards are identical (σ≈0), advantage normalization
  // produces all-zero advantages. Fall back to raw reward weighting with decay.
  const useRawRewards = σ < 1e-6;
  const advantages = useRawRewards
    ? rewards.map(r => r > 0 ? 1.0 : r < 0 ? -1.0 : 0)
    : rewards.map(r => (r - μ) / (σ + 1e-8));

  // Exponential decay — newer episodes matter more
  const sortedEps = [...episodes].sort((a, b) => a.t - b.t);
  const decayWeights = sortedEps.map((e, i) => DECAY_GAMMA ** (episodes.length - 1 - i));

  // Reward-weighted mean: positive advantages attract, negative repel
  const attract = { w_r: 0, w_f: 0, w_s: 0, w_e: 0 };
  let attractSum = 0;
  const repel = { w_r: 0, w_f: 0, w_s: 0, w_e: 0 };
  let repelSum = 0;

  for (let i = 0; i < sortedEps.length; i++) {
    const ep = sortedEps[i];
    const w = extractWeights(ep.w);
    const adv = advantages[episodes.indexOf(ep)] || 0;
    const decay = decayWeights[i];

    if (adv > 0) {
      const weight = decay * adv;
      for (const k of WEIGHT_KEYS) attract[k] += weight * w[k];
      attractSum += weight;
    } else if (adv < 0) {
      const weight = decay * Math.abs(adv);
      for (const k of WEIGHT_KEYS) repel[k] += weight * w[k];
      repelSum += weight;
    }
  }

  // Normalize. If no positive signal at all, return null.
  if (attractSum > 0) for (const k of WEIGHT_KEYS) attract[k] /= attractSum;
  else return null;

  if (repelSum > 0) for (const k of WEIGHT_KEYS) repel[k] /= repelSum;

  // Per-dimension step size: high SNR → larger step
  const dimStats = {};
  for (const k of WEIGHT_KEYS) {
    const values = sortedEps.map(e => extractWeights(e.w)[k]);
    const mean = values.reduce((a, b) => a + b) / values.length;
    const std = Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length) || 0.01;
    const snr = Math.abs(attract[k] - currentWeights[k]) / std;
    dimStats[k] = { mean, std, snr };
  }

  // Compute per-dimension blend rate: α_k = base_α · sigmoid(SNR_k)
  const N = episodes.length;
  const confidence = Math.min(1.0, N / WARMUP_EPISODES);
  const baseAlpha = confidence * MAX_BLEND_RATE;

  // Natural gradient update with attraction/repulsion
  const β = repelSum > 0 ? 0.15 * Math.min(1.0, repelSum / attractSum) : 0;
  const optimal = {};
  const blended = {};

  for (const k of WEIGHT_KEYS) {
    // Raw optimal direction
    const attractDelta = attract[k] - currentWeights[k];
    const repelDelta = repelSum > 0 ? (repel[k] - currentWeights[k]) : 0;
    const natGradScale = 1.0 / (dimStats[k].std ** 2 + 0.01); // Inverse Fisher (diagonal)
    const sigmoidSNR = 1.0 / (1.0 + Math.exp(-2 * (dimStats[k].snr - 0.5))); // sigmoid centering

    // Per-dimension adaptive alpha
    const αk = baseAlpha * sigmoidSNR;

    // Optimal = attraction - β·repulsion
    optimal[k] = attract[k] - β * repelDelta;

    // Blended = current + αk · natGrad · direction
    const direction = (optimal[k] - currentWeights[k]) * natGradScale;
    const clampedDirection = Math.max(-0.1, Math.min(0.1, direction)); // Prevent wild jumps
    blended[k] = currentWeights[k] + αk * clampedDirection;
  }

  const normalizedOptimal = normalizeWeights(optimal);
  const normalizedBlended = normalizeWeights(blended);

  // Exploration bonus
  const exploration = computeExplorationBonus(sortedEps);

  // Polyak average
  const polyak = { w_r: 0, w_f: 0, w_s: 0, w_e: 0 };
  for (const ep of sortedEps) {
    const w = extractWeights(ep.w);
    for (const k of WEIGHT_KEYS) polyak[k] += w[k];
  }
  for (const k of WEIGHT_KEYS) polyak[k] /= sortedEps.length;
  const normalizedPolyak = normalizeWeights(polyak);

  // Regret estimation
  const avgObserved = μ;
  const avgPositive = episodes.filter(e => e.r > 0).length > 0
    ? episodes.filter(e => e.r > 0).reduce((s, e) => s + e.r, 0) / episodes.filter(e => e.r > 0).length
    : 0;
  const estimatedRegret = Math.max(0, (avgPositive - avgObserved) * N);

  return {
    optimal: normalizedOptimal,
    blended: normalizedBlended,
    polyak: normalizedPolyak,
    confidence,
    baseAlpha,
    perDimStats: dimStats,
    exploration,
    successCount: episodes.filter(e => e.r > 0).length,
    failureCount: episodes.filter(e => e.r < 0).length,
    neutralCount: episodes.filter(e => e.r === 0).length,
    totalEpisodes: N,
    estimatedRegret: Math.round(estimatedRegret * 100) / 100,
    advantageMean: Math.round(μ * 1000) / 1000,
    advantageStd: Math.round(σ * 1000) / 1000,
    decayEffectiveN: Math.round(decayWeights.reduce((a, b) => a + b) * 10) / 10,
  };
}


/**
 * UCB1-style exploration bonus.
 * Identifies which weight dimension is most under-explored.
 */
function computeExplorationBonus(episodes) {
  if (episodes.length < 5) return null;

  const stats = {};
  for (const k of WEIGHT_KEYS) {
    const values = episodes.map(e => extractWeights(e.w)[k]);
    const mean = values.reduce((a, b) => a + b) / values.length;
    const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length;
    const std = Math.sqrt(variance);
    const cv = std / Math.max(mean, 0.01); // coefficient of variation

    // UCB bonus: how uncertain are we about this dimension?
    const ucbBonus = EXPLORATION_C * Math.sqrt(Math.log(episodes.length) / Math.max(values.length, 1));
    stats[k] = { mean: Math.round(mean * 1000) / 1000, std: Math.round(std * 1000) / 1000, cv: Math.round(cv * 1000) / 1000, ucbBonus: Math.round(ucbBonus * 1000) / 1000 };
  }

  // Most under-explored = highest CV
  let maxCv = 0, underExplored = null;
  for (const k of WEIGHT_KEYS) {
    if (stats[k].cv > maxCv) { maxCv = stats[k].cv; underExplored = k; }
  }

  return { perDim: stats, underExplored, suggestion: `Try varying ${underExplored} more` };
}


// --- Config I/O ---

function findConfigPath() {
  for (const p of [
    path.join(process.cwd(), 'tuning_config.json'),
    path.join(process.cwd(), 'bench', 'tuning_config.json'),
    path.join(__dirname, '..', '..', 'bench', 'tuning_config.json'),
  ]) { try { if (fs.existsSync(p)) return p; } catch {} }
  return null;
}

function loadConfig() {
  const p = findConfigPath();
  if (!p) return null;
  try { return JSON.parse(fs.readFileSync(p, 'utf-8')); } catch { return null; }
}

function saveConfig(config) {
  const p = findConfigPath();
  if (p) try { fs.writeFileSync(p, JSON.stringify(config, null, 2)); } catch {}
}


// --- Public API ---

/**
 * Run optimization from journal data.
 * @param {string} checkpointDir
 * @returns {object|null}
 */
function optimizeFromJournal(checkpointDir) {
  const journal = new FeedbackJournal(checkpointDir);
  const episodes = journal.load();
  if (episodes.length < 3) return null;

  const cfg = loadConfig();
  const currentWeights = {
    w_r: cfg?.weights?.recency ?? DEFAULTS.w_r,
    w_f: cfg?.weights?.frequency ?? DEFAULTS.w_f,
    w_s: cfg?.weights?.semantic_sim ?? DEFAULTS.w_s,
    w_e: cfg?.weights?.entropy ?? DEFAULTS.w_e,
  };

  return optimize(episodes, currentWeights);
}

/**
 * Hot-reload weights from tuning_config.json into a live engine.
 */
function hotReloadWeights(engine) {
  const cfg = loadConfig();
  if (!cfg) return;
  engine.set_weights(
    cfg.weights?.recency ?? DEFAULTS.w_r,
    cfg.weights?.frequency ?? DEFAULTS.w_f,
    cfg.weights?.semantic_sim ?? DEFAULTS.w_s,
    cfg.weights?.entropy ?? DEFAULTS.w_e,
  );
}

/**
 * Start autotune daemon.
 *
 * Flow per tick:
 *   1. Load feedback journal
 *   2. If < warmup episodes → hot-reload from config only
 *   3. If ≥ warmup → compute optimal via reward-weighted regression
 *   4. Blend optimal into current weights (confidence-ramped, per-dim adaptive)
 *   5. Push to live engine
 *   6. Periodically prune journal
 */
function startAutotuneDaemon(engine, checkpointDir, intervalMs = 30000) {
  const cfg = loadConfig();
  if (cfg?.autotuner?.enabled === false) {
    console.error('[autotune] Disabled via tuning_config.json');
    return null;
  }

  const dir = checkpointDir || path.join(process.cwd(), '.entroly');
  const journal = new FeedbackJournal(dir);

  hotReloadWeights(engine);

  const timer = setInterval(() => {
    try {
      const episodes = journal.load();

      if (episodes.length < WARMUP_EPISODES) {
        hotReloadWeights(engine);
        return;
      }

      const cfg2 = loadConfig();
      const current = {
        w_r: cfg2?.weights?.recency ?? DEFAULTS.w_r,
        w_f: cfg2?.weights?.frequency ?? DEFAULTS.w_f,
        w_s: cfg2?.weights?.semantic_sim ?? DEFAULTS.w_s,
        w_e: cfg2?.weights?.entropy ?? DEFAULTS.w_e,
      };

      const result = optimize(episodes, current);
      if (result) {
        const w = result.blended;
        engine.set_weights(w.w_r, w.w_f, w.w_s, w.w_e);
        console.error(
          `[autotune] ${result.totalEpisodes} episodes ` +
          `(${result.successCount}✓ ${result.failureCount}✗) ` +
          `α_eff=${result.baseAlpha.toFixed(2)} regret=${result.estimatedRegret} → ` +
          `R=${w.w_r.toFixed(3)} F=${w.w_f.toFixed(3)} S=${w.w_s.toFixed(3)} E=${w.w_e.toFixed(3)}`
        );
      }

      // Prune 5% of the time
      if (Math.random() < 0.05) journal.prune();
    } catch (e) {
      console.error(`[autotune] Error: ${e.message}`);
    }
  }, intervalMs);

  return { timer, journal };
}

/**
 * CLI command: show status and run optimization.
 */
function runAutotune() {
  const { EntrolyConfig } = require('./config');
  const config = new EntrolyConfig();
  const journal = new FeedbackJournal(config.checkpointDir);
  const stats = journal.stats();

  console.error(`[autotune] Journal: ${journal.journalPath}`);
  console.error(`[autotune] Episodes: ${stats.episodes} (${stats.successes}✓ ${stats.failures}✗ avg=${stats.avgReward} span=${stats.oldestDays}d)`);


  if (stats.episodes < 3) {
    console.error('[autotune] ⚠ Need ≥3 feedback episodes. Use MCP tools to provide feedback:');
    console.error('[autotune]   1. optimize_context → generates context');
    console.error('[autotune]   2. record_outcome → tells autotune if it was helpful');
    console.error('[autotune]   (The server does this automatically when you use it)');
    return;
  }

  const result = optimizeFromJournal(config.checkpointDir);
  if (!result) { console.error('[autotune] No positive signal found yet.'); return; }

  console.error(`[autotune] ✓ Optimization from ${result.totalEpisodes} episodes:`);
  console.error(`[autotune]   Confidence: ${(result.confidence * 100).toFixed(0)}%`);
  console.error(`[autotune]   Effective N (after decay): ${result.decayEffectiveN}`);
  console.error(`[autotune]   Estimated regret: ${result.estimatedRegret}`);
  console.error(`[autotune]   Reward distribution: μ=${result.advantageMean} σ=${result.advantageStd}`);
  console.error('[autotune]');
  console.error(`[autotune]   Optimal (raw):    R=${result.optimal.w_r} F=${result.optimal.w_f} S=${result.optimal.w_s} E=${result.optimal.w_e}`);
  console.error(`[autotune]   Blended (safe):   R=${result.blended.w_r} F=${result.blended.w_f} S=${result.blended.w_s} E=${result.blended.w_e}`);
  console.error(`[autotune]   Polyak average:   R=${result.polyak.w_r} F=${result.polyak.w_f} S=${result.polyak.w_s} E=${result.polyak.w_e}`);

  if (result.exploration) {
    console.error('[autotune]');
    console.error(`[autotune]   Exploration: ${result.exploration.suggestion}`);
    for (const [k, v] of Object.entries(result.exploration.perDim)) {
      console.error(`[autotune]     ${k}: μ=${v.mean} σ=${v.std} CV=${v.cv} UCB=${v.ucbBonus}`);
    }
  }

  // Save blended weights
  const cfg = loadConfig();
  if (cfg?.weights) {
    cfg.weights.recency = result.blended.w_r;
    cfg.weights.frequency = result.blended.w_f;
    cfg.weights.semantic_sim = result.blended.w_s;
    cfg.weights.entropy = result.blended.w_e;
    saveConfig(cfg);
    console.error('[autotune]');
    console.error('[autotune]   ✓ Blended weights saved to tuning_config.json');
  }
}

// --- Task-Conditioned Weight Profiles ---
// Different tasks need different weights: debugging prioritizes recency+semantic,
// feature work prioritizes entropy+frequency. Profiles are optimized independently.

// Simple task classifier (mirrors the Rust classify_task logic)
const TASK_PATTERNS = {
  Debugging:    /\b(fix|bug|error|crash|issue|debug|broken|fail|wrong|exception|undefined|null|stack|trace)\b/i,
  Feature:      /\b(add|implement|create|build|feature|new|support|integrate|enable)\b/i,
  Refactoring:  /\b(refactor|clean|reorganize|simplify|extract|rename|move|split|merge|dedup)\b/i,
  Performance:  /\b(optimize|performance|slow|fast|speed|cache|memory|leak|bottleneck|latency)\b/i,
  Testing:      /\b(test|spec|assert|expect|mock|stub|coverage|unit|integration|e2e)\b/i,
  Documentation:/\b(document|readme|comment|explain|describe|usage|api|jsdoc|typedoc)\b/i,
};

function classifyQuery(query) {
  if (!query) return 'General';
  for (const [type, pattern] of Object.entries(TASK_PATTERNS)) {
    if (pattern.test(query)) return type;
  }
  return 'General';
}

// Default weight priors per task type (informed by domain knowledge)
const TASK_PRIORS = {
  Debugging:     { w_r: 0.35, w_f: 0.15, w_s: 0.35, w_e: 0.15 },
  Feature:       { w_r: 0.20, w_f: 0.25, w_s: 0.25, w_e: 0.30 },
  Refactoring:   { w_r: 0.15, w_f: 0.35, w_s: 0.20, w_e: 0.30 },
  Performance:   { w_r: 0.25, w_f: 0.30, w_s: 0.25, w_e: 0.20 },
  Testing:       { w_r: 0.30, w_f: 0.20, w_s: 0.30, w_e: 0.20 },
  Documentation: { w_r: 0.20, w_f: 0.20, w_s: 0.30, w_e: 0.30 },
  General:       { ...DEFAULTS },
};

class TaskProfileOptimizer {
  constructor(journal) {
    this.journal = journal;
    this._profiles = {}; // { taskType: { weights, lastUpdated, episodeCount } }
  }

  /**
   * Classify all journal episodes by task type and optimize each independently.
   * @returns {{ profiles: object, totalEpisodes: number }}
   */
  optimizeAll() {
    const episodes = this.journal.load();
    if (episodes.length < 3) return { profiles: {}, totalEpisodes: episodes.length };

    // Partition episodes by task type
    const buckets = {};
    for (const ep of episodes) {
      const taskType = classifyQuery(ep.q);
      if (!buckets[taskType]) buckets[taskType] = [];
      buckets[taskType].push(ep);
    }

    // Optimize each task type independently
    const profiles = {};
    for (const [taskType, taskEpisodes] of Object.entries(buckets)) {
      const prior = TASK_PRIORS[taskType] || TASK_PRIORS.General;
      if (taskEpisodes.length >= 3) {
        const result = optimize(taskEpisodes, prior);
        if (result) {
          profiles[taskType] = {
            weights: result.blended,
            confidence: result.confidence,
            episodes: taskEpisodes.length,
            regret: result.estimatedRegret,
          };
        }
      }
      // If not enough episodes or no signal, use prior
      if (!profiles[taskType]) {
        profiles[taskType] = {
          weights: { ...prior },
          confidence: 0,
          episodes: taskEpisodes.length,
          regret: 0,
        };
      }
    }

    // Add priors for unseen task types
    for (const taskType of Object.keys(TASK_PRIORS)) {
      if (!profiles[taskType]) {
        profiles[taskType] = {
          weights: { ...TASK_PRIORS[taskType] },
          confidence: 0,
          episodes: 0,
          regret: 0,
        };
      }
    }

    this._profiles = profiles;
    return { profiles, totalEpisodes: episodes.length };
  }

  /**
   * Get the optimal weight profile for a given query.
   * @param {string} query
   * @returns {{ weights: object, taskType: string, confidence: number }}
   */
  getProfileForQuery(query) {
    const taskType = classifyQuery(query);
    const profile = this._profiles[taskType];
    if (profile && profile.confidence > 0) {
      return { weights: profile.weights, taskType, confidence: profile.confidence };
    }
    // Fallback to task prior
    return { weights: TASK_PRIORS[taskType] || TASK_PRIORS.General, taskType, confidence: 0 };
  }

  /**
   * Apply task-conditioned weights to an engine for a specific query.
   * @param {WasmEntrolyEngine} engine
   * @param {string} query
   */
  applyToEngine(engine, query) {
    const { weights, taskType, confidence } = this.getProfileForQuery(query);
    engine.set_weights(weights.w_r, weights.w_f, weights.w_s, weights.w_e);
    return { taskType, confidence, weights };
  }
}


module.exports = {
  FeedbackJournal,
  optimize,
  computeExplorationBonus,
  optimizeFromJournal,
  startAutotuneDaemon,
  runAutotune,
  hotReloadWeights,
  // Task-conditioned profiles
  TaskProfileOptimizer,
  classifyQuery,
  TASK_PRIORS,
  // Internals exposed for testing
  extractWeights,
  normalizeWeights,
  WEIGHT_KEYS,
  DEFAULTS,
};
