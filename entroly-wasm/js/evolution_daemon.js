/**
 * Evolution Daemon — Zero-Token Autonomous Self-Improvement (JavaScript)
 * ======================================================================
 *
 * Background daemon that orchestrates:
 *   1. Dreaming Loop — idle-time weight optimization via self-play
 *   2. Federation — contribute + merge global weights via GitHub
 *
 * Mirrors the Python EvolutionDaemon but adapted for Node.js event loop.
 *
 * @module evolution_daemon
 */

'use strict';

const { FeedbackJournal, optimize, WEIGHT_KEYS, DEFAULTS } = require('./autotune');

// ── Dreaming Loop ──

const IDLE_THRESHOLD_MS = 60000;    // 60s idle before dreaming
const DREAM_PERTURBATION = 0.08;    // Weight perturbation magnitude
const DREAM_MAX_TRIALS = 5;         // Trials per dream cycle

class DreamingLoop {
  /**
   * @param {FeedbackJournal} journal
   * @param {Object} [options]
   * @param {number} [options.idleThresholdMs]
   */
  constructor(journal, options = {}) {
    this._journal = journal;
    this._idleThreshold = options.idleThresholdMs || IDLE_THRESHOLD_MS;
    this._lastActivity = Date.now();
    this._stats = { cycles: 0, improvements: 0, totalTrials: 0 };
  }

  /** Record user activity — resets idle timer. */
  recordActivity() {
    this._lastActivity = Date.now();
  }

  /** Check if system has been idle long enough to dream. */
  shouldDream() {
    return (Date.now() - this._lastActivity) >= this._idleThreshold;
  }

  /**
   * Run one dream cycle: perturb weights, evaluate, keep improvements.
   *
   * Uses counterfactual self-play:
   *   1. Load best weights from journal history
   *   2. Generate random perturbations
   *   3. Score each variant against journal episodes
   *   4. If any beats current → adopt it
   *
   * @param {Object} currentWeights - Current {w_r, w_f, w_s, w_e}
   * @returns {{ status: string, improvements: number, bestScore: number }}
   */
  runDreamCycle(currentWeights) {
    const episodes = this._journal.load();
    if (episodes.length < 5) {
      return { status: 'insufficient_data', improvements: 0, bestScore: 0 };
    }

    this._stats.cycles++;
    let bestWeights = { ...currentWeights };
    let bestScore = this._scoreWeights(currentWeights, episodes);
    let improvements = 0;

    for (let trial = 0; trial < DREAM_MAX_TRIALS; trial++) {
      this._stats.totalTrials++;

      // Generate perturbation (Gaussian)
      const candidate = {};
      for (const k of WEIGHT_KEYS) {
        const noise = (Math.random() + Math.random() + Math.random() - 1.5) * DREAM_PERTURBATION;
        candidate[k] = Math.max(0.05, Math.min(0.80, (bestWeights[k] || 0.25) + noise));
      }
      // Normalize to sum=1
      const sum = WEIGHT_KEYS.reduce((s, k) => s + candidate[k], 0);
      for (const k of WEIGHT_KEYS) candidate[k] = Math.round(candidate[k] / sum * 10000) / 10000;

      const score = this._scoreWeights(candidate, episodes);
      if (score > bestScore) {
        bestScore = score;
        bestWeights = { ...candidate };
        improvements++;
      }
    }

    if (improvements > 0) {
      this._stats.improvements += improvements;
    }

    return {
      status: 'completed',
      improvements,
      bestScore: Math.round(bestScore * 1000) / 1000,
      bestWeights,
      trials: DREAM_MAX_TRIALS,
    };
  }

  /**
   * Score a weight configuration against historical episodes.
   * Simulates reward: higher reward episodes → higher agreement with weights.
   * @private
   */
  _scoreWeights(weights, episodes) {
    let totalScore = 0;
    let count = 0;

    for (const ep of episodes) {
      if (!ep.w || ep.r === undefined) continue;

      // Cosine similarity between candidate weights and episode's weights
      const epW = ep.w;
      let dot = 0, normA = 0, normB = 0;
      for (const k of WEIGHT_KEYS) {
        const a = weights[k] || 0.25;
        const b = epW[k] || epW[k.replace('w_', 'w_')] || 0.25;
        dot += a * b;
        normA += a * a;
        normB += b * b;
      }
      const similarity = dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-8);

      // Reward-weighted: positive episodes attract, negative repel
      totalScore += ep.r * similarity;
      count++;
    }

    return count > 0 ? totalScore / count : 0;
  }

  get stats() { return { ...this._stats }; }
}


// ── Evolution Daemon ──

class EvolutionDaemon {
  /**
   * @param {Object} options
   * @param {string} options.checkpointDir - Directory for state
   * @param {Object} [options.engine] - WASM engine instance
   * @param {number} [options.pollIntervalMs=30000]
   */
  constructor(options = {}) {
    this._checkpointDir = options.checkpointDir || '.entroly';
    this._engine = options.engine || null;
    this._pollInterval = options.pollIntervalMs || 30000;

    // Journal
    this._journal = new FeedbackJournal(this._checkpointDir);

    // Dreaming
    this._dreaming = new DreamingLoop(this._journal);

    // Federation (lazy loaded)
    this._federation = null;
    this._transport = null;
    this._federationCycleCounter = 0;

    // State
    this._timer = null;
    this._stats = {
      dreamCycles: 0,
      federationContributions: 0,
      federationMerges: 0,
    };
  }

  /**
   * Initialize federation (call after construction).
   */
  async initFederation() {
    try {
      const { FederationClient, GitHubTransport } = require('./federation');
      this._federation = new FederationClient({ dataDir: this._checkpointDir });

      if (this._federation.enabled) {
        this._transport = new GitHubTransport();

        // Sync on startup
        try {
          const packets = await this._transport.pull(48);
          let saved = 0;
          for (const packet of packets) {
            if (this._federation.saveContribution(packet)) saved++;
          }
          if (saved > 0) {
            this._federation.mergeGlobal();
            this._stats.federationMerges++;
          }
        } catch {}
      }
    } catch {}
  }

  /** Record user activity — resets dreaming idle timer. */
  recordActivity() {
    this._dreaming.recordActivity();
  }

  /**
   * Start the daemon (non-blocking, uses setInterval).
   */
  start() {
    if (this._timer) return;

    this._timer = setInterval(() => {
      try {
        this.runOnce();
      } catch (e) {
        console.error(`[evolution] Error: ${e.message}`);
      }
    }, this._pollInterval);

    // Make timer non-blocking (don't prevent process exit)
    if (this._timer.unref) this._timer.unref();

    console.error('[evolution] Daemon started');
  }

  /** Stop the daemon. */
  stop() {
    if (this._timer) {
      clearInterval(this._timer);
      this._timer = null;
    }
  }

  /**
   * Execute one daemon cycle.
   */
  runOnce() {
    const result = { dream: null, federation: null };

    // ── Phase 1: Dreaming ──
    if (this._dreaming.shouldDream()) {
      const currentWeights = this._getCurrentWeights();
      const dreamResult = this._dreaming.runDreamCycle(currentWeights);
      result.dream = dreamResult;

      if (dreamResult.status === 'completed') {
        this._stats.dreamCycles++;

        // Apply improved weights to engine
        if (dreamResult.improvements > 0 && dreamResult.bestWeights && this._engine) {
          const w = dreamResult.bestWeights;
          try {
            this._engine.set_weights(w.w_r, w.w_f, w.w_s, w.w_e);
          } catch {}
        }

        // ── Phase 2: Federation ──
        if (dreamResult.improvements > 0 && this._federation && this._federation.enabled) {
          this._federationCycleCounter++;
          this._handleFederation(dreamResult);
        }
      }
    }

    return result;
  }

  /** @private */
  _getCurrentWeights() {
    // Try to read from engine, fall back to defaults
    if (this._engine && typeof this._engine.get_weights === 'function') {
      try {
        const w = this._engine.get_weights();
        return {
          w_r: w.recency ?? DEFAULTS.w_r,
          w_f: w.frequency ?? DEFAULTS.w_f,
          w_s: w.semantic_sim ?? DEFAULTS.w_s,
          w_e: w.entropy ?? DEFAULTS.w_e,
        };
      } catch {}
    }
    return { ...DEFAULTS };
  }

  /** @private */
  _handleFederation(dreamResult) {
    if (!this._federation || !this._transport) return;

    try {
      // Contribute improved weights
      const w = dreamResult.bestWeights;
      const packet = this._federation.contribute(
        'js_project', // archetype
        w,
        this._journal.count(),
        Math.min(1.0, this._journal.count() / 20),
      );

      if (packet && this._transport.canWrite) {
        // Async push — fire and forget
        this._transport.push(packet).then(ok => {
          if (ok) this._stats.federationContributions++;
        }).catch(() => {});
      }

      // Merge global every 10 cycles
      if (this._federationCycleCounter % 10 === 0) {
        this._transport.pull(48).then(packets => {
          let saved = 0;
          for (const p of packets) {
            if (this._federation.saveContribution(p)) saved++;
          }
          if (saved > 0) {
            this._federation.mergeGlobal();
            this._stats.federationMerges++;

            // Apply merged weights
            const global = this._federation.getGlobalWeights('js_project');
            if (global && this._engine) {
              const gw = global.weights;
              try {
                this._engine.set_weights(gw.w_r, gw.w_f, gw.w_s, gw.w_e);
              } catch {}
            }
          }
        }).catch(() => {});
      }
    } catch {}
  }

  get stats() { return { ...this._stats, dreaming: this._dreaming.stats }; }
}


module.exports = { EvolutionDaemon, DreamingLoop };
