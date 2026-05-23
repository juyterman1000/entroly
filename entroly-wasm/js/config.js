// Entroly Configuration — JS port of config.py
// Central configuration for the context optimization engine.

const path = require('path');
const os = require('os');
const fs = require('fs');
const crypto = require('crypto');

function projectCheckpointDir() {
  const explicit = process.env.ENTROLY_DIR;
  if (explicit) return explicit;

  const cwd = process.cwd();
  const projectHash = crypto.createHash('sha256').update(cwd).digest('hex').slice(0, 12);
  const defaultDir = path.join(os.homedir(), '.entroly', 'checkpoints', projectHash);

  try {
    fs.mkdirSync(defaultDir, { recursive: true });
    const probe = path.join(defaultDir, '.entroly_write_probe');
    fs.writeFileSync(probe, 'ok');
    fs.unlinkSync(probe);
    return defaultDir;
  } catch {
    const fallback = path.join(os.tmpdir(), 'entroly', 'checkpoints', projectHash);
    fs.mkdirSync(fallback, { recursive: true });
    return fallback;
  }
}

class EntrolyConfig {
  constructor(opts = {}) {
    this.defaultTokenBudget = opts.defaultTokenBudget ?? 128_000;
    this.maxFragments = opts.maxFragments ?? 10_000;
    this.weightRecency = opts.weightRecency ?? 0.30;
    this.weightFrequency = opts.weightFrequency ?? 0.25;
    this.weightSemanticSim = opts.weightSemanticSim ?? 0.25;
    this.weightEntropy = opts.weightEntropy ?? 0.20;
    this.decayHalfLifeTurns = opts.decayHalfLifeTurns ?? 15;
    this.minRelevanceThreshold = opts.minRelevanceThreshold ?? 0.05;
    this.dedupSimilarityThreshold = opts.dedupSimilarityThreshold ?? 0.92;
    this.prefetchDepth = opts.prefetchDepth ?? 2;
    this.maxPrefetchFragments = opts.maxPrefetchFragments ?? 10;
    this.checkpointDir = opts.checkpointDir ?? projectCheckpointDir();
    this.autoCheckpointInterval = opts.autoCheckpointInterval ?? 5;
    this.serverName = opts.serverName ?? 'entroly';
    this.serverVersion = opts.serverVersion ?? (() => {
      try { return require('../package.json').version; }
      catch { return '0.0.0'; }
    })();
  }
}

module.exports = { EntrolyConfig, projectCheckpointDir };
