// Entroly Configuration — JS port of config.py
// Central configuration for the context optimization engine.

const path = require('path');
const os = require('os');
const fs = require('fs');
const crypto = require('crypto');

// This account's own directory under the shared temporary directory, used when
// HOME cannot be written. Mirrors entroly/private_temp.py: the fixed name
// <tmp>/entroly belonged to whichever account created it first, so the next
// account to fall back could not write it and a world-writable one decided
// where everyone else's checkpoints went. Windows already gives each account
// its own temporary directory, so there the name stays "entroly".
function accountTempPath() {
  const name = typeof process.getuid === 'function' ? `entroly-${process.getuid()}` : 'entroly';
  return path.join(os.tmpdir(), name);
}

function notPrivate(dir) {
  const stat = fs.lstatSync(dir);
  if (stat.isSymbolicLink()) return 'it is a symbolic link';
  if (!stat.isDirectory()) return 'it is not a directory';
  if (typeof process.getuid !== 'function') return null;
  if (stat.uid !== process.getuid()) return `it belongs to uid ${stat.uid}, not this account`;
  if (stat.mode & 0o077) {
    return `other accounts can open it, mode ${(stat.mode & 0o7777).toString(8).padStart(4, '0')}`;
  }
  return null;
}

function accountTempDir() {
  const dir = accountTempPath();
  if (typeof process.getuid !== 'function') {
    fs.mkdirSync(dir, { recursive: true });
    return dir;
  }
  fs.mkdirSync(path.dirname(dir), { recursive: true });
  try {
    fs.mkdirSync(dir, { mode: 0o700 });
    fs.chmodSync(dir, 0o700);
  } catch (error) {
    if (error.code !== 'EEXIST') throw error;
  }
  const reason = notPrivate(dir);
  if (reason) {
    const error = new Error(`${dir}: ${reason}`);
    error.code = 'EACCES';
    error.path = dir;
    error.reason = reason;
    throw error;
  }
  return dir;
}

function existingAccountTempDir() {
  const dir = accountTempPath();
  try {
    return notPrivate(dir) ? null : dir;
  } catch {
    return null;
  }
}

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
  } catch (homeError) {
    try {
      const fallback = path.join(accountTempDir(), 'checkpoints', projectHash);
      fs.mkdirSync(fallback, { recursive: true, mode: 0o700 });
      return fallback;
    } catch (fallbackError) {
      const why = (error) => error.reason || error.code || error.message;
      throw new Error(
        `Entroly has nowhere private to keep this project's data: it cannot write ${defaultDir} ` +
        `(${why(homeError)}) and will not use ${accountTempPath()} (${why(fallbackError)}). ` +
        'Set ENTROLY_DIR to a writable directory only you can read.'
      );
    }
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

module.exports = {
  EntrolyConfig,
  projectCheckpointDir,
  accountTempPath,
  accountTempDir,
  existingAccountTempDir,
};
