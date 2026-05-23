// Entroly Checkpoint Manager — JS port of checkpoint.py
// State persistence with gzip-compressed JSON.

const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const crypto = require('crypto');

class CheckpointManager {
  constructor(checkpointDir, autoInterval = 5) {
    this.checkpointDir = checkpointDir;
    this.autoInterval = autoInterval;
    this._callsSinceCheckpoint = 0;
    this._totalCheckpoints = 0;
    fs.mkdirSync(checkpointDir, { recursive: true });
  }

  shouldAutoCheckpoint() {
    this._callsSinceCheckpoint++;
    return this._callsSinceCheckpoint >= this.autoInterval;
  }

  save(data) {
    this._callsSinceCheckpoint = 0;
    this._totalCheckpoints++;
    const id = crypto.randomBytes(8).toString('hex');
    const checkpoint = {
      checkpoint_id: id,
      timestamp: new Date().toISOString(),
      ...data,
    };
    const json = JSON.stringify(checkpoint);
    const compressed = zlib.gzipSync(json);
    const filePath = path.join(this.checkpointDir, `checkpoint_${id}.json.gz`);
    fs.writeFileSync(filePath, compressed);

    // Keep only last 5 checkpoints
    this._pruneOld(5);
    return filePath;
  }

  loadLatest() {
    try {
      const files = fs.readdirSync(this.checkpointDir)
        .filter(f => f.startsWith('checkpoint_') && f.endsWith('.json.gz'))
        .sort()
        .reverse();
      if (!files.length) return null;
      const filePath = path.join(this.checkpointDir, files[0]);
      const compressed = fs.readFileSync(filePath);
      const json = zlib.gunzipSync(compressed).toString();
      return JSON.parse(json);
    } catch { return null; }
  }

  _pruneOld(keep) {
    try {
      const files = fs.readdirSync(this.checkpointDir)
        .filter(f => f.startsWith('checkpoint_') && f.endsWith('.json.gz'))
        .sort();
      while (files.length > keep) {
        const old = files.shift();
        try { fs.unlinkSync(path.join(this.checkpointDir, old)); } catch {}
      }
    } catch {}
  }

  stats() {
    return {
      checkpoint_dir: this.checkpointDir,
      total_checkpoints: this._totalCheckpoints,
      calls_since_last: this._callsSinceCheckpoint,
    };
  }
}

/**
 * Persist engine index to a gzip-compressed JSON file.
 * @param {WasmEntrolyEngine} engine
 * @param {string} indexPath - Path to index.json.gz
 */
function persistIndex(engine, indexPath) {
  try {
    const state = engine.export_state();
    const json = JSON.stringify(state);
    const compressed = zlib.gzipSync(json);
    fs.writeFileSync(indexPath, compressed);
    return true;
  } catch { return false; }
}

/**
 * Load engine index from a gzip-compressed JSON file.
 * @param {WasmEntrolyEngine} engine
 * @param {string} indexPath
 */
function loadIndex(engine, indexPath) {
  try {
    if (!fs.existsSync(indexPath)) return false;
    const compressed = fs.readFileSync(indexPath);
    const json = zlib.gunzipSync(compressed).toString();
    const state = JSON.parse(json);
    engine.import_state(JSON.stringify(state));
    return true;
  } catch { return false; }
}

module.exports = { CheckpointManager, persistIndex, loadIndex };
