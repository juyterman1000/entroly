/**
 * Workspace Change Listener — pure-JS port of entroly/change_listener.py
 * Detects file changes, marks beliefs stale, recompiles, verifies.
 */

'use strict';

const fs = require('fs');
const path = require('path');
const { SOURCE_EXTS } = require('./vault');

class WorkspaceChangeListener {
  constructor(vault, compiler, verifier, changePipe, projectDir) {
    this._vault = vault;
    this._compiler = compiler;
    this._verifier = verifier;
    this._changePipe = changePipe;
    this._projectDir = projectDir || process.cwd();
    this._lastScan = {};  // file -> mtime
    this._watcher = null;
    this._running = false;
  }

  scanOnce(force = false, maxFiles = 100) {
    const changes = { new: [], modified: [], deleted: [] };
    const currentFiles = {};
    const walk = (dir, depth = 0) => {
      if (depth > 8) return;
      let entries;
      try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
      for (const e of entries) {
        if (e.name.startsWith('.') || e.name === 'node_modules' || e.name === '__pycache__' || e.name === 'target') continue;
        const full = path.join(dir, e.name);
        if (e.isDirectory()) { walk(full, depth + 1); continue; }
        if (!SOURCE_EXTS.has(path.extname(e.name))) continue;
        try {
          const stat = fs.statSync(full);
          const mtime = stat.mtimeMs;
          const rel = path.relative(this._projectDir, full).replace(/\\/g, '/');
          currentFiles[rel] = mtime;
          if (!this._lastScan[rel]) changes.new.push(rel);
          else if (force || mtime > this._lastScan[rel]) changes.modified.push(rel);
        } catch {}
      }
    };
    walk(this._projectDir);

    // Detect deletions
    for (const rel of Object.keys(this._lastScan)) {
      if (!currentFiles[rel]) changes.deleted.push(rel);
    }
    this._lastScan = currentFiles;

    const allChanged = [...changes.new, ...changes.modified].slice(0, maxFiles);
    let beliefsStaled = 0, beliefsRecompiled = 0;

    if (allChanged.length > 0) {
      const staleResult = this._vault.markBeliefsStaleForFiles(allChanged);
      beliefsStaled = staleResult.updated_entities ? staleResult.updated_entities.length : 0;

      // Recompile changed files
      for (const rel of allChanged.slice(0, 20)) {
        try {
          const fp = path.join(this._projectDir, rel);
          if (fs.existsSync(fp)) {
            const dir = path.dirname(fp);
            const result = this._compiler.compileDirectory(dir, 1);
            beliefsRecompiled += result.beliefs_written;
          }
        } catch {}
      }
    }

    return {
      status: 'scanned',
      new_files: changes.new.length,
      modified_files: changes.modified.length,
      deleted_files: changes.deleted.length,
      beliefs_staled: beliefsStaled,
      beliefs_recompiled: beliefsRecompiled,
      total_tracked: Object.keys(currentFiles).length,
      engine: 'javascript',
    };
  }

  start(intervalS = 120, maxFiles = 100, forceInitial = false) {
    if (this._running) return { status: 'already_running', engine: 'javascript' };
    this._running = true;

    // Initial scan
    const initial = this.scanOnce(forceInitial, maxFiles);

    // Set up interval
    this._interval = setInterval(() => {
      try { this.scanOnce(false, maxFiles); } catch {}
    }, intervalS * 1000);

    // Unref so it doesn't keep the process alive
    if (this._interval.unref) this._interval.unref();

    return {
      status: 'started',
      interval_s: intervalS,
      initial_scan: initial,
      engine: 'javascript',
    };
  }

  stop() {
    if (this._interval) { clearInterval(this._interval); this._interval = null; }
    this._running = false;
    return { status: 'stopped', engine: 'javascript' };
  }
}

module.exports = { WorkspaceChangeListener };
