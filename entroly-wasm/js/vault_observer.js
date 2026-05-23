/**
 * VaultObserver — derive self-evolution stats from the on-disk vault.
 *
 * The vault IS the state machine. Whichever runtime (pip or npm) is
 * actually running the daemon, the effects land here:
 *
 *   evolution/skills/<id>/SKILL.md      (status: draft|promoted|pruned)
 *   evolution/skills/<id>/metrics.json  (fitness, runs, successes)
 *   evolution/gap_*.md                  (pending gaps)
 *   value_tracker.json (in data dir)    (budget + savings)
 *
 * This observer reads them. It exposes the same `.stats()` shape the
 * gateways expect, so you can do:
 *
 *   const obs = new VaultObserver('.entroly/vault');
 *   new TelegramGateway({ token, chatId }).attach(obs).start();
 *
 * No daemon required. No duplicated orchestration logic. Works
 * whether the Python or Node runtime owns the vault.
 */

'use strict';

const fs = require('fs');
const path = require('path');
const os = require('os');

function parseFrontmatter(text) {
  if (!text.startsWith('---')) return {};
  const end = text.indexOf('\n---', 3);
  if (end < 0) return {};
  const meta = {};
  for (const line of text.slice(3, end).split('\n')) {
    const i = line.indexOf(':');
    if (i > 0) meta[line.slice(0, i).trim()] = line.slice(i + 1).trim();
  }
  return meta;
}

class VaultObserver {
  /**
   * @param {string} vaultPath - Path to the vault root (default: .entroly/vault)
   * @param {object} opts
   * @param {string} opts.dataDir - Where value_tracker.json lives (default: ~/.entroly)
   */
  constructor(vaultPath = '.entroly/vault', opts = {}) {
    this._vault = vaultPath;
    this._dataDir = opts.dataDir || path.join(os.homedir(), '.entroly');
  }

  // ── Derived state ──────────────────────────────────────────

  _listSkills() {
    const dir = path.join(this._vault, 'evolution', 'skills');
    if (!fs.existsSync(dir)) return [];
    return fs.readdirSync(dir)
      .map(name => {
        const sdir = path.join(dir, name);
        const skillMd = path.join(sdir, 'SKILL.md');
        if (!fs.existsSync(skillMd)) return null;
        const meta = parseFrontmatter(fs.readFileSync(skillMd, 'utf8'));
        let metrics = {};
        const mp = path.join(sdir, 'metrics.json');
        if (fs.existsSync(mp)) {
          try { metrics = JSON.parse(fs.readFileSync(mp, 'utf8')); } catch (_) {}
        }
        return { id: name, status: meta.status || 'draft', entity: meta.entity || '', metrics };
      })
      .filter(Boolean);
  }

  _listGaps() {
    const dir = path.join(this._vault, 'evolution');
    if (!fs.existsSync(dir)) return [];
    return fs.readdirSync(dir).filter(f => f.startsWith('gap_') && f.endsWith('.md'));
  }

  _readValueTracker() {
    const p = path.join(this._dataDir, 'value_tracker.json');
    if (!fs.existsSync(p)) return null;
    try { return JSON.parse(fs.readFileSync(p, 'utf8')); }
    catch (_) { return null; }
  }

  _readRegistry() {
    const p = path.join(this._vault, 'evolution', 'registry.md');
    return fs.existsSync(p) ? fs.readFileSync(p, 'utf8') : '';
  }

  // ── Public API (matches EvolutionDaemon.stats() shape) ─────

  stats() {
    const skills = this._listSkills();
    const promoted = skills.filter(s => s.status === 'promoted').length;
    const pruned = skills.filter(s => s.status === 'pruned').length;
    const structuralSuccesses = skills.filter(s =>
      (s.metrics && s.metrics.fitness_score >= 0.5)
    ).length;

    const vt = this._readValueTracker();
    const lifetimeSaved = vt ? (vt.lifetime?.cost_saved_usd || 0) : 0;
    const evolutionSpent = vt ? (vt.lifetime?.evolution_spent_usd || 0) : 0;
    const earned = lifetimeSaved * 0.05;
    const available = Math.max(0, earned - evolutionSpent);

    return {
      running: true,
      skills_promoted: promoted,
      skills_pruned: pruned,
      structural_successes: structuralSuccesses,
      dream_cycles: 0,  // not observable from vault alone
      gaps_pending: this._listGaps().length,
      budget: {
        available_usd: +available.toFixed(6),
        total_earned_usd: +earned.toFixed(6),
        total_spent_usd: +evolutionSpent.toFixed(6),
        can_evolve: available > 0.001,
      },
    };
  }

  /** Raw accessors for gateway command handlers (/skills, /gaps, /status). */
  skills() { return this._listSkills(); }
  gaps() { return this._listGaps(); }
  registry() { return this._readRegistry(); }
  budget() { return this.stats().budget; }
}

module.exports = { VaultObserver };
