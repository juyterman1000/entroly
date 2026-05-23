/**
 * Pure-JS VaultManager — port of entroly/vault.py
 * Manages the persistent Knowledge Vault (Obsidian-compatible markdown).
 *
 * Directory contract:
 *   vault/beliefs/        vault/verification/    vault/actions/
 *   vault/evolution/skills/<id>/  vault/media/
 */

'use strict';

const fs = require('fs');
const path = require('path');
const { randomUUID } = require('crypto');

const VAULT_DIRS = ['beliefs', 'verification', 'actions', 'evolution', 'media'];
const SOURCE_EXTS = new Set(['.py', '.rs', '.ts', '.js', '.tsx', '.jsx', '.go', '.java', '.c', '.cpp', '.h', '.hpp', '.cs', '.rb', '.swift', '.kt']);

function safeFilename(s) {
  let safe = s.trim().toLowerCase().replace(/[^\w\-.]/g, '_').replace(/_+/g, '_').replace(/^_|_$/g, '');
  return (safe || 'untitled').slice(0, 80);
}

function nowISO() { return new Date().toISOString(); }
function timestamp() { return new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d+Z/, ''); }

function parseFrontmatter(content) {
  if (!content.startsWith('---')) return null;
  const end = content.indexOf('---', 3);
  if (end < 0) return null;
  const fm = {};
  for (const line of content.slice(3, end).trim().split('\n')) {
    if (line.trim().startsWith('-')) continue;
    const idx = line.indexOf(':');
    if (idx < 0) continue;
    const key = line.slice(0, idx).trim();
    const val = line.slice(idx + 1).trim();
    if (key && val) fm[key] = val;
  }
  return Object.keys(fm).length ? fm : null;
}

function extractSources(content) {
  if (!content.startsWith('---')) return [];
  const end = content.indexOf('---', 3);
  if (end < 0) return [];
  const lines = content.slice(3, end).trim().split('\n');
  const sources = [];
  let inSources = false;
  for (const line of lines) {
    const t = line.trim();
    if (t.startsWith('sources:')) { inSources = true; continue; }
    if (inSources) {
      if (t.startsWith('- ')) { sources.push(t.slice(2).trim()); continue; }
      if (t && !t.startsWith('-')) break;
    }
  }
  return sources;
}

function extractBody(content) {
  if (!content.startsWith('---')) return content;
  const end = content.indexOf('---', 3);
  if (end < 0) return content;
  return content.slice(end + 3).trim();
}

function beliefToMarkdown(b) {
  const srcs = (b.sources && b.sources.length) ? b.sources.map(s => `  - ${s}`).join('\n') : '  - unknown';
  const derived = (b.derived_from && b.derived_from.length) ? b.derived_from.map(d => `  - ${d}`).join('\n') : '  - system';
  return `---\nclaim_id: ${b.claim_id}\nentity: ${b.entity}\nstatus: ${b.status}\nconfidence: ${b.confidence}\nsources:\n${srcs}\nlast_checked: ${b.last_checked}\nderived_from:\n${derived}\n---\n\n# ${b.title || b.entity}\n\n${b.body}\n`;
}

class VaultManager {
  constructor(basePath) {
    this._base = basePath || path.join(process.cwd(), '.entroly', 'vault');
    this._initialized = false;
  }

  ensureStructure() {
    if (this._initialized) return { status: 'already_initialized', path: this._base };
    const created = [];
    for (const d of VAULT_DIRS) {
      const p = path.join(this._base, d);
      if (!fs.existsSync(p)) { fs.mkdirSync(p, { recursive: true }); created.push(d); }
    }
    const skillsDir = path.join(this._base, 'evolution', 'skills');
    if (!fs.existsSync(skillsDir)) { fs.mkdirSync(skillsDir, { recursive: true }); created.push('evolution/skills'); }
    const reg = path.join(this._base, 'evolution', 'registry.md');
    if (!fs.existsSync(reg)) {
      fs.writeFileSync(reg, '# Skill Registry\n\nIndex of all dynamically generated skills.\n\n| Skill ID | Status | Created | Description |\n|---|---|---|---|\n', 'utf-8');
      created.push('evolution/registry.md');
    }
    this._initialized = true;
    return { status: 'initialized', path: this._base, created };
  }

  writeBelief(artifact) {
    this.ensureStructure();
    const safe = safeFilename(artifact.entity || artifact.claim_id);
    const fp = path.join(this._base, 'beliefs', `${safe}.md`);
    fs.writeFileSync(fp, beliefToMarkdown(artifact), 'utf-8');
    return { status: 'written', directory: 'beliefs', path: fp, claim_id: artifact.claim_id, entity: artifact.entity };
  }

  readBelief(entity) {
    this.ensureStructure();
    const beliefsDir = path.join(this._base, 'beliefs');
    const safe = safeFilename(entity);
    let fp = path.join(beliefsDir, `${safe}.md`);
    if (!fs.existsSync(fp)) {
      const files = this._globBeliefs();
      const match = files.find(f => path.basename(f, '.md').toLowerCase().includes(entity.toLowerCase()));
      if (!match) return null;
      fp = match;
    }
    const content = fs.readFileSync(fp, 'utf-8');
    return { path: fp, frontmatter: parseFrontmatter(content) || {}, body: extractBody(content) };
  }

  listBeliefs() {
    this.ensureStructure();
    const results = [];
    for (const fp of this._globBeliefs()) {
      try {
        const content = fs.readFileSync(fp, 'utf-8');
        const fm = parseFrontmatter(content);
        const beliefsDir = path.join(this._base, 'beliefs');
        results.push({
          file: path.relative(beliefsDir, fp),
          entity: fm ? (fm.entity || path.basename(fp, '.md')) : path.basename(fp, '.md'),
          status: fm ? (fm.status || 'unknown') : 'unknown',
          confidence: fm ? parseFloat(fm.confidence || '0') : 0,
          last_checked: fm ? (fm.last_checked || '') : '',
        });
      } catch {}
    }
    return results;
  }

  coverageIndex() {
    const beliefs = this.listBeliefs();
    const total = beliefs.length;
    const verified = beliefs.filter(b => b.status === 'verified').length;
    const stale = beliefs.filter(b => b.status === 'stale').length;
    const avg = total ? beliefs.reduce((a, b) => a + b.confidence, 0) / total : 0;
    return { total_beliefs: total, verified, stale, inferred: total - verified - stale, average_confidence: Math.round(avg * 1000) / 1000, entities: beliefs.map(b => b.entity) };
  }

  writeVerification(artifact) {
    this.ensureStructure();
    const ts = timestamp();
    const safe = safeFilename(artifact.title || artifact.challenges || 'check');
    const fp = path.join(this._base, 'verification', `${ts}_${safe}.md`);
    const md = `---\nchallenges: ${artifact.challenges}\nresult: ${artifact.result}\nconfidence_delta: ${artifact.confidence_delta >= 0 ? '+' : ''}${artifact.confidence_delta.toFixed(2)}\nchecked_at: ${artifact.checked_at || nowISO()}\nmethod: ${artifact.method || ''}\n---\n\n# ${artifact.title || ''}\n\n${artifact.body || ''}\n`;
    fs.writeFileSync(fp, md, 'utf-8');
    if (artifact.result === 'confirmed' && artifact.challenges) {
      this._updateBeliefConfidence(artifact.challenges, artifact.confidence_delta);
    }
    return { status: 'written', directory: 'verification', path: fp, challenges: artifact.challenges, result: artifact.result };
  }

  writeAction(title, content, actionType = 'report') {
    this.ensureStructure();
    const ts = timestamp();
    const safe = safeFilename(title);
    const fp = path.join(this._base, 'actions', `${ts}_${safe}.md`);
    fs.writeFileSync(fp, `---\ntype: ${actionType}\ntimestamp: ${ts}\n---\n\n# ${title}\n\n${content}\n`, 'utf-8');
    return { status: 'written', directory: 'actions', path: fp, type: actionType };
  }

  markBeliefsStaleForFiles(changedFiles) {
    this.ensureStructure();
    const changedPaths = new Set(changedFiles.map(f => f.replace(/\\/g, '/').toLowerCase()));
    const changedStems = new Set(changedFiles.map(f => path.basename(f).replace(/\.[^.]+$/, '').toLowerCase()));
    const updatedEntities = [], updatedFiles = [], alreadyStale = [];
    for (const fp of this._globBeliefs()) {
      try {
        let content = fs.readFileSync(fp, 'utf-8');
        const fm = parseFrontmatter(content);
        if (!fm) continue;
        const entity = fm.entity || path.basename(fp, '.md');
        const sources = extractSources(content);
        let matched = sources.some(src => {
          const sp = src.split(':')[0].replace(/\\/g, '/').toLowerCase();
          return changedPaths.has(sp) || changedStems.has(path.basename(sp).replace(/\.[^.]+$/, ''));
        });
        if (!matched) matched = [...changedStems].some(stem => entity.toLowerCase().includes(stem));
        if (!matched) continue;
        if (fm.status === 'stale') { alreadyStale.push(entity); continue; }
        content = content.replace(/^status:\s+.+$/m, 'status: stale');
        fs.writeFileSync(fp, content, 'utf-8');
        updatedEntities.push(entity);
        updatedFiles.push(fp);
      } catch {}
    }
    return { status: 'updated', changed_files: changedFiles.length, updated_entities: updatedEntities, updated_files: updatedFiles, already_stale: alreadyStale };
  }

  _updateBeliefConfidence(claimId, delta) {
    for (const fp of this._globBeliefs()) {
      try {
        let content = fs.readFileSync(fp, 'utf-8');
        const fm = parseFrontmatter(content);
        if (!fm || fm.claim_id !== claimId) continue;
        const oldConf = parseFloat(fm.confidence || '0.5');
        const newConf = Math.max(0, Math.min(1, oldConf + delta));
        content = content.replace(`confidence: ${fm.confidence}`, `confidence: ${newConf}`);
        if (delta > 0 && content.includes('status: inferred')) content = content.replace('status: inferred', 'status: verified');
        content = content.replace(/last_checked: .+/, `last_checked: ${nowISO()}`);
        fs.writeFileSync(fp, content, 'utf-8');
        break;
      } catch {}
    }
  }

  _globBeliefs() {
    const dir = path.join(this._base, 'beliefs');
    if (!fs.existsSync(dir)) return [];
    return this._rglob(dir, '.md');
  }

  _rglob(dir, ext) {
    const results = [];
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) results.push(...this._rglob(full, ext));
      else if (entry.name.endsWith(ext)) results.push(full);
    }
    return results.sort();
  }
}

module.exports = {
  VaultManager, safeFilename, parseFrontmatter, extractSources, extractBody,
  beliefToMarkdown, nowISO, timestamp, SOURCE_EXTS,
};
