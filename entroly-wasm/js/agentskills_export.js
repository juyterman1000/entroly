/**
 * agentskills.io export — JS port.
 * Exports promoted vault skills to the portable agentskills.io v0.1 spec
 * so any compatible runtime can consume them.
 *
 * CLI:  node js/agentskills_export.js [out_dir]
 */

'use strict';

const fs = require('fs');
const path = require('path');

const SPEC_VERSION = '0.1';

function parseFrontmatter(text) {
  if (!text.startsWith('---')) return { meta: {}, body: text };
  const end = text.indexOf('\n---', 3);
  if (end < 0) return { meta: {}, body: text };
  const header = text.slice(3, end).trim();
  const body = text.slice(end + 4).replace(/^\n+/, '');
  const meta = {};
  for (const line of header.split('\n')) {
    const i = line.indexOf(':');
    if (i > 0) meta[line.slice(0, i).trim()] = line.slice(i + 1).trim();
  }
  return { meta, body };
}

function loadSkill(dir) {
  const skillMd = path.join(dir, 'SKILL.md');
  const toolPy = path.join(dir, 'tool.py');
  if (!fs.existsSync(skillMd) || !fs.existsSync(toolPy)) return null;

  const { meta, body } = parseFrontmatter(fs.readFileSync(skillMd, 'utf8'));
  if ((meta.status || '') !== 'promoted') return null;

  let metrics = {};
  const mp = path.join(dir, 'metrics.json');
  if (fs.existsSync(mp)) {
    try { metrics = JSON.parse(fs.readFileSync(mp, 'utf8')); } catch (_) {}
  }
  return {
    meta,
    procedure: body,
    toolCode: fs.readFileSync(toolPy, 'utf8'),
    metrics,
  };
}

function exportPromoted({
  vaultPath = '.entroly/vault',
  outDir = './dist/agentskills',
} = {}) {
  const skillsDir = path.join(vaultPath, 'evolution', 'skills');
  if (!fs.existsSync(skillsDir)) {
    return { status: 'no_vault', skillsDir };
  }
  fs.mkdirSync(outDir, { recursive: true });

  const exported = [];
  const skipped = [];

  for (const name of fs.readdirSync(skillsDir).sort()) {
    const dir = path.join(skillsDir, name);
    if (!fs.statSync(dir).isDirectory()) continue;

    const loaded = loadSkill(dir);
    if (!loaded) { skipped.push(name); continue; }

    const target = path.join(outDir, name);
    fs.rmSync(target, { recursive: true, force: true });
    fs.mkdirSync(target, { recursive: true });

    const skillJson = {
      spec_version: SPEC_VERSION,
      id: loaded.meta.skill_id || name,
      name: loaded.meta.name || name,
      entity: loaded.meta.entity || '',
      description: `Skill for handling ${loaded.meta.entity || name} queries`,
      status: loaded.meta.status || 'promoted',
      created_at: loaded.meta.created_at || '',
      metrics: {
        fitness_score: loaded.metrics.fitness_score || 0,
        runs: loaded.metrics.runs || 0,
        successes: loaded.metrics.successes || 0,
        failures: loaded.metrics.failures || 0,
      },
      entrypoint: {
        runtime: 'python',
        module: 'tool',
        match: 'matches',
        execute: 'execute',
      },
      origin: {
        runtime: 'entroly',
        synthesis: 'structural',
        token_cost: 0.0,
      },
    };

    fs.writeFileSync(path.join(target, 'skill.json'), JSON.stringify(skillJson, null, 2));
    fs.writeFileSync(path.join(target, 'procedure.md'), loaded.procedure);
    fs.writeFileSync(path.join(target, 'tool.py'), loaded.toolCode);
    fs.writeFileSync(path.join(target, 'tests.json'), '[]');

    exported.push(skillJson.id);
  }

  fs.writeFileSync(
    path.join(outDir, 'manifest.json'),
    JSON.stringify({
      spec_version: SPEC_VERSION,
      exported_at: new Date().toISOString(),
      source: 'entroly',
      skills: exported,
    }, null, 2),
  );

  return { status: 'ok', outDir, exported, skipped };
}

module.exports = { exportPromoted, SPEC_VERSION };

if (require.main === module) {
  const out = process.argv[2] || './dist/agentskills';
  const result = exportPromoted({ outDir: out });
  console.log(JSON.stringify(result, null, 2));
}
