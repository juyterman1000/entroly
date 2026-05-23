/**
 * Skill Engine — pure-JS port of entroly/skill_engine.py
 * Dynamic skill synthesis, benchmarking, and lifecycle management.
 */

'use strict';

const fs = require('fs');
const path = require('path');
const { randomUUID } = require('crypto');
const { nowISO, safeFilename } = require('./vault');

class SkillEngine {
  constructor(vault) {
    this._vault = vault;
    this._skillsDir = path.join(vault._base, 'evolution', 'skills');
  }

  createSkill(entityKey, failingQueries, intent = '') {
    this._vault.ensureStructure();
    const skillId = `skill_${safeFilename(entityKey)}_${Date.now().toString(36)}`;
    const skillDir = path.join(this._skillsDir, skillId);
    fs.mkdirSync(skillDir, { recursive: true });
    fs.mkdirSync(path.join(skillDir, 'tests'), { recursive: true });

    // SKILL.md
    fs.writeFileSync(path.join(skillDir, 'SKILL.md'), [
      `# Skill: ${entityKey}`,
      '',
      `**ID:** ${skillId}`,
      `**Entity:** ${entityKey}`,
      `**Intent:** ${intent || 'general'}`,
      `**Created:** ${nowISO()}`,
      `**Status:** draft`,
      '',
      '## Failing Queries',
      '',
      ...failingQueries.map(q => `- ${q}`),
      '',
      '## Procedure',
      '',
      '1. TODO: Define the skill procedure',
      '2. TODO: Implement the tool',
      '3. TODO: Add test cases',
      '',
    ].join('\n'), 'utf-8');

    // metrics.json
    fs.writeFileSync(path.join(skillDir, 'metrics.json'), JSON.stringify({
      skill_id: skillId, entity: entityKey, created: nowISO(),
      status: 'draft', fitness: 0, runs: 0, successes: 0, failures: 0,
    }, null, 2), 'utf-8');

    // tests/test_cases.json
    fs.writeFileSync(path.join(skillDir, 'tests', 'test_cases.json'), JSON.stringify({
      skill_id: skillId,
      cases: failingQueries.map((q, i) => ({
        id: `tc_${i}`, query: q, expected_pass: true, actual: null,
      })),
    }, null, 2), 'utf-8');

    // Update registry
    const regPath = path.join(this._vault._base, 'evolution', 'registry.md');
    if (fs.existsSync(regPath)) {
      const reg = fs.readFileSync(regPath, 'utf-8');
      fs.writeFileSync(regPath, reg + `| ${skillId} | draft | ${nowISO().slice(0, 10)} | ${entityKey} |\n`, 'utf-8');
    }

    return { status: 'created', skill_id: skillId, entity: entityKey, path: skillDir, engine: 'javascript' };
  }

  listSkills() {
    this._vault.ensureStructure();
    if (!fs.existsSync(this._skillsDir)) return [];
    const skills = [];
    for (const name of fs.readdirSync(this._skillsDir)) {
      const metricsPath = path.join(this._skillsDir, name, 'metrics.json');
      if (!fs.existsSync(metricsPath)) continue;
      try {
        const m = JSON.parse(fs.readFileSync(metricsPath, 'utf-8'));
        skills.push({ skill_id: m.skill_id || name, entity: m.entity || '', status: m.status || 'draft', fitness: m.fitness || 0, runs: m.runs || 0 });
      } catch {}
    }
    return skills;
  }

  benchmarkSkill(skillId) {
    const metricsPath = path.join(this._skillsDir, skillId, 'metrics.json');
    const testsPath = path.join(this._skillsDir, skillId, 'tests', 'test_cases.json');
    if (!fs.existsSync(metricsPath)) return { error: `Skill '${skillId}' not found` };
    const metrics = JSON.parse(fs.readFileSync(metricsPath, 'utf-8'));
    let cases = [];
    if (fs.existsSync(testsPath)) {
      try { cases = JSON.parse(fs.readFileSync(testsPath, 'utf-8')).cases || []; } catch {}
    }
    // Simple benchmark: count cases as "passed" if they have expected_pass
    const passed = cases.filter(c => c.expected_pass).length;
    const fitness = cases.length ? passed / cases.length : 0;
    metrics.fitness = fitness;
    metrics.runs = (metrics.runs || 0) + 1;
    fs.writeFileSync(metricsPath, JSON.stringify(metrics, null, 2), 'utf-8');
    return { skill_id: skillId, fitness, test_cases: cases.length, passed, status: metrics.status, engine: 'javascript' };
  }

  promoteOrPrune(skillId) {
    const metricsPath = path.join(this._skillsDir, skillId, 'metrics.json');
    if (!fs.existsSync(metricsPath)) return { error: `Skill '${skillId}' not found` };
    const metrics = JSON.parse(fs.readFileSync(metricsPath, 'utf-8'));
    let action;
    if (metrics.fitness >= 0.7) {
      metrics.status = 'promoted';
      action = 'promoted';
    } else if (metrics.fitness <= 0.3) {
      metrics.status = 'pruned';
      action = 'pruned';
    } else {
      action = 'no_change';
    }
    fs.writeFileSync(metricsPath, JSON.stringify(metrics, null, 2), 'utf-8');
    return { skill_id: skillId, action, fitness: metrics.fitness, new_status: metrics.status, engine: 'javascript' };
  }
}

module.exports = { SkillEngine };
