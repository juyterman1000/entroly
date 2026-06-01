'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { resolveProjectDirectory, resolveProjectOutput } = require('./js/server');
const { SkillEngine } = require('./js/skills');
const { VaultManager } = require('./js/vault');

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'entroly-security-'));
const project = path.join(root, 'project');
fs.mkdirSync(path.join(project, 'nested'), { recursive: true });

assert.strictEqual(resolveProjectDirectory(project, 'nested'), path.join(project, 'nested'));
assert.strictEqual(resolveProjectDirectory(project, '..'), null);
assert.strictEqual(resolveProjectOutput(project, 'training.jsonl'), path.join(project, 'training.jsonl'));
assert.strictEqual(resolveProjectOutput(project, '../training.jsonl'), null);

const vault = new VaultManager(path.join(root, 'vault'));
vault.ensureStructure();
const skills = new SkillEngine(vault);
assert.strictEqual(skills.benchmarkSkill('../../outside').status, 'invalid_skill_id');
assert.strictEqual(skills.promoteOrPrune('../outside').status, 'invalid_skill_id');

console.log('Node security hardening tests passed');
