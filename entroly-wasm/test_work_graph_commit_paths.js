'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');
const {
  RepositoryDiscoveryError,
  discoverRepositoryObservation,
} = require('./js/work_graph_repo');

function assert(condition, message) {
  if (!condition) throw new Error(message || 'assertion failed');
}

function git(repo, ...args) {
  return execFileSync('git', ['-C', repo, ...args], {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  }).trim();
}

function makeRepo(root) {
  const repo = path.join(root, 'repo');
  fs.mkdirSync(repo);
  git(repo, 'init', '-b', 'main');
  git(repo, 'config', 'user.email', 'test@example.com');
  git(repo, 'config', 'user.name', 'Test');
  fs.writeFileSync(path.join(repo, 'base.txt'), 'base\n');
  git(repo, 'add', 'base.txt');
  git(repo, 'commit', '-m', 'base');
  git(repo, 'checkout', '-b', 'feature/interrupted');
  return repo;
}

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'entroly-work-commit-paths-'));
try {
  const repo = makeRepo(root);
  fs.writeFileSync(path.join(repo, 'base.txt'), 'changed\n');
  fs.mkdirSync(path.join(repo, 'src'));
  fs.writeFileSync(path.join(repo, 'src', 'new.js'), 'module.exports = 1;\n');
  git(repo, 'add', 'base.txt', 'src/new.js');
  git(repo, 'commit', '-m', 'feature one');
  fs.writeFileSync(path.join(repo, 'docs.md'), 'docs\n');
  git(repo, 'add', 'docs.md');
  git(repo, 'commit', '-m', 'feature two');

  const observation = discoverRepositoryObservation(repo, {
    includeCheckpoint: false,
    observedAtMs: 1,
  });
  assert(observation.changes.length === 0, 'clean feature branch unexpectedly dirty');
  assert(observation.branch.ahead_by === 2, 'ahead count drift');
  assert(observation.commits.length === 2, 'commit count drift');
  const bySubject = new Map(observation.commits.map(commit => [commit.subject, commit]));
  assert(
    JSON.stringify(bySubject.get('feature one').changed_paths) === JSON.stringify(['base.txt', 'src/new.js']),
    'first commit changed paths are incomplete',
  );
  assert(
    JSON.stringify(bySubject.get('feature two').changed_paths) === JSON.stringify(['docs.md']),
    'second commit changed paths are incomplete',
  );

  const hugeRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'entroly-work-commit-paths-huge-'));
  try {
    const hugeRepo = makeRepo(hugeRoot);
    fs.mkdirSync(path.join(hugeRepo, 'bulk'));
    for (let index = 0; index < 513; index += 1) {
      fs.writeFileSync(path.join(hugeRepo, 'bulk', `f${String(index).padStart(3, '0')}.txt`), 'x\n');
    }
    git(hugeRepo, 'add', 'bulk');
    git(hugeRepo, 'commit', '-m', 'large commit');
    let rejected = false;
    try {
      discoverRepositoryObservation(hugeRepo, { includeCheckpoint: false, observedAtMs: 2 });
    } catch (error) {
      rejected = error instanceof RepositoryDiscoveryError && /commit-path observation/.test(error.message);
    }
    assert(rejected, 'oversized commit-path expansion returned partial history');
  } finally {
    fs.rmSync(hugeRoot, { recursive: true, force: true });
  }

  console.log('Work Graph npm committed-path recovery contract: PASS');
} finally {
  fs.rmSync(root, { recursive: true, force: true });
}
