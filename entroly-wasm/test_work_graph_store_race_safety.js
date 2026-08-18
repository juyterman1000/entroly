'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const {
  WorkGraphStateError,
  WorkGraphStore,
} = require('./js/work_graph_store');

function assert(condition, message) {
  if (!condition) throw new Error(message || 'assertion failed');
}

function age(filePath, milliseconds = 5000) {
  const old = new Date(Date.now() - milliseconds);
  fs.utimesSync(filePath, old, old);
}

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'entroly-work-store-race-'));
try {
  const store = new WorkGraphStore('repo:race-safety', {
    root: path.join(root, 'state'),
    lockTimeoutMs: 10,
    staleLockMs: 1000,
  });

  const liveToken = `${os.hostname()}:${process.pid}:live-owner`;
  fs.writeFileSync(store.lockPath, `${liveToken}\n0\n`, { mode: 0o600 });
  age(store.lockPath);
  assert(store.breakStaleLock() === false, 'old live-owner lock was reclaimed');
  assert(fs.existsSync(store.lockPath), 'live-owner lock disappeared');
  assert(store.readLockToken() === liveToken, 'live-owner token changed');
  fs.unlinkSync(store.lockPath);

  const foreignToken = `definitely-other-host:${process.pid}:foreign-owner`;
  fs.writeFileSync(store.lockPath, `${foreignToken}\n0\n`, { mode: 0o600 });
  age(store.lockPath);
  assert(store.breakStaleLock() === false, 'old foreign-host lock was reclaimed without proof');
  assert(fs.existsSync(store.lockPath), 'foreign-host lock disappeared');
  assert(store.readLockToken() === foreignToken, 'foreign-host token changed');
  fs.unlinkSync(store.lockPath);

  fs.writeFileSync(store.lockPath, 'dead-owner\n0\n', { mode: 0o600 });
  age(store.lockPath);
  assert(store.breakStaleLock() === true, 'dead old lock was not reclaimed');
  assert(!fs.existsSync(store.lockPath), 'dead old lock still exists');

  fs.writeFileSync(store.lockPath, 'x'.repeat(5000), { mode: 0o600 });
  let oversizedRejected = false;
  try { store.readLockToken(); }
  catch (error) { oversizedRejected = error instanceof WorkGraphStateError; }
  assert(oversizedRejected, 'oversized lock metadata was accepted');
  fs.unlinkSync(store.lockPath);

  if (process.platform !== 'win32') {
    const outsideLock = path.join(root, 'outside-lock');
    fs.writeFileSync(outsideLock, 'outside\n');
    fs.symlinkSync(outsideLock, store.lockPath);
    let symlinkLockRejected = false;
    try { store.readLockToken(); }
    catch (error) { symlinkLockRejected = error instanceof WorkGraphStateError; }
    assert(symlinkLockRejected, 'symlink lock was followed');
    assert(fs.readFileSync(outsideLock, 'utf8') === 'outside\n', 'outside lock target changed');
    fs.unlinkSync(store.lockPath);

    const outsideState = path.join(root, 'outside-state');
    fs.writeFileSync(outsideState, 'not-a-work-graph\n');
    fs.symlinkSync(outsideState, store.statePath);
    let symlinkStateRejected = false;
    try { store.loadUnlocked(); }
    catch (error) { symlinkStateRejected = error instanceof WorkGraphStateError; }
    assert(symlinkStateRejected, 'symlink state was followed');
    assert(fs.readFileSync(outsideState, 'utf8') === 'not-a-work-graph\n', 'outside state target changed');
  }

  console.log('Work Graph npm store race safety contract: PASS');
} finally {
  fs.rmSync(root, { recursive: true, force: true });
}
