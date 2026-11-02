'use strict';

// When HOME cannot be written, checkpoints fall back to the shared temporary
// directory. The fallback was the fixed name <tmp>/entroly: the account that
// created it first owned it, so the next account to fall back could not
// write it, and one created world-writable decided where every other
// account's checkpoints went. It is now this account's own <tmp>/entroly-<uid>,
// created 0700 and refused unless it is a real directory only we can open.
// Mirrors tests/test_temp_fallback_is_private.py.
//
// Each scenario resolves in a fresh process: Bun reads HOME once at startup.

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

function assert(condition, message) {
  if (!condition) throw new Error(message || 'assertion failed');
}

if (process.argv[2] === '--resolve') {
  const { projectCheckpointDir } = require('./js/config');
  let outcome;
  try { outcome = { dir: projectCheckpointDir() }; }
  catch (error) { outcome = { error: error.message }; }
  process.stdout.write(JSON.stringify(outcome));
  process.exit(0);
}

if (typeof process.getuid !== 'function') {
  console.log('Private temp fallback contract: SKIP (Windows temp directories are per account)');
  process.exit(0);
}

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'entroly-private-temp-'));

function scenario(name, prepare) {
  const dir = path.join(root, name);
  const home = path.join(dir, 'home-is-a-file');
  const tmp = path.join(dir, 'tmp');
  fs.mkdirSync(tmp, { recursive: true });
  fs.writeFileSync(home, 'not a directory');
  const mine = path.join(tmp, `entroly-${process.getuid()}`);
  prepare(tmp, mine);
  const env = { ...process.env, HOME: home, USERPROFILE: home, TMPDIR: tmp };
  delete env.ENTROLY_DIR;
  const child = spawnSync(process.execPath, [__filename, '--resolve'], { env, encoding: 'utf8' });
  assert(child.status === 0, `${name}: resolver crashed: ${child.stderr}`);
  return { tmp, mine, ...JSON.parse(child.stdout) };
}

function refused(label, outcome) {
  assert(outcome.error && outcome.error.includes('ENTROLY_DIR'),
    `${label}: expected an error naming ENTROLY_DIR, got ${JSON.stringify(outcome)}`);
  assert(outcome.error.includes(outcome.mine), `${label}: the error does not name ${outcome.mine}`);
}

try {
  {
    const taken = scenario('taken', (tmp) => {
      fs.writeFileSync(path.join(tmp, 'entroly'), 'another account got here first');
    });
    assert(taken.dir && taken.dir.startsWith(taken.mine + path.sep),
      `fallback ignored the per-account directory: ${JSON.stringify(taken)}`);
    const stat = fs.lstatSync(taken.mine);
    assert((stat.mode & 0o777) === 0o700, `per-account directory is mode ${(stat.mode & 0o777).toString(8)}`);
    assert(stat.uid === process.getuid(), 'per-account directory has the wrong owner');
  }
  {
    const planted = scenario('planted', (tmp) => {
      fs.mkdirSync(path.join(tmp, 'entroly'));
      fs.chmodSync(path.join(tmp, 'entroly'), 0o777);
    });
    const shared = path.join(planted.tmp, 'entroly');
    assert(planted.dir && !planted.dir.startsWith(shared + path.sep), `wrote into a planted directory: ${planted.dir}`);
    assert(fs.readdirSync(shared).length === 0, 'created files in a planted directory');
  }
  refused('world-writable per-account directory',
    scenario('world-writable', (tmp, mine) => { fs.mkdirSync(mine); fs.chmodSync(mine, 0o777); }));
  refused('symlinked per-account directory', scenario('symlink', (tmp, mine) => {
    const elsewhere = path.join(tmp, '..', 'chosen-by-someone-else');
    fs.mkdirSync(elsewhere, { mode: 0o700 });
    fs.symlinkSync(elsewhere, mine, 'dir');
  }));
  refused('per-account path that is a file',
    scenario('not-a-directory', (tmp, mine) => { fs.writeFileSync(mine, 'taken too'); }));

  console.log('Private temp fallback contract: PASS');
} finally {
  fs.rmSync(root, { recursive: true, force: true });
}
