#!/usr/bin/env node
// README check commands must not dead-end on the Node CLI.
//
// Dogfooding 1.0.88: `npm install -g entroly`, then the README's
// `entroly verify-claims` printed "Unknown command" and help titled
// "Usage: entroly-wasm" for a binary the user knew as `entroly`.

'use strict';

const assert = require('assert');
const path = require('path');
const { spawnSync } = require('child_process');

const cli = path.join(__dirname, 'bin', 'entroly-wasm.js');

function run(args, env = {}) {
  return spawnSync(process.execPath, [cli, ...args], {
    encoding: 'utf8',
    env: { ...process.env, NO_COLOR: '1', ...env },
  });
}

for (const cmd of ['verify-claims', 'simulate', 'doctor']) {
  const result = run([cmd]);
  assert.strictEqual(result.status, 1, `${cmd} should exit 1`);
  assert.match(result.stderr, /part of the Python package/, `${cmd} should say where it lives`);
  assert.match(result.stderr, new RegExp(`uvx --from entroly entroly ${cmd}`));
  assert.doesNotMatch(result.stderr, /Unknown command/);
}

const unknown = run(['definitely-not-a-command']);
assert.strictEqual(unknown.status, 1);
assert.match(unknown.stderr, /Unknown command/);

const asEntroly = run(['--help'], { ENTROLY_CLI_NAME: 'entroly' });
assert.strictEqual(asEntroly.status, 0);
assert.match(asEntroly.stdout, /Usage:\s+entroly \[command\]/);

const asWasm = run(['--help'], { ENTROLY_CLI_NAME: '' });
assert.match(asWasm.stdout, /Usage:\s+entroly-wasm \[command\]/);

console.log('test_cli_python_only_commands: ok');
