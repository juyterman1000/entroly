const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const launcher = fs.readFileSync(path.join(__dirname, 'index.js'), 'utf8');

function invoke(result, args = []) {
  const calls = [];
  const errors = [];
  let exitCode;
  const context = {
    require(name) {
      assert.equal(name, 'child_process');
      return {
        spawnSync(...call) {
          calls.push(call);
          return result;
        },
      };
    },
    process: {
      argv: ['node', 'index.js', ...args],
      exit(code) {
        exitCode = code;
        throw new Error('process exited');
      },
    },
    console: { error(message) { errors.push(message); } },
  };
  assert.throws(() => vm.runInNewContext(launcher, context), /process exited/);
  return { calls, errors, exitCode };
}

test('forwards arguments and the Python exit status in one launch', () => {
  const { calls, exitCode } = invoke({ status: 7 }, ['serve', '--verbose']);
  assert.equal(calls.length, 1);
  assert.equal(calls[0][0], 'entroly');
  assert.deepEqual(Array.from(calls[0][1]), ['serve', '--verbose']);
  assert.equal(calls[0][2].stdio, 'inherit');
  assert.equal(exitCode, 7);
});

test('explains how to install the missing Python command', () => {
  const { calls, errors, exitCode } = invoke({ status: null, error: { code: 'ENOENT' } });
  assert.equal(calls.length, 1);
  assert.equal(exitCode, 1);
  assert.match(errors.join('\n'), /pip install entroly/);
});

test('reports other launch failures instead of blaming installation', () => {
  const { errors, exitCode } = invoke({ status: null, error: { code: 'EACCES', message: 'permission denied' } });
  assert.equal(exitCode, 1);
  assert.match(errors.join('\n'), /permission denied/);
  assert.doesNotMatch(errors.join('\n'), /pip install entroly/);
});
