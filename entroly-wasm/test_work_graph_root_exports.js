'use strict';

const pkg = require('./');

function assert(condition, message) {
  if (!condition) throw new Error(message || 'assertion failed');
}

for (const name of [
  'WorkGraph',
  'WorkGraphStore',
  'discoverRepositoryIdentity',
  'discoverRepositoryObservation',
  'resumeRepository',
  'handoffRepository',
]) {
  assert(typeof pkg[name] === 'function', `missing npm root Work Graph export: ${name}`);
}

assert(Number.isSafeInteger(pkg.MAX_RESUME_EVIDENCE), 'MAX_RESUME_EVIDENCE is not exported');
console.log('Work Graph npm root exports: PASS');
