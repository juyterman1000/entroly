#!/usr/bin/env node
/**
 * Tests for Federation + Evolution Daemon (JavaScript)
 * =====================================================
 * Run: node test_federation_js.js
 */

'use strict';

const fs = require('fs');
const path = require('path');
const os = require('os');
const crypto = require('crypto');

const {
  GitHubTransport,
  FederationClient,
  createContribution,
  trimmedMean,
  getClientId,
  calibrateSigma,
  addNoise,
  WEIGHT_KEYS,
} = require('./js/federation');

const { EvolutionDaemon, DreamingLoop } = require('./js/evolution_daemon');
const { FeedbackJournal } = require('./js/autotune');

let passed = 0;
let failed = 0;

function assert(cond, msg) {
  if (cond) { passed++; }
  else { failed++; console.error(`  FAIL: ${msg}`); }
}

// ── Test: Client ID generation ──
{
  const tmpDir = path.join(os.tmpdir(), `entroly-test-${Date.now()}`);
  fs.mkdirSync(tmpDir, { recursive: true });

  const id1 = getClientId(tmpDir);
  assert(id1.length >= 32, 'Client ID ≥ 32 chars');

  // Same directory → same ID (persistent)
  const id2 = getClientId(tmpDir);
  assert(id1 === id2, 'Client ID stable across calls');

  // Different directory → different ID
  const tmpDir2 = path.join(os.tmpdir(), `entroly-test-${Date.now() + 1}`);
  fs.mkdirSync(tmpDir2, { recursive: true });
  const id3 = getClientId(tmpDir2);
  assert(id1 !== id3, 'Different dirs → different IDs');

  // Cleanup
  fs.rmSync(tmpDir, { recursive: true, force: true });
  fs.rmSync(tmpDir2, { recursive: true, force: true });
}

// ── Test: DP noise calibration ──
{
  const sigma = calibrateSigma(1.0, 1.0);
  assert(sigma > 0, 'Sigma is positive');
  assert(sigma < 10, 'Sigma is reasonable');

  // Noise adds variability
  const values = Array.from({ length: 100 }, () => addNoise(0.5, sigma));
  const min = Math.min(...values);
  const max = Math.max(...values);
  assert(max > min, 'Noise adds variability');
  assert(values.every(v => v >= 0 && v <= 1), 'All noised values clipped to [0,1]');
}

// ── Test: Contribution creation ──
{
  const packet = createContribution(
    'test-client-id', 'js_monorepo',
    { w_r: 0.3, w_f: 0.25, w_s: 0.25, w_e: 0.2 },
    10, 0.85,
  );
  assert(packet !== null, 'Packet created');
  assert(packet.client_id === 'test-client-id', 'Client ID set');
  assert(packet.archetype === 'js_monorepo', 'Archetype set');
  assert(packet.dp_epsilon === 1.0, 'DP epsilon set');
  assert(packet.noise_sigma > 0, 'Noise sigma recorded');
  assert(typeof packet.weights.w_r === 'number', 'Weights noised');
}

// ── Test: Contribution rejected when not confident ──
{
  const packet = createContribution('x', 'test', { w_r: 0.25 }, 2, 0.3);
  assert(packet === null, 'Low confidence → null packet');
}

// ── Test: Trimmed mean ──
{
  const result = trimmedMean([1, 2, 3, 4, 5, 6, 7, 8, 9, 100]);
  assert(result > 2 && result < 50, 'Trimmed mean resists outliers');

  const simple = trimmedMean([5, 5, 5]);
  assert(simple === 5, 'Trimmed mean of same values = value');
}

// ── Test: GitHub transport init ──
{
  const transport = new GitHubTransport();
  assert(transport.canRead === true, 'Public repos readable');
  // Without token, can't write
  const noTokenTransport = new GitHubTransport({ token: undefined });
  // Note: canWrite depends on env var, so we just test the property exists
  assert(typeof noTokenTransport.canWrite === 'boolean', 'canWrite is boolean');
}

// ── Test: Federation client ──
{
  const tmpDir = path.join(os.tmpdir(), `entroly-fed-${Date.now()}`);
  const client = new FederationClient({ dataDir: tmpDir });

  assert(client.clientId.length >= 32, 'Federation client has ID');

  // Contribute
  const packet = client.contribute(
    'test_arch', { w_r: 0.3, w_f: 0.25, w_s: 0.25, w_e: 0.2 }, 10, 0.85,
  );
  assert(packet !== null, 'Local contribution created');

  // Load contributions
  const contribs = client.loadContributions();
  assert(contribs.length === 1, 'One contribution saved');

  // Anti-echo: own packet rejected
  const saved = client.saveContribution(packet);
  assert(saved === false, 'Anti-echo: own packet rejected');

  // Foreign packet accepted
  const foreign = { ...packet, client_id: 'foreign-id-123' };
  const savedForeign = client.saveContribution(foreign);
  assert(savedForeign === true, 'Foreign packet accepted');

  // Cleanup
  fs.rmSync(tmpDir, { recursive: true, force: true });
}

// ── Test: Dreaming loop ──
{
  const tmpDir = path.join(os.tmpdir(), `entroly-dream-${Date.now()}`);
  fs.mkdirSync(tmpDir, { recursive: true });
  const journal = new FeedbackJournal(tmpDir);

  // Log enough episodes
  for (let i = 0; i < 10; i++) {
    journal.log({
      weights: { w_r: 0.25 + Math.random() * 0.1, w_f: 0.25, w_s: 0.25, w_e: 0.25 },
      selectedSources: ['a.js'],
      selectedCount: 3,
      tokenBudget: 1000,
      query: i % 2 === 0 ? 'fix the bug' : 'add feature',
      reward: i % 3 === 0 ? 1.0 : -0.5,
      turn: i,
    });
  }

  const dreaming = new DreamingLoop(journal, { idleThresholdMs: 100 });

  // Force idle
  dreaming._lastActivity = Date.now() - 200;
  assert(dreaming.shouldDream() === true, 'Should dream when idle');

  dreaming.recordActivity();
  assert(dreaming.shouldDream() === false, 'Should not dream after activity');

  // Force idleness
  dreaming._lastActivity = Date.now() - 100000;
  const result = dreaming.runDreamCycle({ w_r: 0.25, w_f: 0.25, w_s: 0.25, w_e: 0.25 });
  assert(result.status === 'completed', 'Dream cycle completed');
  assert(typeof result.bestScore === 'number', 'Best score is number');
  assert(result.trials === 5, 'Ran 5 trials');

  fs.rmSync(tmpDir, { recursive: true, force: true });
}

// ── Test: Evolution daemon init ──
{
  const tmpDir = path.join(os.tmpdir(), `entroly-evo-${Date.now()}`);
  const daemon = new EvolutionDaemon({ checkpointDir: tmpDir });

  assert(daemon._dreaming instanceof DreamingLoop, 'Daemon has dreaming loop');
  assert(typeof daemon.stats === 'object', 'Daemon has stats');
  assert(daemon.stats.dreamCycles === 0, 'Fresh daemon: 0 dream cycles');

  // Start and stop
  daemon.start();
  assert(daemon._timer !== null, 'Timer running');
  daemon.stop();
  assert(daemon._timer === null, 'Timer stopped');

  fs.rmSync(tmpDir, { recursive: true, force: true });
}

// ── Results ──
console.log(`\n  Federation + Evolution JS: ${passed} passed, ${failed} failed\n`);
process.exit(failed > 0 ? 1 : 0);
