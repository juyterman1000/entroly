// Test autotune integration: journal + optimizer + server wiring
const { FeedbackJournal, optimize, normalizeWeights, WEIGHT_KEYS } = require('./js/autotune');
const fs = require('fs');
const path = require('path');
const os = require('os');

const tmpDir = path.join(os.tmpdir(), 'entroly_test_' + Date.now());
fs.mkdirSync(tmpDir, { recursive: true });

let pass = 0, fail = 0;
function test(name, fn) {
  try { fn(); pass++; console.log(`  ✓ ${name}`); }
  catch (e) { fail++; console.log(`  ✗ ${name}: ${e.message}`); }
}
function assert(c, m) { if (!c) throw new Error(m || 'assert failed'); }

console.log('Autotune Integration Tests\n');

// 1. Journal write/read
test('FeedbackJournal: log + load', () => {
  const j = new FeedbackJournal(tmpDir);
  j.log({ weights: { w_r: 0.3, w_f: 0.25, w_s: 0.25, w_e: 0.2 }, reward: 1.0, selectedCount: 5, turn: 1 });
  j.log({ weights: { w_r: 0.35, w_f: 0.2, w_s: 0.25, w_e: 0.2 }, reward: -1.0, selectedCount: 3, turn: 2 });
  j.log({ weights: { w_r: 0.28, w_f: 0.27, w_s: 0.25, w_e: 0.2 }, reward: 1.0, selectedCount: 7, turn: 3 });
  const eps = j.load();
  assert(eps.length === 3, `expected 3, got ${eps.length}`);
});

// 2. Journal stats
test('FeedbackJournal: stats', () => {
  const j = new FeedbackJournal(tmpDir);
  const s = j.stats();
  assert(s.episodes === 3);
  assert(s.successes === 2);
  assert(s.failures === 1);
});

// 3. Optimizer with positive signal
test('optimize: returns blended weights', () => {
  const j = new FeedbackJournal(tmpDir);
  const eps = j.load();
  const current = { w_r: 0.3, w_f: 0.25, w_s: 0.25, w_e: 0.2 };
  const result = optimize(eps, current);
  assert(result !== null, 'optimizer returned null');
  assert(result.blended, 'no blended weights');
  assert(result.optimal, 'no optimal weights');
  // Blended weights should sum to ~1
  const sum = WEIGHT_KEYS.reduce((s, k) => s + result.blended[k], 0);
  assert(Math.abs(sum - 1.0) < 0.01, `weights sum=${sum}`);
});

// 4. Optimizer attracts toward success weights
test('optimize: attracts toward success patterns', () => {
  const j = new FeedbackJournal(tmpDir);
  // Add more success episodes with high w_r
  for (let i = 0; i < 10; i++) {
    j.log({ weights: { w_r: 0.45, w_f: 0.2, w_s: 0.2, w_e: 0.15 }, reward: 1.0, selectedCount: 5, turn: 10 + i });
  }
  // Add failures with low w_r
  for (let i = 0; i < 5; i++) {
    j.log({ weights: { w_r: 0.15, w_f: 0.3, w_s: 0.3, w_e: 0.25 }, reward: -1.0, selectedCount: 3, turn: 20 + i });
  }
  const eps = j.load();
  const current = { w_r: 0.3, w_f: 0.25, w_s: 0.25, w_e: 0.2 };
  const result = optimize(eps, current);
  // Optimal should have w_r pulled TOWARD success (0.45) and AWAY from failure (0.15)
  assert(result.optimal.w_r > current.w_r, `w_r should increase: optimal=${result.optimal.w_r} current=${current.w_r}`);
});

// 5. Exploration bonus
test('optimize: exploration bonus identifies under-explored dim', () => {
  const eps = new FeedbackJournal(tmpDir).load();
  if (eps.length < 5) { for (let i = 0; i < 5; i++) new FeedbackJournal(tmpDir).log({ weights: { w_r: 0.3 + i*0.02, w_f: 0.25, w_s: 0.25, w_e: 0.2 }, reward: 1.0, turn: 30+i }); }
  const result = optimize(new FeedbackJournal(tmpDir).load(), { w_r: 0.3, w_f: 0.25, w_s: 0.25, w_e: 0.2 });
  assert(result !== null, 'optimizer returned null');
  assert(result.exploration !== null, 'no exploration data');
  assert(result.exploration.underExplored, 'no underExplored dimension');
});

// 6. Confidence ramp
test('optimize: confidence ramps with episode count', () => {
  const lowDir = path.join(tmpDir, 'low');
  const highDir = path.join(tmpDir, 'high');
  fs.mkdirSync(lowDir, { recursive: true });
  fs.mkdirSync(highDir, { recursive: true });
  const j1 = new FeedbackJournal(lowDir);
  for (let i = 0; i < 4; i++) j1.log({ weights: { w_r: 0.4, w_f: 0.2, w_s: 0.2, w_e: 0.2 }, reward: 1.0, turn: i });
  const r1 = optimize(j1.load(), { w_r: 0.3, w_f: 0.25, w_s: 0.25, w_e: 0.2 });
  assert(r1 !== null, 'low-count optimizer returned null');

  const j2 = new FeedbackJournal(highDir);
  for (let i = 0; i < 20; i++) j2.log({ weights: { w_r: 0.4, w_f: 0.2, w_s: 0.2, w_e: 0.2 }, reward: 1.0, turn: i });
  const r2 = optimize(j2.load(), { w_r: 0.3, w_f: 0.25, w_s: 0.25, w_e: 0.2 });
  assert(r2 !== null, 'high-count optimizer returned null');

  assert(r2.confidence >= r1.confidence, `high=${r2.confidence} should >= low=${r1.confidence}`);
});

// 7. Regret estimation
test('optimize: regret estimation', () => {
  const j = new FeedbackJournal(tmpDir);
  const eps = j.load();
  const result = optimize(eps, { w_r: 0.3, w_f: 0.25, w_s: 0.25, w_e: 0.2 });
  assert(typeof result.estimatedRegret === 'number');
  assert(result.estimatedRegret >= 0, 'regret should be non-negative');
});

// 8. Prune
test('FeedbackJournal: prune removes old', () => {
  const pruneDir = path.join(tmpDir, 'prune_test');
  fs.mkdirSync(pruneDir, { recursive: true });
  const j = new FeedbackJournal(pruneDir);
  j.log({ weights: { w_r: 0.3, w_f: 0.25, w_s: 0.25, w_e: 0.2 }, reward: 1.0, turn: 1 });
  assert(j.count() === 1);
  // prune with maxAge = -1 (everything is "too old")
  j.prune(-1);
  j._cache = null; // force reload
  const after = j.count();
  assert(after === 0, `should be empty after prune, got ${after}`);
});

// Cleanup
fs.rmSync(tmpDir, { recursive: true, force: true });

console.log(`\n${'='.repeat(50)}`);
console.log(`Results: ${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
else console.log('All autotune tests passed! ✓');
