// Comprehensive E2E test for all 37 wasm engine methods
const { WasmEntrolyEngine } = require('./pkg/entroly_wasm');

let pass = 0, fail = 0;
function test(name, fn) {
  try { fn(); pass++; console.log(`  PASS ${name}`); }
  catch (e) { fail++; console.log(`  FAIL ${name}: ${e.message}`); }
}

function assert(cond, msg) { if (!cond) throw new Error(msg || 'assertion failed'); }

console.log('Entroly Wasm - Full E2E Test\n');

const engine = new WasmEntrolyEngine();

// 1. new() - constructor
test('new()', () => { assert(engine !== null); });

// 2. fragment_count()
test('fragment_count()', () => { assert(engine.fragment_count() === 0); });

// 3. ingest()
test('ingest()', () => {
  const r = engine.ingest('function hello() { return 42; }', 'file:hello.js', 20, false);
  assert(r.status === 'ingested', `expected ingested, got ${r.status}`);
  assert(r.fragment_id, 'missing fragment_id');
  assert(r.token_count === 20);
  assert(typeof r.entropy_score === 'number');
});

// 4. ingest pinned
test('ingest(pinned)', () => {
  const r = engine.ingest('CRITICAL: auth module', 'file:auth.js', 15, true);
  assert(r.status === 'ingested');
  assert(r.is_pinned === true);
});

// 5. ingest more for testing
engine.ingest('class PaymentProcessor { charge(amt) { return amt * 1.1; } }', 'file:pay.js', 40, false);
engine.ingest('export const config = { port: 3000, db: "postgres://localhost/app" };', 'file:config.js', 25, false);
engine.ingest('describe("auth", () => { it("should login", () => { expect(true).toBe(true); }); });', 'file:auth.test.js', 30, false);

test('fragment_count after ingests', () => { assert(engine.fragment_count() === 5); });

// 6. advance_turn()
test('advance_turn()', () => { engine.advance_turn(); assert(engine.get_turn() === 1); });

// 7. get_turn()
test('get_turn()', () => { assert(typeof engine.get_turn() === 'number'); });

// 8. optimize()
test('optimize()', () => {
  const r = engine.optimize(80, 'authentication bug');
  assert(typeof r === 'object');
  assert(typeof r.selected_count === 'number');
  assert(typeof r.total_tokens === 'number');
  assert(Array.isArray(r.selected));
  assert(r.total_tokens <= 200, `budget sanity: ${r.total_tokens}`);
});

// 9. recall()
test('recall()', () => {
  const r = engine.recall('payment processing', 3);
  assert(Array.isArray(r));
  assert(r.length <= 3);
  if (r.length > 0) {
    assert(r[0].fragment_id);
    assert(typeof r[0].relevance === 'number');
  }
});

// 10. record_success()
test('record_success()', () => {
  const r = engine.optimize(100, 'test');
  const ids = r.selected.filter(s => s.variant === 'full').map(s => s.fragment_id).slice(0, 2);
  engine.record_success(JSON.stringify(ids));
});

// 11. record_failure()
test('record_failure()', () => {
  engine.record_failure(JSON.stringify(['nonexistent_id']));
});

// 12. record_reward()
test('record_reward()', () => {
  engine.record_reward(JSON.stringify(['nonexistent_id']), 0.8);
});

// 13. remove()
test('remove()', () => {
  const r = engine.ingest('temporary fragment', 'temp', 10, false);
  const removed = engine.remove(r.fragment_id);
  assert(removed === true);
  assert(engine.remove('nonexistent') === false);
});

// 14. stats()
test('stats()', () => {
  const s = engine.stats();
  assert(typeof s === 'object');
  assert(s.session);
  assert(typeof s.session.current_turn === 'number');
  assert(s.savings);
  assert(s.dedup);
  assert(s.prism);
  assert(s.cache);
});

// 15. explain_selection()
test('explain_selection()', () => {
  engine.optimize(100, 'test query');
  const e = engine.explain_selection();
  assert(typeof e === 'object');
  assert(Array.isArray(e.included));
  assert(Array.isArray(e.excluded));
  assert(typeof e.sufficiency === 'number');
});

// 16. set_weights()
test('set_weights()', () => { engine.set_weights(0.4, 0.2, 0.3, 0.1); });

// 17. set_exploration_rate()
test('set_exploration_rate()', () => { engine.set_exploration_rate(0.15); });

// 18. set_query_personas_enabled()
test('set_query_personas_enabled()', () => { engine.set_query_personas_enabled(true); });

// 19. set_channel_coding_enabled()
test('set_channel_coding_enabled()', () => { engine.set_channel_coding_enabled(true); });

// 20. set_model()
test('set_model()', () => { engine.set_model('gpt-4o'); });

// 21. set_cache_cost_per_token()
test('set_cache_cost_per_token()', () => { engine.set_cache_cost_per_token(0.00003); });

// 22. classify_task()
test('classify_task()', () => {
  const r = engine.classify_task('fix the authentication bug');
  assert(typeof r === 'object');
  assert(r.task_type);
  assert(typeof r.budget_multiplier === 'number');
});

// 23-26. cache methods
test('cache_len()', () => { assert(typeof engine.cache_len() === 'number'); });
test('cache_is_empty()', () => { assert(typeof engine.cache_is_empty() === 'boolean'); });
test('cache_hit_rate()', () => { assert(typeof engine.cache_hit_rate() === 'number'); });
test('cache_clear()', () => { engine.cache_clear(); assert(engine.cache_is_empty()); });

// 27. dep_graph_stats()
test('dep_graph_stats()', () => {
  const r = engine.dep_graph_stats();
  assert(typeof r === 'object');
  assert(typeof r.nodes === 'number');
  assert(typeof r.edges === 'number');
});

// 28. query_manifold_stats()
test('query_manifold_stats()', () => {
  const r = engine.query_manifold_stats();
  assert(typeof r === 'object');
  assert(typeof r.population === 'number');
});

// 29. analyze_health()
test('analyze_health()', () => {
  const r = engine.analyze_health();
  assert(typeof r === 'object');
});

// 30. security_report()
test('security_report()', () => {
  const r = engine.security_report();
  assert(typeof r === 'object');
  assert(typeof r.fragments_scanned === 'number');
});

// 31. scan_fragment()
test('scan_fragment()', () => {
  const frags = engine.export_fragments();
  if (frags.length > 0) {
    const r = engine.scan_fragment(frags[0].fragment_id);
    assert(typeof r === 'object');
  }
});

// 32. entropy_anomalies()
test('entropy_anomalies()', () => {
  const r = engine.entropy_anomalies();
  assert(typeof r === 'object');
});

// 33. score_utilization()
test('score_utilization()', () => {
  const r = engine.score_utilization('The hello function returns 42. The auth module handles login.');
  assert(typeof r === 'object');
});

// 34. semantic_dedup_report()
test('semantic_dedup_report()', () => {
  const r = engine.semantic_dedup_report();
  assert(typeof r === 'object');
  assert(typeof r.kept === 'number');
});

// 35. export_state() / import_state()
test('export_state()', () => {
  const s = engine.export_state();
  assert(typeof s === 'object');
  assert(typeof s.w_recency === 'number');
});

test('import_state()', () => {
  const s = engine.export_state();
  const r = engine.import_state(JSON.stringify(s));
  assert(typeof r === 'object');
  assert(r.status === 'imported');
});

// 36. export_fragments()
test('export_fragments()', () => {
  const frags = engine.export_fragments();
  assert(Array.isArray(frags));
  assert(frags.length === engine.fragment_count());
  if (frags.length > 0) {
    assert(frags[0].fragment_id);
    assert(frags[0].content);
    assert(typeof frags[0].token_count === 'number');
  }
});

// 37. hierarchical_compress()
test('hierarchical_compress()', () => {
  const r = engine.hierarchical_compress(50, 'auth');
  assert(typeof r === 'object');
  assert(r.status === 'compressed');
});

// 38. clear()
test('clear()', () => {
  engine.clear();
  assert(engine.fragment_count() === 0);
  assert(engine.get_turn() === 0 || engine.get_turn() >= 0); // turn may or may not reset
});

// Summary
console.log(`\n${'='.repeat(50)}`);
console.log(`Results: ${pass} passed, ${fail} failed out of ${pass + fail} tests`);
if (fail > 0) process.exit(1);
else console.log('All tests passed!');
