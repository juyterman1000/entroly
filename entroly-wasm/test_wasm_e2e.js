// E2E Test: Verify entroly-wasm works correctly in Node.js
// This simulates what a user would experience after `npm install entroly`

const wasm = require('./pkg/entroly_wasm.js');

function assert(condition, msg) {
    if (!condition) {
        console.error(`FAIL: ${msg}`);
        process.exit(1);
    }
    console.log(`  PASS: ${msg}`);
}

function test_basic_lifecycle() {
    console.log('\n=== Test 1: Basic Lifecycle ===');
    
    const engine = new wasm.WasmEntrolyEngine();
    assert(engine.fragment_count() === 0, 'Engine starts empty');

    // Ingest a Python file
    const r1 = engine.ingest('auth.py', 
        'def authenticate(user, password):\n    h = hashlib.sha256(password.encode())\n    return db.verify(user, h.hexdigest())',
        50, false);
    assert(r1 !== null, 'Ingest returns result');
    assert(r1.fragment_id !== undefined, 'Result has fragment_id');
    assert(r1.token_count === 50, 'Token count preserved');
    assert(r1.entropy_score > 0, `Entropy score is positive: ${r1.entropy_score.toFixed(3)}`);
    assert(r1.is_duplicate === false, 'First ingest is not duplicate');

    assert(engine.fragment_count() === 1, 'Fragment count is 1');

    // Ingest a Java file
    const r2 = engine.ingest('UserService.java',
        'public class UserService {\n    public User findById(Long id) {\n        return userRepository.findById(id).orElseThrow();\n    }\n}',
        80, false);
    assert(r2.is_duplicate === false, 'Java file is unique');
    assert(engine.fragment_count() === 2, 'Fragment count is 2');

    // Ingest a Go file
    const r3 = engine.ingest('main.go',
        'func main() {\n    http.HandleFunc("/api/users", handleUsers)\n    log.Fatal(http.ListenAndServe(":8080", nil))\n}',
        40, false);
    assert(engine.fragment_count() === 3, 'Mixed codebase: 3 fragments');

    engine.free();
    console.log('  Test 1 PASSED');
}

function test_optimize() {
    console.log('\n=== Test 2: Optimize (Knapsack) ===');
    
    const engine = new wasm.WasmEntrolyEngine();

    // Ingest several fragments with varying quality
    engine.ingest('core_logic.py', 
        'def calculate_risk_score(portfolio, market_data):\n    volatility = compute_volatility(market_data)\n    exposure = sum(p.weight * p.risk for p in portfolio)\n    return volatility * exposure * correlation_factor(portfolio)',
        200, false);

    engine.ingest('imports.py',
        'import os\nimport sys\nimport json\nimport logging\nimport pathlib\nfrom typing import List, Dict, Optional',
        100, false);

    engine.ingest('auth_handler.py',
        'def verify_jwt_token(token, secret):\n    try:\n        payload = jwt.decode(token, secret, algorithms=["HS256"])\n        return payload["user_id"]\n    except jwt.ExpiredSignatureError:\n        raise AuthenticationError("Token expired")',
        150, false);

    // Budget only fits ~250 tokens — should prefer high-entropy code over imports
    const result = engine.optimize(250, 'find risk calculation bug');
    assert(result !== null, 'Optimize returns result');
    assert(result.token_budget > 0, `Token budget set: ${result.token_budget}`);
    assert(result.fragments_considered === 3, 'All 3 fragments considered');

    const selected = result.selected.filter(f => f.selected);
    assert(selected.length > 0, `Selected ${selected.length} fragment(s)`);

    const total = selected.reduce((sum, f) => sum + f.token_count, 0);
    assert(total <= result.token_budget, `Total tokens ${total} <= budget ${result.token_budget}`);

    engine.free();
    console.log('  Test 2 PASSED');
}

function test_dedup() {
    console.log('\n=== Test 3: Deduplication ===');
    
    const engine = new wasm.WasmEntrolyEngine();

    const content = 'def process_data(items):\n    return [transform(item) for item in items]';
    const r1 = engine.ingest('utils.py', content, 30, false);
    assert(r1.is_duplicate === false, 'First ingest is unique');

    // Ingest near-identical content
    const r2 = engine.ingest('utils_copy.py', content, 30, false);
    assert(r2.is_duplicate === true, 'Duplicate detected');
    assert(r2.duplicate_of !== null, `Duplicate of: ${r2.duplicate_of}`);

    // Only the original should be in the engine
    assert(engine.fragment_count() === 1, 'Duplicate not stored');

    engine.free();
    console.log('  Test 3 PASSED');
}

function test_feedback_loop() {
    console.log('\n=== Test 4: RL Feedback Loop ===');
    
    const engine = new wasm.WasmEntrolyEngine();
    
    engine.ingest('good_code.py', 'def critical_algorithm(): return optimized_result()', 50, false);
    engine.ingest('bad_code.py', 'print("hello world")', 20, false);

    // Simulate success/failure feedback
    engine.record_success();
    engine.record_success();
    engine.record_failure();

    // Engine should not crash — feedback is recorded internally
    assert(engine.fragment_count() === 2, 'Fragments intact after feedback');

    engine.free();
    console.log('  Test 4 PASSED');
}

function test_stats() {
    console.log('\n=== Test 5: Engine Stats ===');
    
    const engine = new wasm.WasmEntrolyEngine();
    
    engine.ingest('file1.rs', 'fn main() { println!("hello"); }', 20, false);
    engine.ingest('file2.rs', 'pub struct Config { pub port: u16 }', 25, false);

    const stats = engine.stats();
    assert(stats !== null, 'Stats returns result');
    assert(stats.total_fragments === 2, `Fragment count: ${stats.total_fragments}`);
    assert(stats.total_tokens === 45, `Total tokens: ${stats.total_tokens}`);
    assert(stats.cache !== undefined, 'Cache stats present');
    assert(stats.cache.hit_rate !== undefined, 'Cache hit rate present');

    engine.free();
    console.log('  Test 5 PASSED');
}

function test_remove() {
    console.log('\n=== Test 6: Fragment Removal ===');
    
    const engine = new wasm.WasmEntrolyEngine();
    
    const r1 = engine.ingest('temp.py', 'x = 42', 10, false);
    assert(engine.fragment_count() === 1, 'One fragment');

    const removed = engine.remove(r1.fragment_id);
    assert(removed === true, 'Remove returns true');
    assert(engine.fragment_count() === 0, 'Fragment removed');

    const removed_again = engine.remove('nonexistent');
    assert(removed_again === false, 'Remove nonexistent returns false');

    engine.free();
    console.log('  Test 6 PASSED');
}

function test_clear() {
    console.log('\n=== Test 7: Clear Engine ===');
    
    const engine = new wasm.WasmEntrolyEngine();
    
    for (let i = 0; i < 10; i++) {
        engine.ingest(`file${i}.py`, `content_${i} = ${i * 100}`, 10, false);
    }
    assert(engine.fragment_count() === 10, '10 fragments ingested');

    engine.clear();
    assert(engine.fragment_count() === 0, 'All fragments cleared');

    engine.free();
    console.log('  Test 7 PASSED');
}

function test_pinned_fragments() {
    console.log('\n=== Test 8: Pinned Fragments ===');
    
    const engine = new wasm.WasmEntrolyEngine();

    // Pinned fragment should always be included
    engine.ingest('LICENSE', 'MIT License\nCopyright 2024', 30, true);
    engine.ingest('boilerplate.py', 'import os\nimport sys', 20, false);

    const result = engine.optimize(40, 'check license');
    const selected = result.selected.filter(f => f.selected);
    
    // With only 40 tokens budget — pinned LICENSE (30) should be included
    const licenseSelected = selected.some(f => f.source === 'LICENSE');
    assert(licenseSelected, 'Pinned LICENSE fragment is selected');

    engine.free();
    console.log('  Test 8 PASSED');
}

// ── Run all tests ──────────────────────────────────
console.log('╔═══════════════════════════════════════════════╗');
console.log('║  Entroly Wasm E2E Test Suite                  ║');
console.log('╚═══════════════════════════════════════════════╝');

try {
    test_basic_lifecycle();
    test_optimize();
    test_dedup();
    test_feedback_loop();
    test_stats();
    test_remove();
    test_clear();
    test_pinned_fragments();

    console.log('\n════════════════════════════════════════');
    console.log('  ALL 8 TESTS PASSED ✓');
    console.log('  npm install entroly → VERIFIED');
    console.log('════════════════════════════════════════\n');
} catch (e) {
    console.error(`\nFATAL ERROR: ${e.message}`);
    console.error(e.stack);
    process.exit(1);
}
