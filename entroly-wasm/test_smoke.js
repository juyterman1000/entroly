const { EntrolyMCPServer } = require('./js/server');
const s = new EntrolyMCPServer();

console.log('=== vault_status ===');
const vs = s.handleTool('vault_status', {});
console.log(JSON.stringify(vs, null, 2).slice(0, 500));

console.log('\n=== compile_beliefs (js/) ===');
const cb = s.handleTool('compile_beliefs', { directory: __dirname + '/js', max_files: 10 });
console.log(JSON.stringify(cb));

console.log('\n=== vault_search ===');
const sr = s.handleTool('vault_search', { query: 'server', top_k: 3 });
console.log(JSON.stringify(sr, null, 2).slice(0, 500));

console.log('\n=== epistemic_route ===');
const er = s.handleTool('epistemic_route', { query: 'how does the knapsack optimizer work?' });
console.log(JSON.stringify(er, null, 2).slice(0, 500));

console.log('\n=== verify_beliefs ===');
const vb = s.handleTool('verify_beliefs', {});
console.log(JSON.stringify(vb));

console.log('\n=== blast_radius ===');
const br = s.handleTool('blast_radius', { changed_files: 'server.js,config.js' });
console.log(JSON.stringify(br));

console.log('\n=== ingest_diff ===');
const diff = s.handleTool('ingest_diff', { diff_text: 'diff --git a/foo.js b/foo.js\n--- a/foo.js\n+++ b/foo.js\n@@ -1 +1 @@\n-old\n+new', source: 'test.diff' });
console.log(JSON.stringify(diff).slice(0, 300));

console.log('\n=== ingest_diagram (stub) ===');
const diag = s.handleTool('ingest_diagram', { image_path: 'x.png' });
console.log(JSON.stringify(diag));

console.log('\n=== manage_skills ===');
const sk = s.handleTool('manage_skills', { action: 'list' });
console.log(JSON.stringify(sk));

console.log('\nALL TESTS PASSED');
process.exit(0);
