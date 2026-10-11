import assert from 'node:assert/strict';
import test from 'node:test';

import { BufferOverlay, OverlayTransport } from './bufferOverlay';

interface Call {
  op: 'push' | 'clear';
  path: string;
  content?: string;
}

function recorder(failOn?: (path: string) => boolean) {
  const calls: Call[] = [];
  const transport: OverlayTransport = {
    async push(path, content) {
      if (failOn?.(path)) {
        throw new Error('transport down');
      }
      calls.push({ op: 'push', path, content });
    },
    async clear(path) {
      if (failOn?.(path)) {
        throw new Error('transport down');
      }
      calls.push({ op: 'clear', path });
    },
  };
  return { calls, transport };
}

test('nothing is sent until flush', async () => {
  const { calls, transport } = recorder();
  const overlay = new BufferOverlay(transport);

  overlay.recordChange('/w/a.py', 'one', true);
  overlay.recordChange('/w/a.py', 'two', true);
  assert.equal(calls.length, 0, 'recording must not send');

  await overlay.flush();
  assert.deepEqual(calls, [{ op: 'push', path: '/w/a.py', content: 'two' }]);
});

test('a burst of edits collapses to the final text', async () => {
  const { calls, transport } = recorder();
  const overlay = new BufferOverlay(transport);

  for (const text of ['a', 'ab', 'abc', 'abcd']) {
    overlay.recordChange('/w/a.py', text, true);
  }
  await overlay.flush();

  assert.equal(calls.length, 1, 'one push per flush, not one per keystroke');
  assert.equal(calls[0].content, 'abcd');
});

test('saving retracts the overlay so disk is authoritative again', async () => {
  const { calls, transport } = recorder();
  const overlay = new BufferOverlay(transport);

  overlay.recordChange('/w/a.py', 'dirty', true);
  await overlay.flush();
  assert.deepEqual(overlay.activePaths(), ['/w/a.py']);

  overlay.recordSave('/w/a.py');
  await overlay.flush();

  assert.deepEqual(calls.map((c) => c.op), ['push', 'clear']);
  assert.deepEqual(overlay.activePaths(), []);
});

test('closing retracts the overlay', async () => {
  const { calls, transport } = recorder();
  const overlay = new BufferOverlay(transport);

  overlay.recordChange('/w/a.py', 'dirty', true);
  await overlay.flush();
  overlay.recordClose('/w/a.py');
  await overlay.flush();

  assert.deepEqual(overlay.activePaths(), []);
  assert.equal(calls.at(-1)?.op, 'clear');
});

test('an undo back to the saved state retracts rather than pushing', async () => {
  const { calls, transport } = recorder();
  const overlay = new BufferOverlay(transport);

  overlay.recordChange('/w/a.py', 'dirty', true);
  await overlay.flush();

  // The editor reports isDirty=false once the buffer matches disk again.
  overlay.recordChange('/w/a.py', 'saved text', false);
  await overlay.flush();

  assert.deepEqual(overlay.activePaths(), []);
  assert.equal(calls.at(-1)?.op, 'clear');
});

test('saving a file that was never pushed sends nothing', async () => {
  const { calls, transport } = recorder();
  const overlay = new BufferOverlay(transport);

  overlay.recordSave('/w/never-dirty.py');
  await overlay.flush();

  assert.deepEqual(calls, [], 'no overlay was installed, so none to retract');
});

test('secrets and vendored paths are never pushed', async () => {
  const { calls, transport } = recorder();
  const overlay = new BufferOverlay(transport);

  for (const path of [
    '/w/.env',
    '/w/.env.local',
    '/w/node_modules/pkg/index.js',
    '/w/.git/COMMIT_EDITMSG',
    '/w/certs/server.pem',
  ]) {
    assert.equal(overlay.eligible(path, 'x'), false, path);
    overlay.recordChange(path, 'x', true);
  }
  await overlay.flush();

  assert.deepEqual(calls, []);
});

test('an oversized buffer is not pushed', async () => {
  const { calls, transport } = recorder();
  const overlay = new BufferOverlay(transport, { maxBytes: 16 });

  overlay.recordChange('/w/big.py', 'x'.repeat(64), true);
  await overlay.flush();

  assert.deepEqual(calls, []);
});

test('byte length, not character count, bounds the push', async () => {
  const overlay = new BufferOverlay(recorder().transport, { maxBytes: 8 });
  // Four CJK characters are 12 bytes in UTF-8 but only 4 characters.
  assert.equal(overlay.eligible('/w/a.py', '这是一个'), false);
});

test('a failed push does not retry and does not claim success', async () => {
  const { calls, transport } = recorder((p) => p === '/w/a.py');
  const overlay = new BufferOverlay(transport);

  overlay.recordChange('/w/a.py', 'dirty', true);
  await overlay.flush();

  assert.deepEqual(calls, [], 'nothing recorded because the transport threw');
  assert.deepEqual(overlay.activePaths(), [], 'must not claim an overlay it failed to install');
  assert.equal(overlay.pendingCount(), 0, 'a failed push is dropped, not retried forever');
});

test('dispose retracts every installed overlay', async () => {
  const { calls, transport } = recorder();
  const overlay = new BufferOverlay(transport);

  overlay.recordChange('/w/a.py', 'x', true);
  overlay.recordChange('/w/b.py', 'y', true);
  await overlay.flush();
  assert.deepEqual(overlay.activePaths(), ['/w/a.py', '/w/b.py']);

  await overlay.dispose();

  assert.deepEqual(overlay.activePaths(), []);
  assert.deepEqual(
    calls.filter((c) => c.op === 'clear').map((c) => c.path).sort(),
    ['/w/a.py', '/w/b.py']
  );
});
