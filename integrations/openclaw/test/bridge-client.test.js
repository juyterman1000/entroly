import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import test from "node:test";

import { EntrolyBridgeClient } from "../bridge-client.js";

function createFakeChild({ respond = true } = {}) {
  const child = new EventEmitter();
  child.stdin = new PassThrough();
  child.stdout = new PassThrough();
  child.stderr = new PassThrough();
  child.killed = false;
  child.kill = () => {
    child.killed = true;
    queueMicrotask(() => child.emit("exit", null, "SIGTERM"));
    return true;
  };
  if (respond) {
    child.stdin.on("data", (chunk) => {
      const request = JSON.parse(String(chunk).trim());
      child.stdout.write(
        `${JSON.stringify({
          ok: true,
          request_id: request.request_id,
          status: "ready",
        })}\n`,
      );
    });
  }
  return child;
}

test("health request correlates a bridge response", async () => {
  const child = createFakeChild();
  const client = new EntrolyBridgeClient({
    spawnProcess: () => child,
    timeoutMs: 100,
  });
  const result = await client.health();
  assert.equal(result.status, "ready");
  await client.dispose();
});

test("timeout kills a wedged bridge and the next request restarts", async () => {
  const children = [createFakeChild({ respond: false }), createFakeChild()];
  let spawnCount = 0;
  const client = new EntrolyBridgeClient({
    spawnProcess: () => children[spawnCount++],
    timeoutMs: 10,
  });
  await assert.rejects(client.health(), /timed out/);
  assert.equal(children[0].killed, true);

  const recovered = await client.health();
  assert.equal(recovered.status, "ready");
  assert.equal(spawnCount, 2);
  await client.dispose();
});
