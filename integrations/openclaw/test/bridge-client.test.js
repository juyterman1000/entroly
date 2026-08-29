import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { EventEmitter } from "node:events";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { PassThrough } from "node:stream";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  ENTROLY_BRIDGE_SCHEMA,
  EntrolyBridgeClient,
  buildBridgeEnvironment,
  validateBridgeHealth,
} from "../bridge-client.js";

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
          schema_version: ENTROLY_BRIDGE_SCHEMA,
          provider_independent: true,
          receipt_commit_protocol: "two_phase",
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

test("health rejects an incompatible Python bridge with an upgrade action", () => {
  assert.throws(
    () =>
      validateBridgeHealth({
        ok: true,
        schema_version: "entroly.openclaw.bridge.v1",
        provider_independent: true,
      }),
    /pip install -U entroly/,
  );
});

test("bridge environment excludes provider credentials and unrelated host secrets", () => {
  const environment = buildBridgeEnvironment({
    PATH: "/usr/local/bin:/usr/bin",
    HOME: "/home/operator",
    ENTROLY_MODEL_REGISTRY: "/home/operator/models.json",
    ENTROLY_OPENCLAW_RECEIPT_KEY_FILE: "/home/operator/receipt.key",
    OPENAI_API_KEY: "openai-secret",
    ANTHROPIC_API_KEY: "anthropic-secret",
    OPENROUTER_API_KEY: "openrouter-secret",
    AWS_SECRET_ACCESS_KEY: "aws-secret",
    CLAWHUB_TOKEN: "clawhub-secret",
  });

  assert.deepEqual(environment, {
    PATH: "/usr/local/bin:/usr/bin",
    HOME: "/home/operator",
    ENTROLY_MODEL_REGISTRY: "/home/operator/models.json",
    ENTROLY_OPENCLAW_RECEIPT_KEY_FILE: "/home/operator/receipt.key",
  });
});

test("spawn receives only the bounded bridge environment", async () => {
  const child = createFakeChild();
  let spawnOptions;
  const client = new EntrolyBridgeClient({
    environment: {
      PATH: "/usr/bin",
      ENTROLY_AIR_GAP: "1",
      OPENAI_API_KEY: "must-not-cross-the-boundary",
    },
    spawnProcess: (_command, _args, options) => {
      spawnOptions = options;
      return child;
    },
    timeoutMs: 100,
  });

  await client.health();
  assert.deepEqual(spawnOptions.env, {
    PATH: "/usr/bin",
    ENTROLY_AIR_GAP: "1",
  });
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

test(
  "real Python bridge completes the health contract",
  { skip: !process.env.ENTROLY_TEST_PYTHON },
  async () => {
    const keyDir = await mkdtemp(join(tmpdir(), "entroly-openclaw-health-"));
    const client = new EntrolyBridgeClient({
      pythonCommand: process.env.ENTROLY_TEST_PYTHON,
      timeoutMs: 10_000,
      environment: {
        ...process.env,
        ENTROLY_OPENCLAW_RECEIPT_KEY_FILE: join(keyDir, "signing.key"),
      },
    });
    try {
      const result = await client.health();
      assert.equal(result.ok, true);
      assert.equal(result.status, "ready");
      assert.equal(result.schema_version, ENTROLY_BRIDGE_SCHEMA);
    } finally {
      await client.dispose();
      await rm(keyDir, { recursive: true, force: true });
    }
  },
);

test(
  "real Python bridge preserves opaque blocks and commits an accepted receipt",
  { skip: !process.env.ENTROLY_TEST_PYTHON },
  async () => {
    const receiptDir = await mkdtemp(join(tmpdir(), "entroly-openclaw-test-"));
    const signingKeyDir = await mkdtemp(join(tmpdir(), "entroly-openclaw-key-"));
    const signingKeyPath = join(signingKeyDir, "signing.key");
    const client = new EntrolyBridgeClient({
      pythonCommand: process.env.ENTROLY_TEST_PYTHON,
      timeoutMs: 10_000,
      environment: {
        ...process.env,
        ENTROLY_OPENCLAW_RECEIPT_KEY_FILE: signingKeyPath,
      },
    });
    const signedBlock = {
      type: "text",
      text: "signed replay text",
      textSignature: "opaque-signature",
    };
    const messages = [
      {
        role: "assistant",
        content: [
          { type: "text", text: "old provider-neutral detail ".repeat(600) },
          signedBlock,
        ],
        provider: "acme-private",
        model: "future-model-v99",
      },
      { role: "user", content: "continue safely" },
    ];
    const receiptCommitToken = "b".repeat(64);
    const receiptCommitChallenge = createHash("sha256")
      .update(receiptCommitToken, "utf8")
      .digest("hex");
    try {
      const result = await client.request({
        operation: "assemble",
        session_id: "node-real-bridge",
        messages,
        token_budget: 300,
        preserve_last_n: 1,
        workspace_dir: process.cwd(),
        receipt_dir: receiptDir,
        receipt_commit_challenge_sha256: receiptCommitChallenge,
        distill: false,
      });
      assert.equal(result.ok, true);
      assert.equal(result.changed, true);
      assert.deepEqual(result.messages[0].content[1], signedBlock);
      assert.equal(result.messages[0].provider, "acme-private");
      assert.equal(result.receipt_commit_required, true);

      const proposed = JSON.parse(await readFile(result.receipt_path, "utf8"));
      assert.equal(proposed.acceptance_status, "proposed");
      const committed = await client.request({
        operation: "commit_receipt",
        receipt_id: result.receipt_id,
        proposal_id: result.proposal_id,
        proposal_sha256: result.proposal_sha256,
        receipt_path: result.receipt_path,
        receipt_commit_token: receiptCommitToken,
        workspace_dir: process.cwd(),
      });
      assert.equal(committed.committed, true);
      const accepted = JSON.parse(await readFile(result.receipt_path, "utf8"));
      assert.equal(accepted.acceptance_status, "accepted");
      assert.match(accepted.acceptance_signature, /^[0-9a-f]{64}$/);
      assert.notEqual(accepted.acceptance_signature, receiptCommitToken);
      assert.equal(
        committed.acceptance_commit_sha256,
        accepted.acceptance_commit_sha256,
      );
    } finally {
      await client.dispose();
      await rm(receiptDir, { recursive: true, force: true });
      await rm(signingKeyDir, { recursive: true, force: true });
    }
  },
);
