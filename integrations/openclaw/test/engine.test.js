import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";

import { ENTROLY_BRIDGE_SCHEMA } from "../bridge-client.js";
import {
  createEntrolyContextEngine,
  formatEntrolyDoctor,
  formatEntrolyStatus,
} from "../engine.js";

function createTestEngine(options) {
  return createEntrolyContextEngine({
    delegateCompaction: async ({ currentTokenCount }) => ({
      ok: true,
      compacted: false,
      result: { tokensBefore: currentTokenCount ?? 0 },
    }),
    ...options,
  });
}

const TEST_RECEIPT_ID = "ocr_11111111111111111111";
const TEST_PROPOSAL_ID = "ocp_22222222222222222222222222222222";
const TEST_PROPOSAL_SHA256 = "3".repeat(64);
const TEST_RECEIPT_PATH =
  `/workspace/.entroly/receipts/openclaw/s1-${TEST_RECEIPT_ID}-${TEST_PROPOSAL_ID}.json`;

function sha256(value) {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

test("assemble delegates to Entroly and reports assembled authority", async () => {
  const messages = [{ role: "user", content: "hello" }];
  const requests = [];
  const bridge = {
    request: async (request) => {
      requests.push(request);
      if (request.operation === "commit_receipt") {
        return {
          schema_version: ENTROLY_BRIDGE_SCHEMA,
          ok: true,
          receipt_id: TEST_RECEIPT_ID,
          proposal_id: TEST_PROPOSAL_ID,
          proposal_sha256: TEST_PROPOSAL_SHA256,
          receipt_path: TEST_RECEIPT_PATH,
          acceptance_commit_sha256: sha256(
            `entroly.openclaw.accept.v1:${TEST_PROPOSAL_SHA256}:${request.receipt_commit_token}`,
          ),
          committed: true,
        };
      }
      return {
        schema_version: ENTROLY_BRIDGE_SCHEMA,
        ok: true,
        messages,
        estimated_tokens: 9,
        receipt_id: TEST_RECEIPT_ID,
        proposal_id: TEST_PROPOSAL_ID,
        proposal_sha256: TEST_PROPOSAL_SHA256,
        receipt_path: TEST_RECEIPT_PATH,
        receipt_commit_required: true,
      };
    },
    dispose: async () => {},
  };
  const engine = createTestEngine({
    bridge,
    config: { workspaceDir: "/workspace" },
  });
  const result = await engine.assemble({
    sessionId: "s1",
    messages,
    tokenBudget: 100,
    prompt: "hello",
  });
  assert.deepEqual(result.messages, messages);
  assert.equal(result.estimatedTokens, 9);
  assert.equal(result.promptAuthority, "assembled");
  assert.equal(engine.getStatus("s1").receipt_id, TEST_RECEIPT_ID);
  assert.equal(requests[0].workspace_dir, "/workspace");
  assert.equal(requests[1].operation, "commit_receipt");
  assert.equal(requests[1].receipt_id, TEST_RECEIPT_ID);
  assert.equal(requests[1].proposal_id, TEST_PROPOSAL_ID);
  assert.equal(requests[1].proposal_sha256, TEST_PROPOSAL_SHA256);
  assert.equal(requests[1].receipt_path, TEST_RECEIPT_PATH);
  assert.equal(requests[1].workspace_dir, "/workspace");
  assert.match(requests[1].receipt_commit_token, /^[0-9a-f]{64}$/);
  assert.equal(
    requests[0].receipt_commit_challenge_sha256,
    sha256(requests[1].receipt_commit_token),
  );
  assert.equal(engine.getStatus("s1").messages, undefined);
  assert.equal(engine.getStatus("s1").receipt_path, undefined);
});

test("assemble fails open to the exact original messages", async () => {
  const messages = [{ role: "system", content: "never change" }];
  const warnings = [];
  const engine = createTestEngine({
    bridge: {
      request: async () => {
        throw new Error("bridge unavailable");
      },
      dispose: async () => {},
    },
    buildMemoryPrompt: () => "Active memory guidance",
    logger: { warn: (message) => warnings.push(message) },
  });
  const result = await engine.assemble({ sessionId: "s2", messages, tokenBudget: 1 });
  assert.strictEqual(result.messages, messages);
  assert.equal(result.promptAuthority, "preassembly_may_overflow");
  assert.equal(result.systemPromptAddition, "Active memory guidance");
  assert.match(warnings[0], /exact original context/);
});

test("assemble rejects malformed host input instead of erasing context", async () => {
  let calls = 0;
  const errors = [];
  const engine = createTestEngine({
    bridge: {
      request: async () => {
        calls += 1;
        return {};
      },
      dispose: async () => {},
    },
    logger: { error: (message) => errors.push(message) },
  });
  await assert.rejects(
    engine.assemble({ sessionId: "invalid-host", messages: undefined, tokenBudget: 100 }),
    /refused to assemble an empty prompt/,
  );
  assert.equal(calls, 0);
  assert.match(errors[0], /did not supply context messages as an array/);
  assert.equal(engine.getStatus("invalid-host").ok, false);
});

test("compact delegates to OpenClaw's recovery path", async () => {
  const calls = [];
  const engine = createTestEngine({
    bridge: { request: async () => ({}), dispose: async () => {} },
    delegateCompaction: async (params) => {
      calls.push(params);
      return {
        ok: true,
        compacted: true,
        result: { tokensBefore: 1234, tokensAfter: 400, summary: "safe summary" },
      };
    },
  });
  const params = {
    sessionId: "s-compact",
    sessionKey: "agent:main",
    currentTokenCount: 1234,
    force: true,
  };
  const result = await engine.compact(params);
  assert.equal(result.ok, true);
  assert.equal(result.compacted, true);
  assert.equal(result.result.tokensBefore, 1234);
  assert.equal(result.result.tokensAfter, 400);
  assert.deepEqual(calls, [params]);
});

test("engine declares prompt-control and compaction host requirements", () => {
  const engine = createTestEngine({
    bridge: { request: async () => ({}), dispose: async () => {} },
  });
  assert.deepEqual(
    engine.info.hostRequirements["agent-run"].requiredCapabilities,
    ["assemble-before-prompt"],
  );
  assert.deepEqual(
    engine.info.hostRequirements["manual-compact"].requiredCapabilities,
    ["compact"],
  );
});

test("assemble is provider-independent across OpenClaw routes", async () => {
  const routes = [
    ["openai", "openai/gpt-5.6-sol"],
    ["anthropic", "anthropic/claude-opus-4-6"],
    ["google", "google/gemini-2.5-pro"],
    ["nvidia", "nvidia/nemotron-3-ultra-550b-a55b"],
    ["openrouter", "openrouter/deepseek/deepseek-r1"],
    ["ollama", "ollama/qwen3:8b"],
    ["acme-private", "acme-private/future-model-v99"],
  ];
  const requests = [];
  const messages = [{ role: "user", content: "provider-neutral context" }];
  const engine = createTestEngine({
    bridge: {
      request: async (request) => {
        requests.push(request);
        return {
          schema_version: ENTROLY_BRIDGE_SCHEMA,
          ok: true,
          messages,
          estimated_tokens: 12,
          provider_independent: true,
          provider_mode: "openclaw_managed",
          budget_source: request.budget_source,
        };
      },
      dispose: async () => {},
    },
  });

  for (const [provider, resolved] of routes) {
    const result = await engine.assemble({
      sessionId: `route-${provider}`,
      messages,
      prompt: "provider-neutral context",
      runtimeSettings: {
        schemaVersion: 1,
        runtime: { host: "openclaw", mode: "normal", harnessId: "pi", runtimeId: null },
        model: { requested: "auto", resolved, provider, family: "test" },
        limits: { promptTokenBudget: 777, maxOutputTokens: 64 },
        credentials: { apiKey: "must-never-cross-the-bridge" },
      },
    });
    assert.deepEqual(result.messages, messages);
    assert.equal(result.promptAuthority, "assembled");
  }

  assert.equal(requests.length, routes.length);
  for (let index = 0; index < routes.length; index += 1) {
    const [provider, resolved] = routes[index];
    const request = requests[index];
    assert.equal(request.token_budget, 777);
    assert.equal(request.budget_source, "openclaw_runtime_settings");
    assert.equal(request.model, resolved);
    assert.equal(request.openclaw_runtime.model.provider, provider);
    assert.equal(request.openclaw_runtime.model.resolved, resolved);
    assert.equal(JSON.stringify(request).includes("must-never-cross-the-bridge"), false);
    assert.equal("credentials" in request.openclaw_runtime, false);
  }
});

test("explicit host budget wins and an operator fallback is opt-in", async () => {
  const requests = [];
  const messages = [{ role: "user", content: "hello" }];
  const engine = createTestEngine({
    bridge: {
      request: async (request) => {
        requests.push(request);
        return {
          schema_version: ENTROLY_BRIDGE_SCHEMA,
          ok: true,
          messages,
          estimated_tokens: 3,
        };
      },
      dispose: async () => {},
    },
    config: { fallbackTokenBudget: 4096 },
  });
  await engine.assemble({
    sessionId: "explicit",
    messages,
    tokenBudget: 1000,
    runtimeSettings: { limits: { promptTokenBudget: 2000 } },
  });
  await engine.assemble({ sessionId: "fallback", messages });
  assert.equal(requests[0].token_budget, 1000);
  assert.equal(requests[0].budget_source, "openclaw_token_budget");
  assert.equal(requests[1].token_budget, 4096);
  assert.equal(requests[1].budget_source, "operator_fallback");
});

test("missing host budget fails open without calling the bridge", async () => {
  const messages = [{ role: "user", content: "never guess my model window" }];
  const warnings = [];
  let calls = 0;
  const engine = createTestEngine({
    bridge: {
      request: async () => {
        calls += 1;
        return {};
      },
      dispose: async () => {},
    },
    logger: { warn: (message) => warnings.push(message) },
  });
  const result = await engine.assemble({ sessionId: "no-budget", messages });
  assert.strictEqual(result.messages, messages);
  assert.equal(result.promptAuthority, "preassembly_may_overflow");
  assert.equal(calls, 0);
  assert.match(warnings[0], /positive prompt token budget/);
});

test("invalid budgets are rejected instead of silently clamped", async () => {
  const invalidBudgets = [0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY, true, "4096"];
  const messages = [{ role: "user", content: "keep the original" }];
  let calls = 0;

  for (const tokenBudget of invalidBudgets) {
    const engine = createTestEngine({
      bridge: {
        request: async () => {
          calls += 1;
          return {};
        },
        dispose: async () => {},
      },
      config: { fallbackTokenBudget: 100 },
      logger: { warn: () => {} },
    });
    const result = await engine.assemble({
      sessionId: `invalid-${String(tokenBudget)}`,
      messages,
      tokenBudget,
    });
    assert.strictEqual(result.messages, messages);
    assert.equal(result.promptAuthority, "preassembly_may_overflow");
  }
  assert.equal(calls, 0);
});

test("malformed bridge success fails open to the exact original array", async () => {
  const messages = [
    { role: "system", content: "never modify" },
    {
      role: "assistant",
      content: [{ type: "text", text: "signed", textSignature: "opaque" }],
      provider: "openai",
    },
    { role: "user", content: "latest" },
  ];
  const valid = {
    schema_version: ENTROLY_BRIDGE_SCHEMA,
    ok: true,
    messages,
    estimated_tokens: 10,
  };
  const cases = [
    { ...valid, schema_version: "wrong" },
    { ...valid, messages: undefined },
    { ...valid, messages: messages.slice(1) },
    { ...valid, messages: [{ ...messages[0], role: "user" }, ...messages.slice(1)] },
    { ...valid, messages: [{ ...messages[0], content: "changed" }, ...messages.slice(1)] },
    {
      ...valid,
      messages: [messages[0], { ...messages[1], provider: "changed" }, messages[2]],
    },
    {
      ...valid,
      messages: [
        messages[0],
        { ...messages[1], content: [{ ...messages[1].content[0], text: "changed" }] },
        messages[2],
      ],
    },
    { ...valid, estimated_tokens: -1 },
    { ...valid, estimated_tokens: Number.NaN },
  ];

  for (const response of cases) {
    const engine = createTestEngine({
      bridge: { request: async () => response, dispose: async () => {} },
      logger: { warn: () => {} },
    });
    const result = await engine.assemble({
      sessionId: "malformed",
      messages,
      tokenBudget: 100,
    });
    assert.strictEqual(result.messages, messages);
    assert.equal(result.promptAuthority, "preassembly_may_overflow");
    assert.equal(engine.getStatus("malformed").ok, false);
  }
});

test("over-budget safe output requests native OpenClaw recovery", async () => {
  const messages = [{ role: "system", content: "protected context" }];
  const engine = createTestEngine({
    bridge: {
      request: async () => ({
        schema_version: ENTROLY_BRIDGE_SCHEMA,
        ok: true,
        messages,
        estimated_tokens: 200,
      }),
      dispose: async () => {},
    },
  });
  const result = await engine.assemble({
    sessionId: "overflow-recovery",
    messages,
    tokenBudget: 100,
  });
  assert.strictEqual(result.messages, messages);
  assert.equal(result.promptAuthority, "preassembly_may_overflow");
});

test("receipt acknowledgement failure returns the exact original context", async () => {
  const messages = [{ role: "user", content: "receipt safety" }];
  const warnings = [];
  const engine = createTestEngine({
    bridge: {
      request: async (request) =>
        request.operation === "commit_receipt"
          ? { schema_version: ENTROLY_BRIDGE_SCHEMA, ok: false }
          : {
              schema_version: ENTROLY_BRIDGE_SCHEMA,
              ok: true,
              messages,
              estimated_tokens: 4,
              receipt_id: TEST_RECEIPT_ID,
              proposal_id: TEST_PROPOSAL_ID,
              proposal_sha256: TEST_PROPOSAL_SHA256,
              receipt_path: TEST_RECEIPT_PATH,
              receipt_commit_required: true,
            },
      dispose: async () => {},
    },
    logger: { warn: (message) => warnings.push(message) },
  });
  const result = await engine.assemble({
    sessionId: "receipt-failure",
    messages,
    tokenBudget: 100,
  });
  assert.strictEqual(result.messages, messages);
  assert.equal(result.promptAuthority, "preassembly_may_overflow");
  assert.equal(engine.getStatus("receipt-failure").ok, false);
  assert.match(warnings[0], /exact original context/);
});

test("status retention is scalar-only and LRU bounded", async () => {
  const engine = createTestEngine({
    bridge: {
      request: async (request) => ({
        schema_version: ENTROLY_BRIDGE_SCHEMA,
        ok: true,
        messages: request.messages,
        estimated_tokens: 3,
        source_tokens: 10,
        warnings: ["safe warning"],
      }),
      dispose: async () => {},
    },
    maxStatusSessions: 2,
  });
  for (const sessionId of ["one", "two", "three"]) {
    await engine.assemble({
      sessionId,
      messages: [{ role: "user", content: `private-${sessionId}` }],
      tokenBudget: 100,
    });
  }
  assert.equal(engine.getStatus("one"), undefined);
  assert.equal(engine.getStatus("two").messages, undefined);
  assert.equal(JSON.stringify(engine.getStatus("two")).includes("private-two"), false);
  assert.equal(engine.getStatus("three").warnings[0], "safe warning");
});

test("a provider failure is non-sticky across same-session route changes", async () => {
  const firstMessages = [{ role: "user", content: "first" }];
  const secondMessages = [{ role: "user", content: "second" }];
  let call = 0;
  const engine = createTestEngine({
    bridge: {
      request: async (request) => {
        call += 1;
        if (call === 1) throw new Error("OpenAI route unavailable");
        return {
          schema_version: ENTROLY_BRIDGE_SCHEMA,
          ok: true,
          messages: secondMessages,
          estimated_tokens: 4,
          model: request.model,
          provider_hint: request.openclaw_runtime.model.provider,
          budget_source: request.budget_source,
        };
      },
      dispose: async () => {},
    },
    logger: { warn: () => {} },
  });
  const failed = await engine.assemble({
    sessionId: "switch",
    messages: firstMessages,
    tokenBudget: 100,
  });
  assert.strictEqual(failed.messages, firstMessages);

  const recovered = await engine.assemble({
    sessionId: "switch",
    messages: secondMessages,
    tokenBudget: 200,
    runtimeSettings: {
      model: {
        requested: "auto",
        resolved: "anthropic/claude-opus-4-6",
        provider: "anthropic",
      },
    },
  });
  assert.deepEqual(recovered.messages, secondMessages);
  assert.equal(engine.getStatus("switch").ok, true);
  assert.equal(engine.getStatus("switch").provider_hint, "anthropic");
  assert.equal(engine.getStatus("switch").model, "anthropic/claude-opus-4-6");
});

test("assemble preserves active OpenClaw memory guidance", async () => {
  const messages = [{ role: "user", content: "remember this" }];
  const calls = [];
  const engine = createTestEngine({
    bridge: {
      request: async () => ({
        schema_version: ENTROLY_BRIDGE_SCHEMA,
        ok: true,
        messages,
        estimated_tokens: 3,
      }),
      dispose: async () => {},
    },
    buildMemoryPrompt: (params) => {
      calls.push(params);
      return "Active memory guidance";
    },
  });
  const availableTools = new Set(["memory_search"]);
  const result = await engine.assemble({
    sessionId: "memory",
    sessionKey: "agent:main:memory",
    messages,
    tokenBudget: 100,
    availableTools,
    citationsMode: "on",
  });
  assert.equal(result.systemPromptAddition, "Active memory guidance");
  assert.deepEqual(calls, [
    {
      availableTools,
      citationsMode: "on",
      agentSessionKey: "agent:main:memory",
    },
  ]);
});

test("memory guidance failures are visible without breaking context assembly", async () => {
  const messages = [{ role: "user", content: "continue safely" }];
  const warnings = [];
  const engine = createTestEngine({
    bridge: {
      request: async () => ({
        schema_version: ENTROLY_BRIDGE_SCHEMA,
        ok: true,
        messages,
        estimated_tokens: 3,
      }),
      dispose: async () => {},
    },
    buildMemoryPrompt: () => {
      throw new Error("memory helper failed apiKey=sk-super-secret-token");
    },
    logger: { warn: (message) => warnings.push(message) },
  });
  const result = await engine.assemble({
    sessionId: "memory-failure",
    messages,
    tokenBudget: 100,
  });
  assert.equal(result.promptAuthority, "assembled");
  assert.equal(result.systemPromptAddition, undefined);
  assert.match(warnings[0], /memory guidance could not be added/);
  assert.match(warnings[0], /\[REDACTED\]/);
  assert.doesNotMatch(warnings[0], /super-secret/);
});

test("status output labels estimates and exposes the receipt", () => {
  const output = formatEntrolyStatus({
    ok: true,
    source_tokens: 1000,
    estimated_tokens: 400,
    tokens_saved: 600,
    changed: true,
    assembly_strategy: "query_aware_evidence_pinning",
    evidence_pinned: 2,
    evidence_pin_blocked: 1,
    receipt_id: "ocr_test",
    warnings: ["Token counts are estimates."],
  });
  assert.match(output, /Estimated tokens: 1,000 -> 400/);
  assert.match(output, /Evidence pinned verbatim: 2 message/);
  assert.match(output, /Evidence pins blocked by firewall: 1/);
  assert.match(output, /Estimated reduction: 60.0%/);
  assert.match(output, /Receipt:/);
  assert.doesNotMatch(output, /\/workspace/);
  assert.match(output, /Warnings:/);
});

test("doctor output is actionable without exposing multiline errors", () => {
  const ready = formatEntrolyDoctor({ ok: true, pythonCommand: "python3" });
  assert.match(ready, /doctor: ready/);
  assert.match(ready, /python3/);

  const failed = formatEntrolyDoctor({
    ok: false,
    pythonCommand: "missing-python",
    error: new Error("spawn failed\nwith details"),
  });
  assert.match(failed, /doctor: not ready/);
  assert.match(failed, /pip install -U entroly/);
  assert.doesNotMatch(failed, /failed\nwith/);
});

test("status and doctor diagnostics are bounded and credential-redacted", () => {
  const secret = "sk-1234567890secret";
  const status = formatEntrolyStatus({
    ok: false,
    error: `request failed Authorization: Bearer ${secret} apiKey=${secret} ${"x".repeat(600)}`,
  });
  const doctor = formatEntrolyDoctor({
    ok: false,
    error: new Error(`spawn failed token=${secret}`),
  });
  assert.match(status, /\[REDACTED\]/);
  assert.doesNotMatch(status, new RegExp(secret));
  assert.ok(status.length < 600);
  assert.match(doctor, /\[REDACTED\]/);
  assert.doesNotMatch(doctor, new RegExp(secret));
});
