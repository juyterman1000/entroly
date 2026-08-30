import assert from "node:assert/strict";
import test from "node:test";

import { createProofGuidedHooks } from "../proof-hooks.js";

const sourceMessages = [
  { role: "user", content: "Entroly retains exact evidence." },
  {
    role: "assistant",
    content: "Restart recovery replays durable events in their original order.",
  },
];

function proofResult(overrides = {}) {
  return {
    schema_version: "entroly.openclaw.bridge.v2",
    ok: true,
    status: "retry_with_exact_evidence",
    verified_output: "Entroly withheld unsupported claims.",
    changed: true,
    recovered_messages: [sourceMessages[1]],
    retry_instruction: "Revise using exact recovered evidence.",
    audit_artifact_id: "vea_123",
    provider_call_performed: false,
    local_only: true,
    ...overrides,
  };
}

test("proof hooks request one bounded revision and suppress unsupported delivery", async () => {
  const requests = [];
  const bridge = {
    async request(payload) {
      requests.push(payload);
      return proofResult();
    },
  };
  const proofStateBySession = new Map([
    [
      "session-1",
      {
        prompt: "question",
        workspaceDir: "/workspace",
        sourceMessages,
        assembledMessages: [sourceMessages[0], { role: "assistant", content: "Restart." }],
        recoveredMessages: [],
        attempts: 0,
        disabled: false,
      },
    ],
  ]);
  const statusBySession = new Map();
  const hooks = createProofGuidedHooks({
    bridge,
    config: { proofGuidedMaxRounds: 2 },
    proofStateBySession,
    statusBySession,
  });

  await hooks.onLlmOutput({
    runId: "run-1",
    sessionId: "session-1",
    assistantTexts: ["Unsupported first draft."],
  });
  assert.equal(requests.length, 1);
  assert.equal(requests[0].operation, "verify_proof_guided_output");
  assert.equal(requests[0].round_index, 0);
  assert.equal(requests[0].provider, undefined);
  assert.equal(proofStateBySession.get("session-1").attempts, 1);
  assert.deepEqual(
    proofStateBySession.get("session-1").recoveredMessages,
    [sourceMessages[1]],
  );

  const revision = await hooks.onBeforeAgentFinalize({
    runId: "run-1",
    sessionId: "session-1",
  });
  assert.equal(revision.action, "revise");
  assert.equal(revision.retry.maxAttempts, 1);
  assert.match(revision.retry.instruction, /exact recovered evidence/i);
  assert.equal(
    await hooks.onBeforeAgentFinalize({ runId: "run-1", sessionId: "session-1" }),
    undefined,
  );

  const delivery = await hooks.onReplyPayloadSending({
    runId: "run-1",
    usageState: { sessionId: "session-1" },
    payload: { text: "Unsupported first draft.", other: true },
  });
  assert.equal(delivery.payload.text, "Entroly withheld unsupported claims.");
  assert.equal(delivery.payload.other, true);
  assert.equal(statusBySession.get("session-1").proof_guided_attempts, 1);
});

test("supported revision finalizes without another paid attempt", async () => {
  const bridge = {
    async request() {
      return proofResult({
        status: "supported",
        verified_output: "Supported answer.",
        changed: false,
        recovered_messages: [],
        retry_instruction: null,
        audit_artifact_id: "vea_456",
      });
    },
  };
  const proofStateBySession = new Map([
    [
      "session-2",
      {
        sourceMessages,
        assembledMessages: sourceMessages,
        recoveredMessages: [],
        attempts: 1,
        disabled: false,
      },
    ],
  ]);
  const hooks = createProofGuidedHooks({
    bridge,
    config: { proofGuidedMaxRounds: 2 },
    proofStateBySession,
  });
  await hooks.onLlmOutput({
    runId: "run-2",
    sessionId: "session-2",
    assistantTexts: ["Supported answer."],
  });
  assert.equal(proofStateBySession.get("session-2").attempts, 2);
  assert.equal(
    await hooks.onBeforeAgentFinalize({ runId: "run-2", sessionId: "session-2" }),
    undefined,
  );
});

test("invalid bridge proof disables retries instead of looping", async () => {
  const warnings = [];
  const proofStateBySession = new Map([
    [
      "session-3",
      {
        sourceMessages,
        assembledMessages: sourceMessages,
        recoveredMessages: [],
        attempts: 0,
        disabled: false,
      },
    ],
  ]);
  const hooks = createProofGuidedHooks({
    bridge: { request: async () => ({ ok: true }) },
    logger: { warn: (message) => warnings.push(message) },
    proofStateBySession,
  });
  await hooks.onLlmOutput({
    runId: "run-3",
    sessionId: "session-3",
    assistantTexts: ["answer"],
  });
  assert.equal(proofStateBySession.get("session-3").disabled, true);
  assert.equal(warnings.length, 1);
  assert.equal(
    await hooks.onBeforeAgentFinalize({ runId: "run-3", sessionId: "session-3" }),
    undefined,
  );
  const delivery = await hooks.onReplyPayloadSending({
    runId: "run-3",
    usageState: { sessionId: "session-3" },
    payload: { text: "unverified answer" },
  });
  assert.match(delivery.payload.text, /withheld.*verification failed/i);
});

// ── before_tool_call ────────────────────────────────────────────────────
//
// A tool call is where an unsupported claim stops being text and becomes an
// action. OpenClaw gives this hook a 15-second budget and fails CLOSED on
// expiry, so the handler must never do work that can hang: everything it
// needs was already computed at `llm_output`.

function gateHooks(config = {}, state = undefined) {
  const proofStateBySession = new Map();
  if (state) proofStateBySession.set("s1", state);
  return createProofGuidedHooks({
    bridge: {
      request() {
        throw new Error("before_tool_call must not call the bridge");
      },
    },
    config: { proofGuidedRecovery: true, ...config },
    logger: { error() {}, warn() {}, info() {} },
    proofStateBySession,
    statusBySession: new Map(),
  });
}

const unsupported = () => ({ runId: "r1", lastProofResult: proofResult() });

test("before_tool_call asks for approval when the claim behind it was unsupported", () => {
  const hooks = gateHooks({}, unsupported());
  const decision = hooks.onBeforeToolCall({
    sessionId: "s1",
    runId: "r1",
    toolName: "exec",
  });

  assert.equal(decision?.block, undefined, "default must not block outright");
  assert.ok(decision?.requireApproval, "the operator decides, not the plugin");
  assert.match(decision.requireApproval.title, /exec/);
  assert.deepEqual(decision.requireApproval.allowedDecisions, [
    "allow-once",
    "deny",
  ]);
});

test("before_tool_call blocks when explicitly configured to", () => {
  const decision = gateHooks({ gateToolCalls: "block" }, unsupported())
    .onBeforeToolCall({ sessionId: "s1", runId: "r1", toolName: "exec" });

  assert.equal(decision.block, true);
  assert.match(decision.blockReason, /did not support/);
});

test("before_tool_call is inert when the gate is off", () => {
  const decision = gateHooks({ gateToolCalls: "off" }, unsupported())
    .onBeforeToolCall({ sessionId: "s1", runId: "r1" });

  assert.equal(decision, undefined);
});

test("an unknown gate value asks rather than acts", () => {
  const decision = gateHooks({ gateToolCalls: "nonsense" }, unsupported())
    .onBeforeToolCall({ sessionId: "s1", runId: "r1" });

  assert.ok(decision?.requireApproval, "unrecognised config must fail safe");
});

test("a supported verdict allows the tool call", () => {
  const decision = gateHooks({}, {
    runId: "r1",
    lastProofResult: proofResult({ status: "supported", changed: false }),
  }).onBeforeToolCall({ sessionId: "s1", runId: "r1" });

  assert.equal(decision, undefined);
});

test("a verdict from an earlier run does not gate this one", () => {
  const decision = gateHooks({}, unsupported())
    .onBeforeToolCall({ sessionId: "s1", runId: "r2" });

  assert.equal(decision, undefined, "a stale verdict says nothing about this call");
});

test("no verdict, no session, and disabled verification all allow", () => {
  assert.equal(
    gateHooks({}, undefined).onBeforeToolCall({ sessionId: "s1", runId: "r1" }),
    undefined,
    "an unknown session must not block the user",
  );
  assert.equal(
    gateHooks({}, { runId: "r1" }).onBeforeToolCall({ sessionId: "s1", runId: "r1" }),
    undefined,
    "no verdict yet must not block the user",
  );
  assert.equal(
    gateHooks({}, { runId: "r1", disabled: true, lastProofResult: proofResult() })
      .onBeforeToolCall({ sessionId: "s1", runId: "r1" }),
    undefined,
    "verification that failed must not become a tool-call outage",
  );
});

test("before_tool_call never calls the bridge", () => {
  // The bridge in gateHooks throws on use. A 15-second fail-closed budget
  // means a round trip here converts a slow bridge into blocked tool calls.
  assert.doesNotThrow(() =>
    gateHooks({}, unsupported()).onBeforeToolCall({ sessionId: "s1", runId: "r1" }),
  );
});

test("before_tool_call is synchronous", () => {
  const decision = gateHooks({}, unsupported())
    .onBeforeToolCall({ sessionId: "s1", runId: "r1" });

  assert.ok(
    typeof decision?.then !== "function",
    "returning a promise would put bridge latency inside a fail-closed budget",
  );
});

test("a malformed event allows rather than throws", () => {
  const hooks = gateHooks({}, unsupported());
  for (const event of [undefined, null, {}, { sessionId: null }]) {
    assert.equal(hooks.onBeforeToolCall(event), undefined);
  }
});
