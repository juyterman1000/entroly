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
