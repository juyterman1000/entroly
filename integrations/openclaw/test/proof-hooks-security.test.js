import assert from "node:assert/strict";
import test from "node:test";

import { createProofGuidedHooks } from "../proof-hooks.js";

function state() {
  return {
    sourceMessages: [],
    assembledMessages: [],
    recoveredMessages: [],
    attempts: 0,
    disabled: false,
  };
}

test("diagnostic normalization cannot be used to bypass credential redaction", async () => {
  const proofState = state();
  const proofStateBySession = new Map([["session-obfuscated-secret", proofState]]);
  const warnings = [];
  const hooks = createProofGuidedHooks({
    bridge: {
      async request() {
        throw new Error(
          "bridge failed Ｂｅａｒｅｒ odd:token%value; Ｂａｓｉｃ basic:credential%value; " +
            "ｔｏｋｅｎ＝compat-secret; Be\u200barer abc\u001b.def; to\u200bken=hunter2; " +
            "authorization=Basic c2VjcmV0; password=\"two words\"; " +
            "\"api_key\": \"quoted secret\"",
        );
      },
    },
    logger: { warn: (message) => warnings.push(message) },
    proofStateBySession,
  });

  await hooks.onLlmOutput({
    runId: "run-obfuscated-secret",
    sessionId: "session-obfuscated-secret",
    assistantTexts: ["answer"],
  });

  assert.equal(proofState.disabled, true);
  assert.equal(warnings.length, 1);
  for (const leakedSecret of [
    "odd:token%value",
    "basic:credential%value",
    "compat-secret",
    "abc.def",
    "hunter2",
    "c2VjcmV0",
    "two words",
    "quoted secret",
  ]) {
    assert.equal(proofState.error.includes(leakedSecret), false);
  }
  assert.match(proofState.error, /Bearer \\?\[REDACTED\\?\]/);
  assert.match(proofState.error, /Basic \\?\[REDACTED\\?\]/);
  assert.match(proofState.error, /token=\\?\[REDACTED\\?\]/);
  assert.match(proofState.error, /authorization=\\?\[REDACTED\\?\]/);
  assert.match(proofState.error, /password=\\?\[REDACTED\\?\]/);
  assert.match(proofState.error, /api\\?_key=\\?\[REDACTED\\?\]/);
});

test("diagnostic coercion failure cannot escape the fail-closed verification path", async () => {
  const proofState = state();
  const proofStateBySession = new Map([["session-hostile-object", proofState]]);
  const warnings = [];
  const hostile = {};
  Object.defineProperty(hostile, "message", {
    get() {
      throw new Error("getter must not escape");
    },
  });

  const hooks = createProofGuidedHooks({
    bridge: {
      async request() {
        throw hostile;
      },
    },
    logger: { warn: (message) => warnings.push(message) },
    proofStateBySession,
  });

  await assert.doesNotReject(() =>
    hooks.onLlmOutput({
      runId: "run-hostile-object",
      sessionId: "session-hostile-object",
      assistantTexts: ["answer"],
    }),
  );

  assert.equal(proofState.disabled, true);
  assert.equal(proofState.error, "unknown error");
  assert.equal(warnings.length, 1);
});
