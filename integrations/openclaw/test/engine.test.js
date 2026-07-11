import assert from "node:assert/strict";
import test from "node:test";

import {
  createEntrolyContextEngine,
  formatEntrolyDoctor,
  formatEntrolyStatus,
} from "../engine.js";

test("assemble delegates to Entroly and reports assembled authority", async () => {
  const messages = [{ role: "user", content: "hello" }];
  const bridge = {
    request: async (request) => ({
      ok: true,
      messages,
      estimated_tokens: 9,
      receipt_id: "ocr_test",
      request,
    }),
    dispose: async () => {},
  };
  const engine = createEntrolyContextEngine({
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
  assert.equal(engine.getStatus("s1").receipt_id, "ocr_test");
  assert.equal(engine.getStatus("s1").request.workspace_dir, "/workspace");
});

test("assemble fails open to the exact original messages", async () => {
  const messages = [{ role: "system", content: "never change" }];
  const warnings = [];
  const engine = createEntrolyContextEngine({
    bridge: {
      request: async () => {
        throw new Error("bridge unavailable");
      },
      dispose: async () => {},
    },
    logger: { warn: (message) => warnings.push(message) },
  });
  const result = await engine.assemble({ sessionId: "s2", messages, tokenBudget: 1 });
  assert.strictEqual(result.messages, messages);
  assert.equal(result.promptAuthority, "preassembly_may_overflow");
  assert.match(warnings[0], /exact original context/);
});

test("compact is explicit and never claims transcript mutation", async () => {
  const engine = createEntrolyContextEngine({
    bridge: { request: async () => ({}), dispose: async () => {} },
  });
  const result = await engine.compact({ currentTokenCount: 1234 });
  assert.equal(result.ok, true);
  assert.equal(result.compacted, false);
  assert.equal(result.result.tokensBefore, 1234);
  assert.match(result.reason, /does not rewrite/);
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
