import assert from "node:assert/strict";
import test from "node:test";

import { registerCommunicationTools } from "../communication-tools.js";

function registerFixture({
  config = {},
  bridgeHandler,
} = {}) {
  const registrations = new Map();
  const api = {
    registerTool(factory, options) {
      registrations.set(options?.name, { factory, options });
    },
  };
  const bridge = {
    requests: [],
    async request(payload) {
      this.requests.push(payload);
      if (bridgeHandler) return await bridgeHandler(payload, this.requests);
      if (payload.operation === "communication_digest") {
        return { ok: true, digest: { total_events: 1, attention_count: 1 } };
      }
      if (payload.operation === "communication_resolve_taste") {
        return { ok: true, resolved: { response_length: "short" } };
      }
      if (payload.operation === "communication_assure") {
        return {
          ok: true,
          decision: "approval_required",
          action_id: "ca-1",
          execution_state: "awaiting_approval",
        };
      }
      throw new Error(`unexpected operation: ${payload.operation}`);
    },
  };
  const effectiveConfig = {
    communicationSecretaryTools: true,
    ...config,
  };
  registerCommunicationTools(api, { bridge, config: effectiveConfig });
  return { registrations, bridge };
}

function materialize(registrations, name, context) {
  const entry = registrations.get(name);
  assert.ok(entry, `missing registration ${name}`);
  assert.equal(entry.options.optional, true);
  assert.equal(entry.factory.contextVersion, 2);
  return entry.factory.create(context);
}

function ownerContext(overrides = {}) {
  let currentChecks = 0;
  const ctx = {
    senderIsOwner: true,
    messageChannel: "whatsapp",
    nativeChannelId: "chat-123",
    requesterSenderId: "person-7",
    agentId: "main",
    deliveryContext: { channel: "whatsapp", accountId: "personal", to: "chat-123" },
    assertInvocationCurrent() {
      currentChecks += 1;
    },
    ...overrides,
  };
  Object.defineProperty(ctx, "currentChecks", {
    get: () => currentChecks,
  });
  return ctx;
}

test("secretary read and assurance tools are absent until explicitly enabled", () => {
  const { registrations } = registerFixture({
    config: { communicationSecretaryTools: false },
  });
  const ctx = ownerContext();

  for (const name of [
    "entroly_communication_brief",
    "entroly_communication_assure",
    "entroly_communication_taste",
  ]) {
    assert.equal(materialize(registrations, name, ctx), null);
  }
});

test("communication tools disappear for non-owner turns", () => {
  const { registrations } = registerFixture();

  for (const name of [
    "entroly_communication_brief",
    "entroly_communication_assure",
    "entroly_communication_taste",
  ]) {
    assert.equal(
      materialize(registrations, name, {
        senderIsOwner: false,
        assertInvocationCurrent() {},
      }),
      null,
    );
  }
});

test("brief binds current scope from OpenClaw rather than model arguments", async () => {
  const { registrations, bridge } = registerFixture();
  const ctx = ownerContext();
  const tool = materialize(registrations, "entroly_communication_brief", ctx);

  const result = await tool.execute("call-1", { scope: "current", limit: 25 });

  assert.equal(bridge.requests[0].operation, "communication_digest");
  assert.equal(bridge.requests[0].channel, "whatsapp");
  assert.equal(bridge.requests[0].account_id, "personal");
  assert.equal(bridge.requests[0].conversation_id, "chat-123");
  assert.equal(bridge.requests[0].owner_authorized, true);
  assert.equal(bridge.requests[1].operation, "communication_resolve_taste");
  assert.deepEqual(result.details.resolved_taste, { response_length: "short" });
  assert.ok(ctx.currentChecks >= 1);
});

test("global brief is denied unless cross-conversation access is explicitly enabled", async () => {
  const { registrations, bridge } = registerFixture({
    config: {
      communicationSecretaryTools: true,
      communicationGlobalAccess: false,
    },
  });
  const tool = materialize(
    registrations,
    "entroly_communication_brief",
    ownerContext(),
  );

  await assert.rejects(
    () => tool.execute("call-global-denied", { scope: "all" }),
    /communicationGlobalAccess=true/,
  );
  assert.equal(bridge.requests.length, 0);
});

test("global brief is owner-bound and opt-in", async () => {
  const { registrations, bridge } = registerFixture({
    config: {
      communicationSecretaryTools: true,
      communicationGlobalAccess: true,
    },
  });
  const tool = materialize(
    registrations,
    "entroly_communication_brief",
    ownerContext(),
  );

  await tool.execute("call-global", { scope: "all", limit: 50 });

  assert.equal(bridge.requests[0].operation, "communication_digest");
  assert.equal(bridge.requests[0].owner_authorized, true);
  assert.equal("conversation_id" in bridge.requests[0], false);
});

test("assurance cannot target a different conversation", async () => {
  const { registrations, bridge } = registerFixture();
  const ctx = ownerContext();
  const tool = materialize(registrations, "entroly_communication_assure", ctx);

  await tool.execute("call-2", {
    action_type: "reply",
    source_event_ids: ["comm-1"],
    text: "Thanks",
    conversation_id: "attacker-supplied",
  });

  const request = bridge.requests[0];
  assert.equal(request.conversation_id, "chat-123");
  assert.equal(request.channel, "whatsapp");
  assert.deepEqual(request.source_event_ids, ["comm-1"]);
  assert.equal("to" in request, false);
});

test("execution tool is absent unless execution is explicitly enabled", () => {
  const { registrations } = registerFixture({
    config: { communicationExecution: false },
  });
  const tool = materialize(
    registrations,
    "entroly_communication_execute",
    ownerContext({ delivery: { send: async () => {} } }),
  );
  assert.equal(tool, null);
});

test("execution refuses non-ALLOW assurance without sending", async () => {
  let sends = 0;
  const { registrations } = registerFixture({
    config: {
      communicationExecution: true,
      communicationPolicyMode: "bounded",
      communicationAutoActions: ["reply"],
      communicationAutoCategories: ["birthday_wish"],
    },
  });
  const ctx = ownerContext({
    delivery: { send: async () => { sends += 1; } },
  });
  const tool = materialize(registrations, "entroly_communication_execute", ctx);

  const result = await tool.execute("call-3", {
    action_type: "reply",
    source_event_ids: ["comm-1"],
    text: "Thank you!",
  });

  assert.equal(result.details.dispatched, false);
  assert.equal(sends, 0);
});

test("bounded execution atomically claims dispatch and sends once", async () => {
  let sends = 0;
  const { registrations, bridge } = registerFixture({
    config: {
      communicationExecution: true,
      communicationPolicyMode: "bounded",
      communicationAutoActions: ["reply"],
      communicationAutoCategories: ["birthday_wish"],
    },
    bridgeHandler: async (payload) => {
      if (payload.operation === "communication_assure") {
        return {
          ok: true,
          decision: "allow",
          action_id: "ca-birthday",
          execution_state: "assured",
        };
      }
      if (payload.operation === "communication_begin_action") {
        return {
          ok: true,
          action_id: payload.action_id,
          claimed: true,
          execution_state: "dispatching",
        };
      }
      throw new Error(`unexpected operation: ${payload.operation}`);
    },
  });
  const ctx = ownerContext({
    delivery: { send: async ({ text }) => {
      assert.equal(text, "Thank you so much!");
      sends += 1;
    } },
  });
  const tool = materialize(registrations, "entroly_communication_execute", ctx);

  const result = await tool.execute("call-4", {
    action_type: "reply",
    source_event_ids: ["comm-1"],
    text: "Thank you so much!",
  });

  assert.equal(result.details.dispatched, true);
  assert.equal(result.details.execution_state, "dispatching");
  assert.equal(sends, 1);
  assert.deepEqual(
    bridge.requests.map((request) => request.operation),
    ["communication_assure", "communication_begin_action"],
  );
  assert.ok(ctx.currentChecks >= 2);
});

test("duplicate dispatch claim prevents a second send", async () => {
  let sends = 0;
  const { registrations } = registerFixture({
    config: {
      communicationExecution: true,
      communicationPolicyMode: "bounded",
      communicationAutoActions: ["reply"],
      communicationAutoCategories: ["birthday_wish"],
    },
    bridgeHandler: async (payload) => {
      if (payload.operation === "communication_assure") {
        return { decision: "allow", action_id: "same-action", execution_state: "dispatching" };
      }
      if (payload.operation === "communication_begin_action") {
        return { claimed: false, execution_state: "dispatching" };
      }
      throw new Error("unexpected operation");
    },
  });
  const tool = materialize(
    registrations,
    "entroly_communication_execute",
    ownerContext({ delivery: { send: async () => { sends += 1; } } }),
  );

  const result = await tool.execute("call-5", {
    action_type: "reply",
    source_event_ids: ["comm-1"],
    text: "Thanks",
  });

  assert.equal(result.details.dispatched, false);
  assert.equal(result.details.dispatch_claimed, false);
  assert.equal(sends, 0);
});

test("host delivery exception is persisted as failed dispatch", async () => {
  const { registrations, bridge } = registerFixture({
    config: {
      communicationExecution: true,
      communicationPolicyMode: "bounded",
      communicationAutoActions: ["reply"],
      communicationAutoCategories: ["birthday_wish"],
    },
    bridgeHandler: async (payload) => {
      if (payload.operation === "communication_assure") {
        return { decision: "allow", action_id: "ca-fail", execution_state: "assured" };
      }
      if (payload.operation === "communication_begin_action") {
        return { claimed: true, execution_state: "dispatching" };
      }
      if (payload.operation === "communication_fail_action") {
        return { recorded: true, execution_state: "failed" };
      }
      throw new Error("unexpected operation");
    },
  });
  const tool = materialize(
    registrations,
    "entroly_communication_execute",
    ownerContext({
      delivery: { send: async () => { throw new Error("network down"); } },
    }),
  );

  await assert.rejects(
    () =>
      tool.execute("call-6", {
        action_type: "reply",
        source_event_ids: ["comm-1"],
        text: "Thanks",
      }),
    /network down/,
  );

  assert.deepEqual(
    bridge.requests.map((request) => request.operation),
    ["communication_assure", "communication_begin_action", "communication_fail_action"],
  );
});
