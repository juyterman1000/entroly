import assert from "node:assert/strict";
import test from "node:test";

import {
  createCommunicationHooks,
  formatCommunicationStatus,
} from "../communication-hooks.js";

function fixture() {
  const requests = [];
  const warnings = [];
  const bridge = {
    request(payload) {
      requests.push(payload);
      return Promise.resolve({
        ok: true,
        stats: {
          events: 2,
          inbound: 1,
          outbound: 1,
          conversations: 1,
          retention_days: 90,
        },
      });
    },
  };
  const hooks = createCommunicationHooks({
    bridge,
    config: {
      communicationStorePath: "/private/communication.sqlite3",
      communicationRetentionDays: 90,
    },
    logger: { warn: (message) => warnings.push(message) },
  });
  return { hooks, requests, warnings };
}

test("message_received maps only structured scope and preserves unknown chat kind", () => {
  const { hooks, requests } = fixture();

  hooks.onMessageReceived(
    {
      content: "hello",
      senderId: "person-a",
      messageId: "m-1",
      timestamp: 1_700_000_000,
      metadata: {
        senderName: "Person A",
        senderUsername: "a",
        originatingChannel: "whatsapp",
      },
    },
    {
      channelId: "whatsapp",
      accountId: "personal",
      conversationId: "conversation-a",
      sessionKey: "opaque-session-key",
    },
  );

  assert.equal(requests.length, 1);
  const request = requests[0];
  assert.equal(request.operation, "communication_ingest");
  assert.equal(request.event.direction, "inbound");
  assert.equal(request.event.channel, "whatsapp");
  assert.equal(request.event.conversation_id, "conversation-a");
  assert.equal(request.event.conversation_kind, "unknown");
  assert.equal(request.event.sender_id, "person-a");
  assert.equal(request.event.message_id, "m-1");
  assert.equal(request.event.content, "hello");
  assert.equal(request.event.metadata.sender_name, "Person A");
  assert.equal(request.store_path, "/private/communication.sqlite3");
  assert.equal(request.retention_days, 90);
});

test("future structured isGroup can be honored without parsing provider ids", () => {
  const { hooks, requests } = fixture();

  hooks.onMessageReceived(
    { content: "hello group", messageId: "g-1", isGroup: true },
    { channelId: "whatsapp", conversationId: "group-ref" },
  );

  assert.equal(requests[0].event.conversation_kind, "group");
});

test("unscoped inbound observations fail closed instead of guessing conversation", () => {
  const { hooks, requests, warnings } = fixture();

  hooks.onMessageReceived(
    { content: "private", from: "opaque-whatsapp-id" },
    { channelId: "whatsapp" },
  );

  assert.equal(requests.length, 0);
  assert.equal(warnings.length, 1);
  assert.match(warnings[0], /unscoped inbound/);
});

test("message_sent records observable delivery outcome without marking source handled", () => {
  const { hooks, requests } = fixture();

  hooks.onMessageSent(
    {
      to: "person-a",
      content: "Thanks",
      success: true,
      messageId: "out-1",
    },
    {
      channelId: "whatsapp",
      accountId: "personal",
      conversationId: "conversation-a",
    },
  );

  const event = requests[0].event;
  assert.equal(event.direction, "outbound");
  assert.equal(event.recipient_id, "person-a");
  assert.equal(event.delivery_state, "sent");
  assert.equal(event.message_id, "out-1");
  assert.equal(event.content, "Thanks");
});

test("failed outbound observation remains failed evidence", () => {
  const { hooks, requests } = fixture();

  hooks.onMessageSent(
    {
      to: "person-a",
      content: "Thanks",
      success: false,
      error: "network error",
    },
    {
      channelId: "whatsapp",
      conversationId: "conversation-a",
    },
  );

  assert.equal(requests[0].event.delivery_state, "failed");
  assert.equal(requests[0].event.metadata.delivery_error, "network error");
});

test("communication status is scalar and explains explicit opt in", async () => {
  const { hooks } = fixture();
  const result = await hooks.status();
  const enabled = formatCommunicationStatus({ enabled: true, result });
  const disabled = formatCommunicationStatus({ enabled: false });

  assert.match(enabled, /Events: 2/);
  assert.match(enabled, /Conversation scopes: 1/);
  assert.match(disabled, /disabled/);
  assert.match(disabled, /messageReceived/);
});


test("100 direct birthday observations map to 100 isolated bridge events", () => {
  const { hooks, requests } = fixture();

  for (let index = 0; index < 100; index += 1) {
    hooks.onMessageReceived(
      {
        content: "Happy birthday!",
        senderId: `person-${String(index).padStart(3, "0")}`,
        messageId: `birthday-${String(index).padStart(3, "0")}`,
      },
      {
        channelId: "whatsapp",
        accountId: "personal",
        conversationId: `dm-${String(index).padStart(3, "0")}`,
      },
    );
  }

  assert.equal(requests.length, 100);
  assert.equal(new Set(requests.map((item) => item.event.conversation_id)).size, 100);
  assert.equal(new Set(requests.map((item) => item.event.message_id)).size, 100);
  assert.ok(requests.every((item) => item.event.conversation_kind === "unknown"));
});

test("100 verified group birthday observations remain in one group scope", () => {
  const { hooks, requests } = fixture();

  for (let index = 0; index < 100; index += 1) {
    hooks.onMessageReceived(
      {
        content: "Happy birthday!",
        senderId: `member-${String(index).padStart(3, "0")}`,
        messageId: `group-birthday-${String(index).padStart(3, "0")}`,
        isGroup: true,
      },
      {
        channelId: "whatsapp",
        accountId: "personal",
        conversationId: "family-birthday-group",
      },
    );
  }

  assert.equal(requests.length, 100);
  assert.ok(
    requests.every(
      (item) =>
        item.event.conversation_id === "family-birthday-group" &&
        item.event.conversation_kind === "group",
    ),
  );
  assert.equal(new Set(requests.map((item) => item.event.sender_id)).size, 100);
});
