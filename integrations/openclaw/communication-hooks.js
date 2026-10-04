function boundedString(value, limit = 1024) {
  return typeof value === "string" && value.trim()
    ? value.trim().slice(0, limit)
    : undefined;
}

function safeDiagnostic(value, limit = 300) {
  return String(value?.message ?? value ?? "unknown error")
    .replace(/\s+/g, " ")
    .replace(/\bBearer\s+[A-Za-z0-9._~+/=-]+/gi, "Bearer [REDACTED]")
    .replace(
      /\b(api[_-]?key|authorization|password|secret|token)\b\s*[:=]\s*["']?[^\s,;"']+/gi,
      "$1=[REDACTED]",
    )
    .slice(0, limit);
}

function conversationKind(event) {
  // Current OpenClaw message_received does not publish isGroup. Preserve
  // "unknown" rather than parsing WhatsApp JIDs/session keys. This will
  // automatically become useful if a future host publishes the structured bit.
  if (event?.isGroup === true) return "group";
  if (event?.isGroup === false) return "direct";
  return "unknown";
}

function communicationMetadata(event = {}) {
  const raw = event.metadata && typeof event.metadata === "object" ? event.metadata : {};
  const metadata = {};
  for (const [source, target, limit] of [
    ["senderName", "sender_name", 256],
    ["senderUsername", "sender_username", 256],
    ["threadId", "thread_id", 1024],
    ["originatingChannel", "originating_channel", 128],
    ["originatingTo", "originating_to", 1024],
    ["provider", "provider", 128],
    ["surface", "surface", 128],
  ]) {
    const value = boundedString(raw[source], limit);
    if (value !== undefined) metadata[target] = value;
  }
  if (typeof event.replyToIsQuote === "boolean") {
    metadata.reply_to_is_quote = event.replyToIsQuote;
  }
  return metadata;
}

function scopedContext(ctx = {}) {
  const channel = boundedString(ctx.channelId, 128);
  const conversation = boundedString(ctx.conversationId, 1024);
  if (!channel || !conversation) return undefined;
  return {
    channel: channel.toLowerCase(),
    account_id: boundedString(ctx.accountId, 256) ?? "",
    conversation_id: conversation,
  };
}

function baseRequest(config) {
  return {
    store_path: boundedString(config.communicationStorePath, 4096),
    retention_days:
      Number.isInteger(config.communicationRetentionDays) &&
      config.communicationRetentionDays >= 0
        ? config.communicationRetentionDays
        : undefined,
  };
}

export function createCommunicationHooks({
  bridge,
  config = {},
  logger = console,
}) {
  const requestBase = baseRequest(config);
  const submit = (event) => {
    // OpenClaw's inbound observer has its own tight timeout. Do not make the
    // channel delivery path wait on SQLite/Python startup.
    void bridge
      .request({
        operation: "communication_ingest",
        ...requestBase,
        event,
      })
      .catch((error) => {
        logger.warn?.(
          `entroly: communication observation failed safely: ${safeDiagnostic(error)}`,
        );
      });
  };

  return {
    onMessageReceived(event, ctx) {
      const scope = scopedContext(ctx);
      if (!scope) {
        logger.warn?.(
          "entroly: skipped unscoped inbound communication observation",
        );
        return;
      }
      submit({
        direction: "inbound",
        ...scope,
        conversation_kind: conversationKind(event),
        sender_id:
          boundedString(event?.senderId, 1024) ??
          boundedString(ctx?.senderId, 1024) ??
          "",
        message_id:
          boundedString(event?.messageId, 1024) ??
          boundedString(ctx?.messageId, 1024) ??
          "",
        reply_to_id:
          boundedString(event?.replyToId, 1024) ??
          boundedString(ctx?.replyToId, 1024) ??
          "",
        session_key:
          boundedString(event?.sessionKey, 2048) ??
          boundedString(ctx?.sessionKey, 2048) ??
          "",
        run_id:
          boundedString(event?.runId, 256) ??
          boundedString(ctx?.runId, 256) ??
          "",
        timestamp: event?.timestamp,
        content: typeof event?.content === "string" ? event.content : "",
        event_type:
          event?.providerUpdate?.kind === "edit"
            ? "edit"
            : event?.providerUpdate?.kind === "delete"
              ? "delete"
              : "message",
        delivery_state: "received",
        provider_update:
          event?.providerUpdate && typeof event.providerUpdate === "object"
            ? {
                id: boundedString(event.providerUpdate.id, 1024),
                kind: boundedString(event.providerUpdate.kind, 128),
              }
            : undefined,
        metadata: communicationMetadata(event),
        source: "openclaw.message_received",
      });
    },

    onMessageSent(event, ctx) {
      const scope = scopedContext(ctx);
      if (!scope) {
        logger.warn?.(
          "entroly: skipped unscoped outbound communication observation",
        );
        return;
      }
      submit({
        direction: "outbound",
        ...scope,
        conversation_kind: conversationKind(event),
        recipient_id: boundedString(event?.to, 1024) ?? "",
        message_id:
          boundedString(event?.messageId, 1024) ??
          boundedString(ctx?.messageId, 1024) ??
          "",
        session_key:
          boundedString(event?.sessionKey, 2048) ??
          boundedString(ctx?.sessionKey, 2048) ??
          "",
        run_id:
          boundedString(event?.runId, 256) ??
          boundedString(ctx?.runId, 256) ??
          "",
        content: typeof event?.content === "string" ? event.content : "",
        event_type: "message",
        delivery_state: event?.success === true ? "sent" : "failed",
        metadata: {
          ...communicationMetadata(event),
          ...(event?.success === false
            ? { delivery_error: safeDiagnostic(event?.error) }
            : {}),
        },
        source: "openclaw.message_sent",
      });
    },

    async status() {
      return await bridge.request({
        operation: "communication_status",
        ...requestBase,
      });
    },
  };
}

export function formatCommunicationStatus({ enabled, result, error }) {
  if (!enabled) {
    return [
      "Entroly Communication Assurance: disabled",
      "Enable plugins.entries.entroly.config.communicationAssurance explicitly.",
      "For WhatsApp, also opt in to channels.whatsapp.pluginHooks.messageReceived.",
    ].join("\n");
  }
  if (error) {
    return [
      "Entroly Communication Assurance: not ready",
      `Reason: ${safeDiagnostic(error)}`,
    ].join("\n");
  }
  const stats = result?.stats ?? {};
  return [
    "Entroly Communication Assurance: observing",
    `Events: ${Number(stats.events ?? 0).toLocaleString()} (${Number(
      stats.inbound ?? 0,
    ).toLocaleString()} inbound / ${Number(stats.outbound ?? 0).toLocaleString()} outbound)`,
    `Conversation scopes: ${Number(stats.conversations ?? 0).toLocaleString()}`,
    `Raw-evidence retention: ${Number(stats.retention_days ?? 0).toLocaleString()} day(s); 0 means no automatic pruning`,
    "Storage: local user-state directory by default; no provider call is made.",
  ].join("\n");
}
