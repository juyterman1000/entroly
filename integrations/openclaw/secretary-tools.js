function textResult(text, details) {
  return {
    content: [{ type: "text", text: String(text ?? "") }],
    details,
  };
}

const TOOL_NAMES = [
  "entroly_communication_brief",
  "entroly_communication_assure",
  "entroly_communication_taste",
];

function boundedString(value, limit = 1024) {
  return typeof value === "string" && value.trim()
    ? value.trim().slice(0, limit)
    : "";
}

function safeArray(value, limit = 200) {
  if (!Array.isArray(value)) return [];
  return [...new Set(value.map((item) => boundedString(item, 256)).filter(Boolean))].slice(0, limit);
}

function bridgeBase(config) {
  return {
    store_path: boundedString(config.communicationStorePath, 4096) || undefined,
    memory_path: boundedString(config.communicationMemoryPath, 4096) || undefined,
    retention_days:
      Number.isInteger(config.communicationRetentionDays) &&
      config.communicationRetentionDays >= 0
        ? config.communicationRetentionDays
        : undefined,
  };
}

function currentRoute(context) {
  return {
    channel:
      boundedString(context.messageChannel, 128) ||
      boundedString(context.deliveryContext?.channel, 128),
    accountId:
      boundedString(context.agentAccountId, 256) ||
      boundedString(context.deliveryContext?.accountId, 256),
    conversationId:
      boundedString(context.nativeChannelId, 1024) ||
      boundedString(context.deliveryContext?.to, 1024),
  };
}

function errorResult(message) {
  return {
    ...textResult(message, { ok: false, error: true }),
    isError: true,
  };
}

function detailsResult(summary, details) {
  return textResult(summary, details);
}

function assertOwnerCurrent(context) {
  context.assertInvocationCurrent();
  if (context.senderIsOwner !== true) {
    throw new Error("Communication secretary tools require the trusted owner.");
  }
}

function assertReadCurrent(context) {
  context.assertInvocationCurrent();
  context.assertMemoryAudienceCurrent?.();
}

function normalizeActionType(value) {
  const action = boundedString(value, 32);
  return ["reply", "react", "group_reply", "send_message", "no_action"].includes(action)
    ? action
    : "";
}

function policyConfig(config) {
  const mode = ["observe", "suggest", "approve", "bounded"].includes(
    config.communicationPolicyMode,
  )
    ? config.communicationPolicyMode
    : "approve";
  return {
    policy_mode: mode,
    auto_actions: safeArray(config.communicationAutoActions, 16),
    auto_categories: safeArray(config.communicationAutoCategories, 32),
  };
}

export function createCommunicationSecretaryTools({
  bridge,
  config = {},
  context,
}) {
  if (
    config.communicationAssurance !== true ||
    config.communicationSecretaryTools !== true ||
    context.senderIsOwner !== true
  ) {
    return null;
  }

  const base = bridgeBase(config);

  const briefTool = {
    name: "entroly_communication_brief",
    label: "Entroly Communication Brief",
    description:
      "Read Entroly's evidence-backed communication attention brief. Use scope=current for this conversation. scope=all requires the operator's explicit communicationGlobalAccess setting. Routine items expose evidence/message IDs and safe candidate action types; important items include bounded excerpts.",
    parameters: {
      type: "object",
      additionalProperties: false,
      properties: {
        scope: { type: "string", enum: ["current", "all"], default: "current" },
        limit: { type: "integer", minimum: 1, maximum: 1000, default: 200 },
      },
    },
    async execute(_toolCallId, raw) {
      try {
        assertOwnerCurrent(context);
        const scope = raw?.scope === "all" ? "all" : "current";
        const limit = Number.isInteger(raw?.limit)
          ? Math.max(1, Math.min(raw.limit, 1000))
          : 200;
        const route = currentRoute(context);
        if (scope === "all" && config.communicationGlobalAccess !== true) {
          return errorResult(
            "Cross-conversation secretary review is disabled. Enable communicationGlobalAccess explicitly.",
          );
        }
        if (scope === "current" && (!route.channel || !route.conversationId)) {
          return errorResult(
            "OpenClaw did not provide a trusted current channel/conversation identity.",
          );
        }

        const result = await bridge.request({
          operation: "communication_digest",
          ...base,
          owner_authorized: scope === "all",
          channel: scope === "current" ? route.channel : "",
          account_id: scope === "current" ? route.accountId : undefined,
          conversation_id: scope === "current" ? route.conversationId : "",
          limit,
        });
        assertReadCurrent(context);
        const digest = result?.digest ?? {};
        return detailsResult(
          [
            `Communication brief (${scope})`,
            `Needs attention: ${Number(digest.attention_count ?? 0)}`,
            `Urgent: ${Number(digest.urgent_count ?? 0)}`,
            `Routine candidates: ${Number(digest.routine_candidate_count ?? 0)}`,
            `Group episodes: ${Array.isArray(digest.group_episodes) ? digest.group_episodes.length : 0}`,
          ].join("\n"),
          result,
        );
      } catch (error) {
        return errorResult(
          `Entroly communication brief failed safely: ${String(error?.message ?? error).slice(0, 300)}`,
        );
      }
    },
  };

  const assureTool = {
    name: "entroly_communication_assure",
    label: "Entroly Communication Action Assurance",
    description:
      "Preflight a proposed reply/reaction before using OpenClaw's own messaging action. Entroly verifies exact evidence scope, duplicate state, risk, commitments, and operator delegation policy. This tool never sends anything. Only proceed to a channel action when decision=allow.",
    parameters: {
      type: "object",
      additionalProperties: false,
      required: ["action_type", "source_event_ids"],
      properties: {
        action_type: {
          type: "string",
          enum: ["reply", "react", "group_reply", "send_message", "no_action"],
        },
        source_event_ids: {
          type: "array",
          minItems: 1,
          maxItems: 200,
          items: { type: "string", minLength: 1, maxLength: 256 },
        },
        payload: { type: "string", maxLength: 4096, default: "" },
        channel: { type: "string", maxLength: 128 },
        account_id: { type: "string", maxLength: 256 },
        conversation_id: { type: "string", maxLength: 1024 },
      },
    },
    async execute(_toolCallId, raw) {
      try {
        assertOwnerCurrent(context);
        const actionType = normalizeActionType(raw?.action_type);
        if (!actionType) {
          return errorResult("Unsupported communication action type.");
        }
        const sourceEventIds = safeArray(raw?.source_event_ids, 200);
        if (sourceEventIds.length === 0) {
          return errorResult("Communication actions require exact source_event_ids.");
        }

        const trusted = currentRoute(context);
        const requested = {
          channel: boundedString(raw?.channel, 128),
          accountId: boundedString(raw?.account_id, 256),
          conversationId: boundedString(raw?.conversation_id, 1024),
        };
        const canRetarget = config.communicationGlobalAccess === true;
        const target = canRetarget
          ? {
              channel: requested.channel || trusted.channel,
              accountId: requested.accountId || trusted.accountId,
              conversationId: requested.conversationId || trusted.conversationId,
            }
          : trusted;

        if (!target.channel || !target.conversationId) {
          return errorResult("No trusted communication target is available.");
        }
        if (
          !canRetarget &&
          ((requested.channel && requested.channel !== trusted.channel) ||
            (requested.conversationId &&
              requested.conversationId !== trusted.conversationId) ||
            (requested.accountId && requested.accountId !== trusted.accountId))
        ) {
          return errorResult(
            "Cross-conversation action preflight is disabled by communicationGlobalAccess.",
          );
        }

        // This request persists the assurance decision, so assert authority at
        // the last synchronous point before handing it to the local bridge.
        assertOwnerCurrent(context);
        const result = await bridge.request({
          operation: "communication_assure",
          ...base,
          ...policyConfig(config),
          action_type: actionType,
          source_event_ids: sourceEventIds,
          payload: typeof raw?.payload === "string" ? raw.payload : "",
          channel: target.channel,
          account_id: target.accountId,
          conversation_id: target.conversationId,
        });
        assertReadCurrent(context);
        return detailsResult(
          `Entroly action assurance: ${result?.decision ?? "unknown"} (${(result?.reasons ?? []).join(", ") || "no reason"})`,
          result,
        );
      } catch (error) {
        return errorResult(
          `Entroly communication assurance failed safely: ${String(error?.message ?? error).slice(0, 300)}`,
        );
      }
    },
  };

  const tasteTool = {
    name: "entroly_communication_taste",
    label: "Entroly Communication Taste",
    description:
      "Resolve or learn low-risk communication style preferences using Entroly MemoryOS and optional Hippocampus. Learned taste can influence draft style but never grants send authority. Learning requires communicationTasteLearning=true.",
    parameters: {
      type: "object",
      additionalProperties: false,
      properties: {
        operation: { type: "string", enum: ["resolve", "learn"], default: "resolve" },
        scope_type: {
          type: "string",
          enum: ["owner", "contact", "group", "conversation"],
          default: "conversation",
        },
        scope_id: { type: "string", maxLength: 1024 },
        channel: { type: "string", maxLength: 128 },
        account_id: { type: "string", maxLength: 256 },
      },
    },
    async execute(_toolCallId, raw) {
      try {
        assertOwnerCurrent(context);
        const operation = raw?.operation === "learn" ? "learn" : "resolve";
        if (operation === "learn" && config.communicationTasteLearning !== true) {
          return errorResult(
            "Automatic communication taste learning is disabled. Enable communicationTasteLearning explicitly.",
          );
        }

        const route = currentRoute(context);
        const scopeType = ["owner", "contact", "group", "conversation"].includes(
          raw?.scope_type,
        )
          ? raw.scope_type
          : "conversation";
        const ownerScopeId = boundedString(context.agentId, 256) || "owner";
        let scopeId = boundedString(raw?.scope_id, 1024);
        if (!scopeId) {
          scopeId = scopeType === "owner" ? ownerScopeId : route.conversationId;
        }
        if (!scopeId) {
          return errorResult("No trusted communication taste scope is available.");
        }
        if (
          scopeType !== "owner" &&
          scopeId !== route.conversationId &&
          config.communicationGlobalAccess !== true
        ) {
          return errorResult(
            "Cross-conversation taste access is disabled by communicationGlobalAccess.",
          );
        }

        const channel = boundedString(raw?.channel, 128) || route.channel;
        const accountId = boundedString(raw?.account_id, 256) || route.accountId;
        let result;
        if (operation === "learn") {
          assertOwnerCurrent(context);
          result = await bridge.request({
            operation: "communication_learn_taste",
            ...base,
            owner_authorized: true,
            scope_type: scopeType,
            scope_id: scopeId,
            channel,
            account_id: accountId || undefined,
          });
        } else {
          const scopes = [{ scope_type: "owner", scope_id: ownerScopeId }];
          if (!(scopeType === "owner" && scopeId === ownerScopeId)) {
            scopes.push({ scope_type: scopeType, scope_id: scopeId });
          }
          result = await bridge.request({
            operation: "communication_resolve_taste",
            ...base,
            owner_authorized: true,
            scopes,
          });
        }
        assertReadCurrent(context);
        return detailsResult(
          operation === "learn"
            ? `Communication taste learning: ${result?.learned === true ? "learned" : result?.reason ?? "no update"}`
            : "Resolved communication taste without changing action authority.",
          result,
        );
      } catch (error) {
        return errorResult(
          `Entroly communication taste failed safely: ${String(error?.message ?? error).slice(0, 300)}`,
        );
      }
    },
  };

  return [briefTool, assureTool, tasteTool];
}

export function registerCommunicationSecretaryTools(api, { bridge, config }) {
  api.registerTool(
    {
      contextVersion: 2,
      create: (context) =>
        createCommunicationSecretaryTools({
          bridge,
          config,
          context,
        }),
    },
    {
      names: TOOL_NAMES,
      optional: true,
    },
  );
}

export { TOOL_NAMES as communicationSecretaryToolNames };
