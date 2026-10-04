function scalar(value, fallback = "") {
  return typeof value === "string" && value.trim() ? value.trim() : fallback;
}

function jsonToolResult(details) {
  return {
    content: [{ type: "text", text: JSON.stringify(details, null, 2) }],
    details,
  };
}

function trustedScope(ctx) {
  const channel = scalar(ctx?.messageChannel) || scalar(ctx?.deliveryContext?.channel);
  const conversationId = scalar(ctx?.nativeChannelId);
  const accountId = scalar(ctx?.deliveryContext?.accountId);
  return { channel, conversationId, accountId };
}

function requireOwner(ctx) {
  if (ctx?.senderIsOwner !== true) {
    throw new Error("entroly communication secretary tools require owner authority");
  }
}

function requireCurrentConversation(ctx) {
  const scope = trustedScope(ctx);
  if (!scope.channel || !scope.conversationId) {
    throw new Error(
      "entroly communication tool requires host-bound channel and nativeChannelId",
    );
  }
  return scope;
}

function bridgeBase(config) {
  return {
    store_path:
      typeof config.communicationStorePath === "string"
        ? config.communicationStorePath
        : undefined,
    memory_path:
      typeof config.communicationMemoryPath === "string"
        ? config.communicationMemoryPath
        : undefined,
    retention_days: Number.isInteger(config.communicationRetentionDays)
      ? config.communicationRetentionDays
      : undefined,
  };
}

function policyArgs(config) {
  return {
    policy_mode:
      typeof config.communicationPolicyMode === "string"
        ? config.communicationPolicyMode
        : "observe",
    auto_actions: Array.isArray(config.communicationAutoActions)
      ? config.communicationAutoActions
      : [],
    auto_categories: Array.isArray(config.communicationAutoCategories)
      ? config.communicationAutoCategories
      : [],
  };
}

function sourceIds(params) {
  if (!Array.isArray(params?.source_event_ids)) return [];
  return [...new Set(params.source_event_ids.map(String).map((v) => v.trim()).filter(Boolean))].slice(
    0,
    500,
  );
}

const BriefSchema = {
  type: "object",
  additionalProperties: false,
  properties: {
    scope: {
      type: "string",
      enum: ["current", "all"],
      description:
        "current = current host-bound conversation; all = owner-authorized recent communication on the current channel.",
    },
    limit: { type: "integer", minimum: 1, maximum: 10000 },
  },
};

const AssureSchema = {
  type: "object",
  additionalProperties: false,
  required: ["action_type", "source_event_ids"],
  properties: {
    action_type: {
      type: "string",
      enum: ["reply", "group_reply", "react", "no_action"],
    },
    source_event_ids: {
      type: "array",
      minItems: 1,
      maxItems: 500,
      uniqueItems: true,
      items: { type: "string", minLength: 1, maxLength: 128 },
    },
    text: { type: "string", maxLength: 20000 },
  },
};

const ExecuteSchema = {
  type: "object",
  additionalProperties: false,
  required: ["action_type", "source_event_ids", "text"],
  properties: {
    action_type: {
      type: "string",
      enum: ["reply", "group_reply"],
    },
    source_event_ids: {
      type: "array",
      minItems: 1,
      maxItems: 500,
      uniqueItems: true,
      items: { type: "string", minLength: 1, maxLength: 128 },
    },
    text: { type: "string", minLength: 1, maxLength: 20000 },
  },
};

const TasteSchema = {
  type: "object",
  additionalProperties: false,
  properties: {
    include_contact: { type: "boolean" },
  },
};

export function registerCommunicationTools(api, { bridge, config = {} }) {
  const base = bridgeBase(config);

  api.registerTool(
    {
      contextVersion: 2,
      create: (ctx) => {
        if (
          config.communicationSecretaryTools !== true ||
          ctx.senderIsOwner !== true
        ) return null;
        return {
          name: "entroly_communication_brief",
          label: "Entroly communication brief",
          description:
            "Return an evidence-referenced communication attention brief. Use current scope by default; all scope is owner-only and never sends messages.",
          parameters: BriefSchema,
          async execute(_toolCallId, params = {}) {
            requireOwner(ctx);
            ctx.assertInvocationCurrent();
            const mode = params.scope === "all" ? "all" : "current";
            if (mode === "all" && config.communicationGlobalAccess !== true) {
              throw new Error(
                "cross-conversation communication brief requires communicationGlobalAccess=true",
              );
            }
            const trusted = trustedScope(ctx);
            const request = {
              operation: "communication_digest",
              ...base,
              channel: trusted.channel,
              account_id: trusted.accountId,
              limit: Number.isInteger(params.limit) ? params.limit : 1000,
              owner_authorized: true,
            };
            if (mode === "current") {
              if (!trusted.channel || !trusted.conversationId) {
                throw new Error(
                  "current communication brief requires a host-bound conversation",
                );
              }
              request.conversation_id = trusted.conversationId;
            }
            const result = await bridge.request(request);

            const scopes = [
              {
                scope_type: "owner",
                scope_id: scalar(ctx.agentId, "default-owner"),
              },
            ];
            if (mode === "current" && trusted.conversationId) {
              if (ctx.requesterSenderId) {
                scopes.push({
                  scope_type: "contact",
                  scope_id: String(ctx.requesterSenderId),
                });
              }
              scopes.push({
                scope_type: "conversation",
                scope_id: trusted.conversationId,
              });
            }
            let taste;
            try {
              taste = await bridge.request({
                operation: "communication_resolve_taste",
                ...base,
                owner_authorized: true,
                scopes,
              });
            } catch {
              taste = undefined;
            }
            ctx.assertInvocationCurrent();
            return jsonToolResult({
              ...result,
              resolved_taste: taste?.resolved,
              taste_authority_expanded: false,
            });
          },
        };
      },
    },
    { name: "entroly_communication_brief", optional: true },
  );

  api.registerTool(
    {
      contextVersion: 2,
      create: (ctx) => {
        if (
          config.communicationSecretaryTools !== true ||
          ctx.senderIsOwner !== true
        ) return null;
        return {
          name: "entroly_communication_assure",
          label: "Entroly communication assurance",
          description:
            "Preflight a proposed reply/reaction against Entroly evidence and delegation policy. This tool never sends anything.",
          parameters: AssureSchema,
          async execute(_toolCallId, params) {
            requireOwner(ctx);
            const scope = requireCurrentConversation(ctx);
            ctx.assertInvocationCurrent();
            const result = await bridge.request({
              operation: "communication_assure",
              ...base,
              ...policyArgs(config),
              channel: scope.channel,
              account_id: scope.accountId,
              conversation_id: scope.conversationId,
              action_type: params.action_type,
              source_event_ids: sourceIds(params),
              payload: typeof params.text === "string" ? params.text : "",
            });
            return jsonToolResult(result);
          },
        };
      },
    },
    { name: "entroly_communication_assure", optional: true },
  );

  api.registerTool(
    {
      contextVersion: 2,
      create: (ctx) => {
        if (
          config.communicationExecution !== true ||
          ctx.senderIsOwner !== true ||
          typeof ctx.delivery?.send !== "function"
        ) {
          return null;
        }
        return {
          name: "entroly_communication_execute",
          label: "Entroly bounded communication",
          description:
            "Send one current-conversation text only after Entroly returns ALLOW and atomically claims dispatch. Disabled unless communicationExecution is explicitly enabled.",
          parameters: ExecuteSchema,
          async execute(_toolCallId, params) {
            requireOwner(ctx);
            const scope = requireCurrentConversation(ctx);
            const ids = sourceIds(params);
            const text = String(params.text ?? "");
            const assured = await bridge.request({
              operation: "communication_assure",
              ...base,
              ...policyArgs(config),
              channel: scope.channel,
              account_id: scope.accountId,
              conversation_id: scope.conversationId,
              action_type: params.action_type,
              source_event_ids: ids,
              payload: text,
            });
            if (assured.decision !== "allow") {
              return jsonToolResult({
                ...assured,
                dispatched: false,
              });
            }

            ctx.assertInvocationCurrent();
            const claim = await bridge.request({
              operation: "communication_begin_action",
              ...base,
              action_id: assured.action_id,
            });
            if (claim.claimed !== true) {
              return jsonToolResult({
                ...assured,
                dispatched: false,
                dispatch_claimed: false,
                execution_state: claim.execution_state,
                reason: "dispatch_already_claimed_or_not_dispatchable",
              });
            }

            try {
              // Final host authority check immediately before the external effect.
              ctx.assertInvocationCurrent();
              await ctx.delivery.send({ text });
              return jsonToolResult({
                ...assured,
                dispatched: true,
                dispatch_claimed: true,
                execution_state: "dispatching",
                final_outcome:
                  "pending message_sent observation; no exactly-once delivery claim",
              });
            } catch (error) {
              await bridge.request({
                operation: "communication_fail_action",
                ...base,
                action_id: assured.action_id,
                error: String(error?.message ?? error ?? "delivery_failed").slice(0, 1000),
              });
              throw error;
            }
          },
        };
      },
    },
    { name: "entroly_communication_execute", optional: true },
  );

  api.registerTool(
    {
      contextVersion: 2,
      create: (ctx) => {
        if (
          config.communicationSecretaryTools !== true ||
          ctx.senderIsOwner !== true
        ) return null;
        return {
          name: "entroly_communication_taste",
          label: "Entroly communication taste",
          description:
            "Resolve the current owner's evidence-bounded communication style preferences from explicit policy plus MemoryOS. This tool cannot grant sending authority.",
          parameters: TasteSchema,
          async execute(_toolCallId, params = {}) {
            requireOwner(ctx);
            ctx.assertInvocationCurrent();
            const scope = trustedScope(ctx);
            const scopes = [
              {
                scope_type: "owner",
                scope_id: scalar(ctx.agentId, "default-owner"),
              },
            ];
            if (params.include_contact === true && ctx.requesterSenderId) {
              scopes.push({
                scope_type: "contact",
                scope_id: String(ctx.requesterSenderId),
              });
            }
            if (scope.conversationId) {
              scopes.push({
                scope_type: "conversation",
                scope_id: scope.conversationId,
              });
            }
            const result = await bridge.request({
              operation: "communication_resolve_taste",
              ...base,
              owner_authorized: true,
              scopes,
            });
            return jsonToolResult(result);
          },
        };
      },
    },
    { name: "entroly_communication_taste", optional: true },
  );
}
