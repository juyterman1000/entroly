import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import {
  buildMemorySystemPromptAddition,
  delegateCompactionToRuntime,
} from "openclaw/plugin-sdk/core";
import { EntrolyBridgeClient } from "./bridge-client.js";
import {
  createEntrolyContextEngine,
  formatEntrolyDoctor,
  formatEntrolyStatus,
} from "./engine.js";
import { createProofGuidedHooks } from "./proof-hooks.js";
import {
  createCommunicationHooks,
  formatCommunicationStatus,
} from "./communication-hooks.js";
import { registerCommunicationTools } from "./communication-tools.js";

export default definePluginEntry({
  id: "entroly",
  name: "Entroly Context Engine",
  register(api) {
    const config = api.pluginConfig ?? {};
    let latestWorkspaceDir;
    const bridge = new EntrolyBridgeClient({
      pythonCommand: config.pythonCommand ?? "python",
      timeoutMs: config.timeoutMs ?? 5000,
      logger: api.logger,
    });
    const statusBySession = new Map();
    const proofStateBySession = new Map();
    api.registerContextEngine("entroly", (factoryContext) => {
      latestWorkspaceDir = factoryContext.workspaceDir;
      return createEntrolyContextEngine({
        bridge,
        delegateCompaction: delegateCompactionToRuntime,
        buildMemoryPrompt: buildMemorySystemPromptAddition,
        config: { ...config, workspaceDir: factoryContext.workspaceDir },
        logger: api.logger,
        statusBySession,
        proofStateBySession,
      });
    });
    let communicationHooks;
    if (config.communicationAssurance === true) {
      if (typeof api.on !== "function") {
        api.logger.error?.(
          "entroly: communicationAssurance requires typed OpenClaw message hooks",
        );
      } else {
        communicationHooks = createCommunicationHooks({
          bridge,
          config,
          logger: api.logger,
        });
        api.on("message_received", communicationHooks.onMessageReceived);
        api.on("message_sent", communicationHooks.onMessageSent);
        registerCommunicationTools(api, { bridge, config });
        if (config.communicationTasteLearning === true) {
          void bridge.request({
            operation: "communication_start_taste_autotune",
            owner_authorized: true,
            store_path:
              typeof config.communicationStorePath === "string"
                ? config.communicationStorePath
                : undefined,
            interval_s:
              Number.isFinite(config.communicationTasteAutotuneIntervalSeconds)
                ? config.communicationTasteAutotuneIntervalSeconds
                : 30,
          }).catch((error) => {
            api.logger.warn?.(
              `entroly: taste autotune did not start; learning remains paused: ${String(
                error?.message ?? error,
              ).replace(/\s+/g, " ").slice(0, 240)}`,
            );
          });
        }
      }
    }
    if (config.proofGuidedRecovery === true) {
      if (typeof api.on !== "function") {
        api.logger.error?.(
          "entroly: this OpenClaw host does not expose typed proof-guided hooks; disable proofGuidedRecovery or upgrade OpenClaw",
        );
      } else {
        const proofHooks = createProofGuidedHooks({
          bridge,
          config,
          logger: api.logger,
          proofStateBySession,
          statusBySession,
        });
        api.on("llm_output", proofHooks.onLlmOutput);
        api.on("before_agent_finalize", proofHooks.onBeforeAgentFinalize);
        api.on("reply_payload_sending", proofHooks.onReplyPayloadSending);
      }
    }
    api.registerCommand({
      name: "entroly-context",
      description: "Show Entroly context status; use `doctor` or `communication`.",
      acceptsArgs: true,
      handler: async (ctx) => {
        const command = ctx.args?.trim().toLowerCase();
        if (command === "communication" || command === "secretary") {
          if (!communicationHooks) {
            return {
              text: formatCommunicationStatus({ enabled: false }),
            };
          }
          try {
            return {
              text: formatCommunicationStatus({
                enabled: true,
                result: await communicationHooks.status(),
              }),
            };
          } catch (error) {
            return {
              text: formatCommunicationStatus({ enabled: true, error }),
            };
          }
        }
        if (command === "doctor") {
          try {
            await bridge.health({
              workspaceDir: latestWorkspaceDir,
              receiptDir: config.receiptDir,
              writeReceipts: config.writeReceipts !== false,
            });
            return {
              text: formatEntrolyDoctor({
                ok: true,
                pythonCommand: config.pythonCommand ?? "python",
              }),
            };
          } catch (error) {
            return {
              text: formatEntrolyDoctor({
                ok: false,
                error,
                pythonCommand: config.pythonCommand ?? "python",
              }),
            };
          }
        }
        return {
          text: formatEntrolyStatus(
            ctx.sessionId ? statusBySession.get(ctx.sessionId) : undefined,
          ),
        };
      },
    });
  },
});
