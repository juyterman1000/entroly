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
      description: "Show Entroly context savings or run `doctor`.",
      acceptsArgs: true,
      handler: async (ctx) => {
        if (ctx.args?.trim().toLowerCase() === "doctor") {
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
