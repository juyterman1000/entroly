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
    api.registerContextEngine("entroly", (factoryContext) => {
      latestWorkspaceDir = factoryContext.workspaceDir;
      return createEntrolyContextEngine({
        bridge,
        delegateCompaction: delegateCompactionToRuntime,
        buildMemoryPrompt: buildMemorySystemPromptAddition,
        config: { ...config, workspaceDir: factoryContext.workspaceDir },
        logger: api.logger,
        statusBySession,
      });
    });
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
