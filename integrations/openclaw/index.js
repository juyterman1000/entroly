import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { EntrolyBridgeClient } from "./bridge-client.js";
import { createEntrolyContextEngine, formatEntrolyStatus } from "./engine.js";

export default definePluginEntry({
  id: "entroly",
  name: "Entroly Context Engine",
  register(api) {
    const config = api.pluginConfig ?? {};
    const bridge = new EntrolyBridgeClient({
      pythonCommand: config.pythonCommand ?? "python",
      timeoutMs: config.timeoutMs ?? 5000,
      logger: api.logger,
    });
    const statusBySession = new Map();
    api.registerContextEngine("entroly", (factoryContext) =>
      createEntrolyContextEngine({
        bridge,
        config: { ...config, workspaceDir: factoryContext.workspaceDir },
        logger: api.logger,
        statusBySession,
      }),
    );
    api.registerCommand({
      name: "entroly-context",
      description: "Show Entroly context savings, warnings, and the latest receipt.",
      acceptsArgs: false,
      handler: (ctx) => ({
        text: formatEntrolyStatus(
          ctx.sessionId ? statusBySession.get(ctx.sessionId) : undefined,
        ),
      }),
    });
  },
});
