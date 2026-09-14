package io.github.juyterman1000.entroly;

import com.intellij.ide.plugins.IdeaPluginDescriptor;
import com.intellij.ide.plugins.PluginManagerCore;
import com.intellij.openapi.extensions.PluginId;

final class EntrolyMcpConfig {
    static final String PLUGIN_ID = "io.github.juyterman1000.entroly";

    private EntrolyMcpConfig() {
    }

    static String current() {
        IdeaPluginDescriptor descriptor = PluginManagerCore.getPlugin(PluginId.getId(PLUGIN_ID));
        if (descriptor == null) {
            throw new IllegalStateException("Entroly plugin descriptor is unavailable");
        }
        return render(descriptor.getVersion());
    }

    static String render(String version) {
        if (!version.matches("[0-9]+\\.[0-9]+\\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?")) {
            throw new IllegalArgumentException("Invalid Entroly version: " + version);
        }

        return """
                {
                  "mcpServers": {
                    "entroly": {
                      "command": "npx",
                      "args": ["-y", "entroly-mcp@%s", "serve"],
                      "env": {
                        "ENTROLY_NO_DOCKER": "1",
                        "ENTROLY_MCP_PASSIVE": "1",
                        "ENTROLY_MCP_PROFILE": "public",
                        "ENTROLY_MAX_FILES": "200"
                      }
                    }
                  }
                }
                """.formatted(version);
    }
}
