package io.github.juyterman1000.entroly;

import com.intellij.ide.plugins.PluginManager;
import com.intellij.openapi.extensions.PluginDescriptor;

final class EntrolyMcpConfig {

    private EntrolyMcpConfig() {
    }

    static String current() {
        // Resolve our own descriptor through the classloader that owns this class.
        // The id-based lookups are both unusable: PluginManagerCore.getPlugin is
        // @ApiStatus.Internal, and PluginManager.getPlugin(PluginId) is a deprecated
        // delegate to it. Going through the class also keeps the id in plugin.xml as
        // the single source of truth instead of restating it here.
        PluginDescriptor descriptor = PluginManager.getPluginByClass(EntrolyMcpConfig.class);
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
