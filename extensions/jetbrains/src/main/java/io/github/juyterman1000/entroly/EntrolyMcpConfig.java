package io.github.juyterman1000.entroly;

final class EntrolyMcpConfig {

    private EntrolyMcpConfig() {
    }

    static String current() {
        return render(EntrolyVersion.VALUE);
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
