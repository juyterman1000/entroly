package io.github.juyterman1000.entroly;

import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

public final class EntrolyMcpConfigTest {
    @Test
    public void rendersVersionPinnedPassiveConfiguration() {
        String config = EntrolyMcpConfig.render("1.2.3");

        assertTrue(config.contains("\"entroly-mcp@1.2.3\""));
        assertTrue(config.contains("\"ENTROLY_MCP_PASSIVE\": \"1\""));
        assertTrue(config.contains("\"ENTROLY_NO_DOCKER\": \"1\""));
        assertTrue(config.contains("\"ENTROLY_MCP_PROFILE\": \"public\""));
        assertTrue(config.contains("\"ENTROLY_MAX_FILES\": \"200\""));
        assertFalse(config.contains("apiKey"));
        assertFalse(config.contains("token"));
    }

    @Test
    public void rejectsNonSemverPackageVersion() {
        assertThrows(IllegalArgumentException.class, () -> EntrolyMcpConfig.render("latest"));
    }

    @Test
    public void pinsTheVersionBakedInAtBuildTime() {
        // current() no longer needs a running IDE, so the generated constant is
        // checked here: it must be wired into the source set and must survive
        // render()'s semver validation. A malformed version in pyproject.toml now
        // fails the build instead of throwing in the user's IDE.
        String config = EntrolyMcpConfig.current();

        assertTrue(config.contains("\"entroly-mcp@" + EntrolyVersion.VALUE + "\""));
    }

    @Test
    public void boundsRuntimeOutputBeforeDisplayingIt() {
        String untrustedOutput = "1.2.3\u0000" + "x".repeat(200) + "\nignored";
        String normalized = EntrolyRuntimeCheck.normalizeOutput(untrustedOutput);

        assertTrue(normalized.startsWith("1.2.3"));
        assertTrue(normalized.endsWith("..."));
        assertTrue(normalized.length() <= 120);
        assertFalse(normalized.contains("ignored"));
        assertFalse(normalized.contains("\u0000"));
    }
}
