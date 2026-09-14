package io.github.juyterman1000.entroly;

import com.intellij.ide.BrowserUtil;
import com.intellij.ide.plugins.PluginManagerCore;
import com.intellij.openapi.options.ShowSettingsUtil;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.extensions.PluginId;
import com.intellij.openapi.ide.CopyPasteManager;
import com.intellij.openapi.ui.Messages;

import java.awt.datatransfer.StringSelection;

final class EntrolyUi {
    private static final PluginId AI_ASSISTANT_ID = PluginId.getId("com.intellij.ml.llm");
    private static final String MCP_DOCS = "https://www.jetbrains.com/help/ai-assistant/mcp.html";
    private static final String ENTROLY_DOCS = "https://github.com/juyterman1000/entroly#works-with-your-stack";

    private EntrolyUi() {
    }

    static void copyConfiguration(Project project, boolean openSettings) {
        CopyPasteManager.getInstance().setContents(new StringSelection(EntrolyMcpConfig.current()));

        if (openSettings && PluginManagerCore.isPluginInstalled(AI_ASSISTANT_ID)) {
            ShowSettingsUtil.getInstance().showSettingsDialog(project, "Model Context Protocol (MCP)");
            return;
        }

        String detail = openSettings
                ? "The MCP configuration was copied. Install or enable JetBrains AI Assistant, then open Settings | Tools | AI Assistant | Model Context Protocol (MCP) and paste it."
                : "The version-pinned Entroly MCP configuration was copied to the clipboard.";
        Messages.showInfoMessage(project, detail, "Entroly");
    }

    static void openMcpDocumentation() {
        BrowserUtil.browse(MCP_DOCS);
    }

    static void openEntrolyDocumentation() {
        BrowserUtil.browse(ENTROLY_DOCS);
    }
}
