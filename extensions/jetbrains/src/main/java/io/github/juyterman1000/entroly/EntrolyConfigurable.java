package io.github.juyterman1000.entroly;

import com.intellij.openapi.application.ApplicationManager;
import com.intellij.openapi.options.Configurable;
import com.intellij.ui.JBColor;
import com.intellij.ui.components.JBLabel;
import com.intellij.ui.components.JBPanel;
import com.intellij.ui.components.JBTextArea;
import com.intellij.util.ui.JBUI;
import org.jetbrains.annotations.Nls;
import org.jetbrains.annotations.Nullable;

import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JComponent;
import javax.swing.JPanel;
import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.FlowLayout;

public final class EntrolyConfigurable implements Configurable {
    private JPanel root;

    @Override
    public @Nls(capitalization = Nls.Capitalization.Title) String getDisplayName() {
        return "Entroly";
    }

    @Override
    public @Nullable JComponent createComponent() {
        root = new JBPanel<>(new BorderLayout());
        root.setBorder(JBUI.Borders.empty(16));

        JPanel content = new JBPanel<>();
        content.setLayout(new BoxLayout(content, BoxLayout.Y_AXIS));

        JBLabel title = new JBLabel("Entroly for JetBrains AI Assistant");
        title.setFont(title.getFont().deriveFont(title.getFont().getSize2D() + 4f));
        title.setAlignmentX(JComponent.LEFT_ALIGNMENT);
        content.add(title);
        content.add(Box.createVerticalStrut(8));

        JBTextArea explanation = new JBTextArea(
                "Entroly runs as a local MCP server. Copy the configuration, add it in " +
                        "Settings | Tools | AI Assistant | Model Context Protocol (MCP), then confirm " +
                        "that JetBrains reports the server as connected."
        );
        explanation.setEditable(false);
        explanation.setLineWrap(true);
        explanation.setWrapStyleWord(true);
        explanation.setRows(3);
        explanation.setOpaque(false);
        explanation.setBorder(null);
        explanation.setFocusable(false);
        explanation.setCaretPosition(0);
        explanation.setMaximumSize(new Dimension(Integer.MAX_VALUE, explanation.getPreferredSize().height));
        explanation.setAlignmentX(JComponent.LEFT_ALIGNMENT);
        content.add(explanation);
        content.add(Box.createVerticalStrut(12));

        JBLabel status = new JBLabel("Runtime has not been checked.");
        status.setAlignmentX(JComponent.LEFT_ALIGNMENT);
        content.add(status);
        content.add(Box.createVerticalStrut(8));

        JPanel buttons = new JBPanel<>(new FlowLayout(FlowLayout.LEFT, 8, 0));
        buttons.setAlignmentX(JComponent.LEFT_ALIGNMENT);

        JButton configure = new JButton("Copy Configuration and Open MCP Settings");
        configure.addActionListener(event -> EntrolyUi.copyConfiguration(null, true));
        buttons.add(configure);

        JButton check = new JButton("Check npx Runtime");
        check.addActionListener(event -> {
            check.setEnabled(false);
            status.setText("Checking npx on PATH...");
            ApplicationManager.getApplication().executeOnPooledThread(() -> {
                EntrolyRuntimeCheck.Result result = EntrolyRuntimeCheck.checkNpx();
                ApplicationManager.getApplication().invokeLater(() -> {
                    status.setForeground(result.available() ? new JBColor(0x237804, 0x73D13D) : JBColor.RED);
                    status.setText(result.message());
                    check.setEnabled(true);
                });
            });
        });
        buttons.add(check);
        content.add(buttons);
        content.add(Box.createVerticalStrut(8));

        JPanel links = new JBPanel<>(new FlowLayout(FlowLayout.LEFT, 8, 0));
        links.setAlignmentX(JComponent.LEFT_ALIGNMENT);
        JButton jetBrainsDocs = new JButton("JetBrains MCP Documentation");
        jetBrainsDocs.addActionListener(event -> EntrolyUi.openMcpDocumentation());
        links.add(jetBrainsDocs);
        JButton entrolyDocs = new JButton("Entroly Documentation");
        entrolyDocs.addActionListener(event -> EntrolyUi.openEntrolyDocumentation());
        links.add(entrolyDocs);
        content.add(links);
        content.add(Box.createVerticalStrut(12));

        JBTextArea boundary = new JBTextArea(
                "Evidence boundary: an MCP connection exposes Entroly tools to AI Assistant. It does not " +
                        "prove that every model request used those tools or that provider traffic was intercepted. " +
                        "Use Entroly receipts to verify actual use."
        );
        boundary.setEditable(false);
        boundary.setLineWrap(true);
        boundary.setWrapStyleWord(true);
        boundary.setRows(3);
        boundary.setOpaque(true);
        boundary.setBorder(JBUI.Borders.empty(8));
        boundary.setBackground(new JBColor(0xF4F8F3, 0x243326));
        boundary.setCaretPosition(0);
        boundary.setMaximumSize(new Dimension(Integer.MAX_VALUE, boundary.getPreferredSize().height));
        boundary.setAlignmentX(JComponent.LEFT_ALIGNMENT);
        content.add(boundary);

        root.add(content, BorderLayout.NORTH);
        return root;
    }

    @Override
    public boolean isModified() {
        return false;
    }

    @Override
    public void apply() {
        // Entroly does not persist hidden settings or modify JetBrains configuration directly.
    }

    @Override
    public void disposeUIResources() {
        root = null;
    }
}
