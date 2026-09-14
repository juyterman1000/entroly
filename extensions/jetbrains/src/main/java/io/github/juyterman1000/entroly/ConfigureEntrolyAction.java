package io.github.juyterman1000.entroly;

import com.intellij.openapi.actionSystem.AnAction;
import com.intellij.openapi.actionSystem.AnActionEvent;
import org.jetbrains.annotations.NotNull;

public final class ConfigureEntrolyAction extends AnAction {
    @Override
    public void actionPerformed(@NotNull AnActionEvent event) {
        EntrolyUi.copyConfiguration(event.getProject(), true);
    }
}
