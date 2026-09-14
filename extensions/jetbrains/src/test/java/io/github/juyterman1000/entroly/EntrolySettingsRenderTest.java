package io.github.juyterman1000.entroly;

import org.junit.Test;

import javax.imageio.ImageIO;
import javax.swing.JComponent;
import javax.swing.SwingUtilities;
import java.awt.Component;
import java.awt.Container;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

public final class EntrolySettingsRenderTest {
    @Test
    public void settingsPanelRendersForVisualReview() throws Exception {
        AtomicReference<JComponent> componentRef = new AtomicReference<>();
        SwingUtilities.invokeAndWait(() -> componentRef.set(new EntrolyConfigurable().createComponent()));

        JComponent component = componentRef.get();
        assertNotNull(component);

        int width = 820;
        int height = 460;
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        SwingUtilities.invokeAndWait(() -> {
            component.setSize(width, height);
            layoutRecursively(component);
            Graphics2D graphics = image.createGraphics();
            try {
                graphics.setColor(component.getBackground());
                graphics.fillRect(0, 0, width, height);
                component.printAll(graphics);
            } finally {
                graphics.dispose();
            }
        });

        Path output = Path.of("build", "reports", "ui", "entroly-settings.png");
        Files.createDirectories(output.getParent());
        ImageIO.write(image, "png", output.toFile());

        int background = image.getRGB(0, 0);
        long visiblePixels = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (image.getRGB(x, y) != background) {
                    visiblePixels++;
                }
            }
        }
        assertTrue("settings panel should paint visible content", visiblePixels > 2_000);
    }

    private static void layoutRecursively(Container container) {
        container.doLayout();
        for (Component child : container.getComponents()) {
            if (child instanceof Container childContainer) {
                layoutRecursively(childContainer);
            }
        }
    }
}
