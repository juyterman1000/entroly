package io.github.juyterman1000.entroly;

import com.intellij.openapi.util.SystemInfo;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

final class EntrolyRuntimeCheck {
    private EntrolyRuntimeCheck() {
    }

    static Result checkNpx() {
        String executable = SystemInfo.isWindows ? "npx.cmd" : "npx";
        Process process;
        try {
            process = new ProcessBuilder(executable, "--version")
                    .redirectErrorStream(true)
                    .start();
        } catch (IOException error) {
            return new Result(false, "npx was not found on PATH");
        }

        CompletableFuture<String> outputFuture = CompletableFuture.supplyAsync(() -> readBoundedOutput(process));
        try {
            if (!process.waitFor(5, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                outputFuture.cancel(true);
                return new Result(false, "npx did not respond within 5 seconds");
            }
            String output = normalizeOutput(outputFuture.get(1, TimeUnit.SECONDS));
            if (process.exitValue() != 0) {
                return new Result(false, output.isEmpty() ? "npx returned an error" : "npx error: " + output);
            }
            return new Result(true, output.isEmpty() ? "npx is available" : "npx " + output + " is available");
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            return new Result(false, "Runtime check was interrupted");
        } catch (ExecutionException | TimeoutException error) {
            return new Result(false, "Unable to read npx output");
        }
    }

    private static String readBoundedOutput(Process process) {
        try (InputStream output = process.getInputStream()) {
            return new String(output.readNBytes(512), StandardCharsets.UTF_8);
        } catch (IOException error) {
            return "";
        }
    }

    static String normalizeOutput(String output) {
        String firstLine = output.lines().findFirst().orElse("")
                .replaceAll("[\\p{Cntrl}&&[^\\t]]", "")
                .trim();
        return firstLine.length() <= 120 ? firstLine : firstLine.substring(0, 117) + "...";
    }

    record Result(boolean available, String message) {
    }
}
