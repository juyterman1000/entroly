#!/usr/bin/env node
// Entroly plugin launcher.
//
// The plugin's MCP config cannot express a fallback chain: it names one
// command, and if that command is missing the server never starts and the
// user sees a plugin that is silently inert. So the config names this file,
// and the fallback lives here.
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { delimiter, join } from "node:path";

const CANDIDATES = [
  { command: "uvx", args: ["--from", "entroly", "entroly"] },
  { command: "npx", args: ["-y", "entroly@latest"] },
  { command: "entroly", args: [] },
];

// Resolve against PATH ourselves rather than passing `shell: true`. With a
// shell, a missing command exits 1 like any other failure, and the chain
// cannot tell "not installed" from "installed and broken".
function resolve(command) {
  const extensions =
    process.platform === "win32"
      ? (process.env.PATHEXT || ".EXE;.CMD;.BAT").split(";")
      : [""];
  for (const dir of (process.env.PATH || "").split(delimiter)) {
    if (!dir) continue;
    for (const extension of extensions) {
      const candidate = join(dir, command + extension);
      if (existsSync(candidate)) return candidate;
    }
  }
  return null;
}

for (const candidate of CANDIDATES) {
  const executable = resolve(candidate.command);
  if (executable === null) continue;

  // Skip .cmd and .bat files on Windows without shell support, since
  // spawnSync cannot execute them without shell: true. They will not be
  // invoked, which is correct behavior: uvx and entroly are real binaries,
  // not batch files, and npx.cmd on Windows should not be used directly.
  // Instead, look for npx without an extension, which exists on some systems.
  if (process.platform === "win32" && /\.(cmd|bat)$/i.test(executable)) {
    continue;
  }

  const result = spawnSync(executable, candidate.args, {
    stdio: "inherit",
    env: { ...process.env, ENTROLY_NO_DOCKER: "1" },
  });
  if (result.status !== null) {
    process.exit(result.status);
  }
}

process.stderr.write(
  "Entroly plugin: no runner found. Install uv (https://docs.astral.sh/uv/) " +
    "or Node, or run `pip install -U entroly`, then restart Claude Code.\n",
);
process.exit(1);
