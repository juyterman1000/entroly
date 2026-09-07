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

// Windows cannot exec a .cmd/.bat directly: spawnSync without a shell throws
// EINVAL. Route those through cmd.exe with every part quoted, so `npx` -- which
// ships as npx.cmd on Windows -- stays a real link in the chain instead of
// being skipped. `/d` skips AutoRun, `/s` makes cmd strip exactly the outer
// quote pair and take the rest verbatim.
function runCandidate(executable, args) {
  const options = {
    stdio: "inherit",
    env: { ...process.env, ENTROLY_NO_DOCKER: "1" },
  };
  if (process.platform === "win32" && /\.(cmd|bat)$/i.test(executable)) {
    const line = [executable, ...args].map((part) => `"${part}"`).join(" ");
    return spawnSync(
      process.env.COMSPEC || "cmd.exe",
      ["/d", "/s", "/c", `"${line}"`],
      { ...options, windowsVerbatimArguments: true },
    );
  }
  return spawnSync(executable, args, options);
}

for (const candidate of CANDIDATES) {
  const executable = resolve(candidate.command);
  if (executable === null) continue;

  const result = runCandidate(executable, candidate.args);
  // A null status means the process never launched. The next candidate may
  // still work, so fall through instead of exiting -- this is exactly the
  // "installed but broken" case the shell-free design exists to detect.
  if (result.status !== null) process.exit(result.status);
}

process.stderr.write(
  "Entroly plugin: no runner found. Install uv (https://docs.astral.sh/uv/) " +
    "or Node, or run `pip install -U entroly`, then restart Claude Code.\n",
);
process.exit(1);
