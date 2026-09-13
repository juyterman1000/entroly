#!/usr/bin/env node
// Entroly plugin launcher.
//
// The plugin's MCP config cannot express a fallback chain: it names one
// command, and if that command is missing the server never starts and the
// user sees a plugin that is silently inert. So the config names this file,
// and the fallback lives here.
import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { delimiter, join } from "node:path";

const requestedArgs = process.argv.slice(2);
const require = createRequire(import.meta.url);
let packagedCli = null;
try {
  packagedCli = require.resolve("entroly-wasm/bin/entroly-wasm.js");
} catch {
  // Git/local plugin bundles intentionally have no vendored dependency. Their
  // fallback chain remains available, while npm marketplace installs resolve
  // the declared local dependency without a network call on every prompt.
}
const PACKAGED_HOOK_CANDIDATES = packagedCli
  ? [{ executable: process.execPath, args: [packagedCli] }]
  : [];
const PACKAGED_SERVER_CANDIDATES = packagedCli
  ? [{ executable: process.execPath, args: [packagedCli, "serve"] }]
  : [];
const SERVER_CANDIDATES = [
  ...PACKAGED_SERVER_CANDIDATES,
  { command: "uvx", args: ["--from", "entroly", "entroly", "serve"] },
  { command: "npx", args: ["-y", "entroly@latest", "serve"] },
  { command: "entroly", args: ["serve"] },
];
// Hooks run once per prompt and must stay cheap. Prefer an installed Entroly
// executable there; retain uvx-first behavior for the long-lived MCP server.
const HOOK_CANDIDATES = [
  ...PACKAGED_HOOK_CANDIDATES,
  { command: "entroly", args: [] },
  { command: "uvx", args: ["--from", "entroly", "entroly"] },
  { command: "npx", args: ["-y", "entroly@latest"] },
];
const CANDIDATES = requestedArgs.length > 0 ? HOOK_CANDIDATES : SERVER_CANDIDATES;
const hookInput = requestedArgs.length > 0 ? readFileSync(0) : null;

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
    env: { ...process.env, ENTROLY_NO_DOCKER: "1" },
  };
  if (hookInput === null) {
    options.stdio = "inherit";
  } else {
    options.input = hookInput;
    options.encoding = "buffer";
    options.maxBuffer = 4 * 1024 * 1024;
    options.windowsHide = true;
  }
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
  const executable = candidate.executable || resolve(candidate.command);
  if (executable === null) continue;

  const result = runCandidate(executable, [...candidate.args, ...requestedArgs]);
  if (result.status === 0) {
    if (hookInput !== null) {
      if (result.stdout) process.stdout.write(result.stdout);
      if (result.stderr) process.stderr.write(result.stderr);
    }
    process.exit(0);
  }
  // A present executable can still be stale or incomplete. Hook output is
  // buffered, so a failed candidate cannot corrupt the host's JSON protocol;
  // continue through the deterministic fallback chain.
}

process.stderr.write(
  "Entroly plugin: no runner found. Install uv (https://docs.astral.sh/uv/) " +
    "or Node, or run `pip install -U entroly`, then restart your agent host.\n",
);
process.exit(1);
