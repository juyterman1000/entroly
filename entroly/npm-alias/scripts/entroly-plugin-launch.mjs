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

function candidateName(candidate) {
  return candidate.command || "bundled entroly-wasm";
}

function outcome(result) {
  // A child that exits before reading the hook payload also leaves EPIPE in
  // result.error; its exit status is what happened. The error alone means it
  // never ran (ENOENT, EACCES).
  if (typeof result.status === "number") return `exit ${result.status}`;
  if (result.signal) return `killed by ${result.signal}`;
  if (result.error) return result.error.code || result.error.message;
  return "no exit status";
}

// The last non-empty stderr line: for a Python traceback, the exception.
function lastStderrLine(result) {
  if (!result.stderr || result.stderr.length === 0) return "";
  const lines = result.stderr
    .toString("utf8")
    .replace(/\x1b\[[0-9;]*m/g, "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const line = lines.length > 0 ? lines[lines.length - 1] : "";
  return line.length > 300 ? `${line.slice(0, 297)}...` : line;
}

// Hosts show only the first stderr line of a failed hook, so that line must
// carry the cause. "Not installed" and "installed and broken" need opposite
// fixes and must not share a message.
function failureReport(attempts) {
  if (attempts.length === 0) {
    return (
      "Entroly plugin: Entroly is not installed (no entroly, uvx or npx on PATH). " +
      "Run `pip install -U entroly` or install uv (https://docs.astral.sh/uv/), " +
      "then restart your agent host.\n"
    );
  }
  const [first] = attempts;
  const cause = lastStderrLine(first.result);
  const lines = [
    `Entroly plugin: ${first.name} failed (${outcome(first.result)})` +
      (cause ? `: ${cause}` : "") +
      "\n",
  ];
  for (const attempt of attempts) {
    lines.push(`  tried ${attempt.name}: ${outcome(attempt.result)}\n`);
  }
  lines.push(
    "  Run `entroly doctor` in a terminal to see what is wrong, then restart your agent host.\n",
  );
  return lines.join("");
}

const attempts = [];
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
  attempts.push({ name: candidateName(candidate), result });
}

process.stderr.write(failureReport(attempts));
process.exit(1);
