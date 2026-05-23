#!/usr/bin/env node

const { spawnSync } = require("child_process");

const target = require.resolve("entroly-wasm/bin/entroly-wasm.js");
const result = spawnSync(process.execPath, [target, ...process.argv.slice(2)], {
  stdio: "inherit",
});

if (result.error) {
  console.error(result.error.message);
  process.exit(1);
}

process.exit(result.status === null ? 1 : result.status);
