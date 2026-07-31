#!/usr/bin/env node

const { spawnSync } = require('child_process');
const path = require('path');

// Pass all arguments straight through to the Python MCP server
const args = process.argv.slice(2);

// Check if 'entroly' command is available (installed via pip/pipx)
const checkCmd = process.platform === 'win32' ? 'where' : 'which';
const checkStatus = spawnSync(checkCmd, ['entroly'], { stdio: 'ignore' });

if (checkStatus.status !== 0) {
  console.error("=====================================================================");
  console.error("  Error: 'entroly' Python package is not installed.");
  console.error("");
  console.error("  The entroly-mcp package is a bridge. To use it,");
  console.error("  you must have the Python engine installed on your system.");
  console.error("");
  console.error("  Please run:");
  console.error("      pip install entroly");
  console.error("  or");
  console.error("      pipx install entroly");
  console.error("=====================================================================");
  process.exit(1);
}

// Spawn the Python process, replacing the current node process's stdio directly
const pyProcess = spawnSync('entroly', args, {
  stdio: 'inherit'
});

process.exit(pyProcess.status !== null ? pyProcess.status : 1);
