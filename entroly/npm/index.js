#!/usr/bin/env node

const { spawnSync } = require('child_process');

// Pass all arguments straight through to the Python MCP server
const args = process.argv.slice(2);

// Launch directly so environments without `which` or `where` still work.
const pyProcess = spawnSync('entroly', args, { stdio: 'inherit' });

if (pyProcess.error && pyProcess.error.code === 'ENOENT') {
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

if (pyProcess.error) {
  console.error(`Error starting 'entroly': ${pyProcess.error.message}`);
  process.exit(1);
}

process.exit(pyProcess.status !== null ? pyProcess.status : 1);
