# Entroly for OpenCode

Entroly adds a local Context Assurance layer to OpenCode without replacing its
models, agents, tools, or session store.

The integration has two parts:

1. an Entroly MCP server for context selection, receipts, verification, and exact
   recovery;
2. an OpenCode compaction hook that preserves evidence-critical state and every
   visible `ccr:<24-hex>` recovery handle.

## Install

```bash
python -m pip install -U entroly
```

Copy the sample configuration into the project root:

```bash
cp integrations/opencode/opencode.jsonc ./opencode.jsonc
mkdir -p .opencode/plugins
cp integrations/opencode/.opencode/plugins/entroly-context-assurance.ts \
  .opencode/plugins/entroly-context-assurance.ts
```

OpenCode automatically loads project plugins from `.opencode/plugins/`. The
sample `opencode.jsonc` starts Entroly locally with:

```jsonc
{
  "mcp": {
    "entroly": {
      "type": "local",
      "command": ["entroly", "serve"],
      "enabled": true,
      "environment": { "ENTROLY_NO_DOCKER": "1" }
    }
  }
}
```

OpenCode prefixes MCP tools with the server name, so Entroly's
`entroly_retrieve` tool is exposed as `entroly_entroly_retrieve`.

## Exact-recovery contract

Discovery and recovery are separate:

- use `entroly_recall_relevant`, `entroly_optimize_context`, or normal OpenCode
  search tools to discover which evidence matters;
- use `entroly_entroly_retrieve` only after Entroly has emitted a visible
  `ccr:<24-hex>` handle;
- pass that exact handle unchanged;
- do not add a natural-language query or substitute a source path;
- recovery returns the complete stored original for that content hash.

A missing handle remains a miss. Entroly does not silently replace an evicted
historical version with the latest file contents.

## Permissions

The sample uses:

```jsonc
{
  "permission": {
    "entroly_*": "ask"
  }
}
```

Teams may allow read-only Entroly tools after reviewing their local policy. Keep
state-changing tools approval-gated when using autonomous OpenCode agents.

## Compaction behavior

The plugin replaces OpenCode's compaction prompt with an evidence-preserving
contract. It requires the compactor to retain:

- exact paths, symbols, hashes, versions, commands, and errors;
- the active goal and user constraints;
- decisions, rejected hypotheses, and remaining work;
- verification results and explicit unknowns;
- every `ccr:` handle byte-for-byte.

The hook does not call a model or provider itself. OpenCode continues to own
provider selection and compaction execution.
