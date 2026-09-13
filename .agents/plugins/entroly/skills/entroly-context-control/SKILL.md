---
name: entroly-context-control
description: "Surgically select, compress, and recover codebase context using Entroly's MCP tools. Slashes token usage by up to 95% with Merkle-exact recovery receipts and WITNESS verification."
---

# Entroly Context Intelligence Skill

Use this skill to optimize context, query the codebase with token budgets, compress bulky tool outputs, and recover omitted code byte-for-byte using the Entroly MCP server.

## Core Workflows

### 1. Token-Budgeted Codebase Selection (`optimize_context`)
When answering a complex question or preparing a refactor across multiple files:
- Call `optimize_context` with your natural language query and a token budget (e.g., `budget=4000` or `budget=8000`).
- Entroly executes a knapsack optimization algorithm that ranks AST fragments by relevance, entropy, and dependency relations.
- Returns surgical code fragments, line numbers, and a cryptographic Merkle receipt.

### 2. Semantic Code Recall (`recall`)
When searching for specific symbols, implementations, or interfaces:
- Call `recall(query="...")` with `top_k=5` or `top_k=10`.
- Returns ranked pointers with filenames, line numbers, and snippet digests.

### 3. Reversible Output Compression (`compress` & `recover`)
When dealing with large terminal dumps (build traces, test logs):
- Call `compress(data=..., budget=...)` to distill down critical errors and signals while stripping noise.
- Every compression produces a content-addressed SHA-256 handle.
- To restore the original uncompressed text at any time, call `recover(handle="...")`.

### 4. Codebase Health & Architecture Diagnostics (`health`)
To assess codebase quality before or after major changes:
- Call `health()` to inspect clone pairs, dead symbols, god files, and architecture violations.

### 5. Grounding & Hallucination Verification (`witness_verify`)
To verify an agent claim or proposed code against source evidence:
- Call `witness_verify(claim=..., evidence=...)` for local NLI-backed verification.
