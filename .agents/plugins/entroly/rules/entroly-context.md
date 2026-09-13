# Entroly Context Intelligence Rules

When the Entroly plugin is active in Antigravity:

1. **Prioritize Token-Budgeted Selection**:
   When gathering context across multiple files or large codebases, prefer Entroly's `optimize_context` or `recall` tools to extract the exact answer-bearing AST fragments within your token budget instead of dumping whole directories into prompts.

2. **Preserve Cryptographic Merkle Receipts**:
   Every compression or selection emits a cryptographic receipt (`sha256:...`). Retain this identifier so any omitted context can be retrieved byte-for-byte on demand with `recover`.

3. **Grounded Fact Verification (WITNESS)**:
   Use `witness_verify` to check generated assumptions, code changes, and assertions against indexed evidence before applying breaking changes.

4. **Cross-Agent Shared Knowledge**:
   Consult `shared_memory_search` for architectural decisions and cross-agent insights recorded across sessions, and record key discoveries with `shared_memory_write`.
