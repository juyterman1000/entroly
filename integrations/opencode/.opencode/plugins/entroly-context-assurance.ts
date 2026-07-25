import type { Plugin } from "@opencode-ai/plugin"

const ASSURANCE_PROMPT = `You are compacting an OpenCode engineering session under Entroly Context Assurance.

Preserve exactly:
- the current user goal and every user-stated constraint;
- repository paths, symbols, commit hashes, versions, commands, exit codes, and error text;
- decisions already made, rejected hypotheses, and why they were rejected;
- files changed and files still requiring inspection;
- test, lint, build, and review results, including incomplete or blocked verification;
- every Entroly ccr:<24-hex> recovery handle byte-for-byte.

Never replace an exact value with a vague summary. Never claim a test, review, push, merge, or release occurred unless the session contains direct evidence. Never infer the content behind a ccr handle. When omitted exact content is needed, call the Entroly MCP tool entroly_entroly_retrieve with that exact handle. Recovery is hash-only and returns the complete original; do not add a query or source path.`

export const EntrolyContextAssurancePlugin: Plugin = async () => {
  return {
    "experimental.session.compacting": async (_input, output) => {
      const existing = typeof output.prompt === "string" ? output.prompt.trim() : ""
      output.prompt = existing
        ? `${ASSURANCE_PROMPT}\n\nHost compaction instructions:\n${existing}`
        : ASSURANCE_PROMPT
    },
  }
}
