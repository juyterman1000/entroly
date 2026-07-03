# Developer Directories & MCP Registry Submissions

To increase search engine authority (backlinks) and user adoption, submit Entroly to high-traffic registries and directories. This document lists targets, URLs, submission steps, and copy-paste templates.

---

## 1. MCP Server Registries

Since Entroly operates as an MCP server (`entroly serve`), it qualifies for inclusion in directories cataloging MCP integrations.

### A. Glama MCP Registry
- **URL:** [glama.ai/mcp/servers](https://glama.ai/mcp/servers)
- **Action:** Click "Submit Server" / login with GitHub.
- **Form Values:**
  - **Name:** `Entroly`
  - **Category:** `Developer Tools` / `Context Engineering`
  - **GitHub URL:** `https://github.com/juyterman1000/entroly`
  - **MCP Command:** `npx -y @juyterman1000/entroly-mcp` or `pip install entroly && entroly serve`
  - **Short Description:** `Local context engineering & prompt compression proxy for AI coding agents. Reduces input tokens by 70–95% while preserving KV cache alignment and validating answers locally.`
  - **Details:** `An open-source (Apache-2.0) local gateway and MCP server that optimizes prompt context for Cursor, Cline, and Claude Code. Features a prompt prefix Cache Aligner, Content-Compressed Retrieval (CCR) handles, and WITNESS—a local Natural Language Inference (NLI) verifier that checks model output against source files.`

### B. Awesome MCP Servers (GitHub List)
- **URL:** [github.com/punkpeye/awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers)
- **Action:** Fork the repository and open a Pull Request.
- **Section:** `Developer Tools` or `Utilities`
- **Markdown Line to Insert:**
  ```markdown
  * [Entroly](https://github.com/juyterman1000/entroly) - A local context-engineering and prompt compression gateway that stabilizes prompt prefixes for KV caching discounts, generates exact-recovery CCR handles, and audits model responses locally.
  ```

---

## 2. Open-Source Comparison & Alternative Directories

### A. AlternativeTo
- **URL:** [alternativeto.net/software/entroly/](https://alternativeto.net)
- **Action:** Log in and select "Add an application".
- **Form Values:**
  - **Title:** `Entroly`
  - **Official Website:** `https://juyterman1000.github.io/entroly/`
  - **GitHub Repository:** `https://github.com/juyterman1000/entroly`
  - **License:** `Open Source (Apache-2.0)`
  - **Platform:** `Mac`, `Windows`, `Linux`
  - **Short Tagline:** `Local context compression & verification gateway for AI coding agents.`
  - **Alternatives to:** List `Headroom AI`, `LeanCTX`.
  - **Description:** 
    ```text
    Entroly is an open-source, local-first context engineering gateway for developer AI workflows. It sits as a local proxy or MCP server between your coding client (such as Cursor, Claude Code, Cline, or Aider) and your LLM provider.

    Key Features:
    - Token Savings: Slashes input tokens by 70-95% on large codebases.
    - Cache Aligner: Stabilizes prompt prefixes to secure Anthropic's 90% and OpenAI's 50% caching discounts.
    - Recoverable Retrieval: Uses CCR handles to let models query full-resolution code fragments if needed.
    - Zero Hallucinations: Includes a local, deterministic NLI verifier (WITNESS) that checks model claims against repository evidence at $0 cost.
    ```

### B. LibHunt (Python Comparisons)
- **URL:** [py.libhunt.com](https://py.libhunt.com)
- **Action:** LibHunt automatically indexes repositories, but you can suggest comparisons or claim the project page once indexed.
- **Suggested Comparison Title:** `Entroly vs Lean-CTX` or `Entroly vs Headroom`.

---

## 3. General AI Developer Platforms

### A. There's An AI For That (TAAFT)
- **URL:** [theresanaiforthat.com](https://theresanaiforthat.com)
- **Action:** Submit tool via their standard submission form.
- **Title:** `Entroly`
- **Category:** `Developer Tools`, `LLM cost optimization`
- **Pricing:** `Free & Open Source`

### B. Toolify.ai
- **URL:** [toolify.ai](https://www.toolify.ai)
- **Action:** Click "Submit Tool".
- **Short Info:** `A local context-engineering tool that slashes AI coding costs by stabilizing prompt caching and compressing repository assets.`
