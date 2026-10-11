# Agent review workflow: master implementation prompt

Starting from verified remote `main`, build Entroly's first complete cross-agent review workflow without replacing its existing Work Graph, handoff, receipt, or verification engines.

1. Inventory current Work Graph, Python and WASM MCP servers, plugin launch paths, host hooks, skills, receipts, tests, and open PRs. Record the exact base SHA and avoid overlapping PR #515 recovery work.
2. Make the installed plugin route expose the required Work Graph operations with Python/WASM parity, a visible capability check, and correct repository identity. Fail clearly when the runtime cannot support the workflow.
3. Provide one clear writer-to-independent-reviewer flow: capture the task and current source state, create a graph-bound handoff, let a different agent review the exact code state, record findings with evidence and observed test outcomes, detect stale source, resolve findings, and resume. Reuse the existing Work Graph and receipts.
4. Update the bundled skill and activation instructions so a user can invoke the flow after installation. Keep MCP tool choice, hook execution, actual activation, and verified outcome distinct. Do not introduce surprise remote calls or publish private prompts.
5. Add focused regression and end-to-end tests, then run the packaging gates for every changed distribution surface. Exercise real Claude and Codex hosts if locally available; otherwise identify the exact boundary that remains unverified.
6. Preserve unrelated local work. Report risks, rollback, and any remaining enterprise storage, identity, or policy decisions without claiming those gaps are closed.

## Smallest useful user-facing improvement

After installing Entroly, a developer can ask one coding agent to hand off an exact code state to another model for review and receive a review record tied to the observed repository state.

## Affected surfaces

Work Graph MCP, Python/WASM parity, Codex and Claude plugin packaging, skill instructions, receipt integrity, tests, and documentation. Proxy billing, WITNESS/RAVS policy, and published package versions change only if required by the implementation.
