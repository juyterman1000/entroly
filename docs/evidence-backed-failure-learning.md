# Evidence-backed agent failure learning

`entroly learn` can correlate a failed operation with a later successful operation in local JSONL transcripts. Analysis never edits an instruction file.

```bash
entroly learn --transcript session.jsonl --output proposal.json
```

The proposal records transcript SHA-256 digests, event line numbers, bounded redacted excerpts, failure and success event hashes, and an explicit limitation: sequence for the same normalized operation does not prove causality. Failures without a later observed success produce no correction.

Review the JSON proposal before application. Applying it is a separate explicit command and requires a named target:

```bash
entroly learn --apply-proposal proposal.json --target AGENTS.md
```

Before writing, Entroly re-hashes every transcript source. It refuses changed evidence, backs up the target, and appends a proposal-ID marker block. Direct `entroly learn --apply` mutation is retired. Entroly does not silently discover and rewrite root instructions.

Supported input is generic line-delimited JSON. Events need an identifiable operation (`tool_name`, `tool`, `name`, or command) and explicit outcome evidence such as `exit_code`, `status`, `is_error`, or `success`. Provider-specific session adapters can normalize their records into this contract without changing the evidence model.
