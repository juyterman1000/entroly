# Entroly Cookbook

Concrete recipes. Each one is a single, copyable command path that solves a real
problem. Skip the architecture; come back to [docs/DETAILS.md](../docs/DETAILS.md)
when you want the deep dive.

| # | Recipe | One-liner |
|---|---|---|
| 1 | [Cut your Claude Code bill in half](#1-cut-your-claude-code-bill-in-half) | `entroly wrap claude` |
| 2 | [Drop the proxy in front of any LLM app](#2-drop-the-proxy-in-front-of-any-llm-app) | `entroly proxy` + base-URL env var |
| 3 | [Compress in your own code, two lines](#3-compress-in-your-own-code-two-lines) | `from entroly import compress` |
| 4 | [Route easy tasks to a cheaper model — automatically](#4-route-easy-tasks-to-a-cheaper-model--automatically) | `entroly ravs` |
| 5 | [Audit your codebase, get a grade](#5-audit-your-codebase-get-a-grade) | `entroly health` |
| 6 | [Weekly digest of what you saved](#6-weekly-digest-of-what-you-saved) | `entroly digest` |
| 7 | [Gate PRs in CI by token budget](#7-gate-prs-in-ci-by-token-budget) | `entroly batch` in GitHub Actions |
| 8 | [Share a teammate-ready context pack](#8-share-a-teammate-ready-context-pack) | `entroly share` |
| 9 | [Strip output filler too](#9-strip-output-filler-too) | `ENTROLY_DISTILL_MODE=ultra` |
| 10 | [Opt into federated learning](#10-opt-into-federated-learning) | `ENTROLY_FEDERATION=1` |

---

## 1. Cut your Claude Code bill in half

You're using Claude Code. Every prompt sends ~180K tokens. Most are duplicated
boilerplate. Wrap it once and forget.

```bash
cd /path/to/your/repo
entroly wrap claude
```

What happens:
- Entroly starts its proxy on `localhost:9377`
- Sets `ANTHROPIC_BASE_URL=http://localhost:9377` for the child process
- Launches `claude` with that env
- Auto-opens the dashboard at `http://localhost:9378` so you can watch
  the cost-saved counter climb

When you're done, close the terminal. Claude on its own (without `entroly wrap`)
goes back to direct billing.

> **Verify the savings:** open `http://localhost:9378` after a few prompts —
> the hero shows real `$X.XX` saved against your actual proxy traffic, not a
> projection.

---

## 2. Drop the proxy in front of any LLM app

You don't have to use Claude Code or Cursor. Anything that hits the OpenAI or
Anthropic APIs works:

```bash
entroly proxy --port 9377
```

Then point your app at the proxy:

```bash
export ANTHROPIC_BASE_URL=http://localhost:9377
export OPENAI_BASE_URL=http://localhost:9377/v1
export GEMINI_BASE_URL=http://localhost:9377/v1beta
your-app
```

Zero code changes. Works with LangChain, LlamaIndex, raw `requests`, the
official SDKs, anything that respects the standard base-URL env vars.

---

## 3. Compress in your own code, two lines

When you control the prompt and want explicit compression rather than a proxy:

```python
from entroly import compress, compress_messages

# Single string (code, JSON, logs, prose)
compressed = compress(api_response, budget=2000)

# Or a full chat history
trimmed = compress_messages(messages, budget=30000)
```

`compress` accepts any text and a token budget. `compress_messages` accepts a
list of `{"role": ..., "content": ...}` dicts and preserves role boundaries.
Both are deterministic — same input + same budget → bit-identical output (run
`python bench/trust_bench.py` to verify the SHA-256 check yourself).

---

## 4. Route easy tasks to a cheaper model — automatically

Most coding sessions mix hard reasoning with simple ops (running tests, reading
a file, formatting). Paying Opus prices for `pytest` is waste. RAVS watches
your outcomes, builds a confidence model per task type, and starts routing the
safe ones to Haiku — only after the data says it's safe.

Enable it once — add this `PostToolUse` hook to your `.claude/settings.json`:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Bash|Read|Grep|Glob|Edit|Write|TodoWrite",
        "hooks": [
          { "type": "command", "command": "entroly ravs capture --stdin --quiet 2>/dev/null || true" }
        ]
      }
    ]
  }
}
```

> The `2>/dev/null || true` tail is intentional: a missing entroly binary or a
> slow capture should never block Claude Code. Failure is silent and harmless.
> The entroly repo's own [`.claude/settings.json`](../.claude/settings.json)
> ships this hook — copy it as a reference.

Then use Claude Code normally for a week. After enough observations:

```bash
entroly ravs report
# or for the last 7 days
entroly ravs report --since 7d
```

The report shows which task types it now routes (with the actual sample sizes
and CI lower bounds it used to decide). Nothing routes until the math passes.
If a routed task ever fails, RAVS auto-reverts that category to the flagship
model.

---

## 5. Audit your codebase, get a grade

Quick A–F grade with the actionable issues called out:

```bash
cd /path/to/your/repo
entroly health
```

Outputs grade, top SAST findings, dead code candidates, clone hotspots, and a
per-file health breakdown. Useful before a refactor or as a one-off review.

---

## 6. Weekly digest of what you saved

Once you've been using the proxy for a few days:

```bash
entroly digest
```

Prints a one-screen summary: tokens saved, dollars saved, request count,
top-saving file paths, biggest wins. Designed to be screenshotted and pasted
into Slack.

---

## 7. Gate PRs in CI by token budget

Catch context-window blowups before they merge. In `.github/workflows/ci.yml`:

```yaml
- name: Entroly token budget check
  run: |
    pip install entroly
    echo "Explain how authentication works" | \
      entroly batch --budget 8000 --fail-over-budget
```

Or use the published action:

```yaml
- uses: juyterman1000/entroly-cost-check-@v1
  with:
    budget: 8000
```

Fails the build if the optimized context for a representative query exceeds
your budget, so a careless `import *` or a doc dump can't silently bloat your
production AI bill.

---

## 8. Share a teammate-ready context pack

You found the right context for a task. You want your teammate to start where
you ended:

```bash
entroly share --task "implement OAuth refresh flow" --out oauth-context.json
```

Generates a portable Context Report Card with the file list, ranking rationale,
and SHA-256 integrity. Your teammate runs:

```bash
entroly import oauth-context.json
```

Their proxy now prioritizes the same files for that task.

---

## 9. Strip output filler too

LLM responses leak ~40% filler ("Sure! I'd be happy to help. Let me take a look...").
Code blocks are never touched. Three intensities:

```bash
export ENTROLY_DISTILL_MODE=lite    # gentle — strip greetings + sign-offs
export ENTROLY_DISTILL_MODE=full    # default — strip filler and hedging
export ENTROLY_DISTILL_MODE=ultra   # aggressive — answer-only mode
entroly proxy
```

Or disable entirely:

```bash
export ENTROLY_DISTILL=0
```

---

## 10. Opt into federated learning

Off by default. Your code never leaves your machine — only optimization weights,
noise-protected and anonymous, shared via GitHub. The benefit: every other
opted-in instance's discoveries land in your weights overnight.

```bash
export ENTROLY_FEDERATION=1
entroly daemon
```

Disable any time:

```bash
unset ENTROLY_FEDERATION
```

The daemon's `vault/evolution/` directory shows exactly which patterns came
from federation versus your local learning loop, with provenance.

---

## Want a recipe that isn't here?

Open an issue: [github.com/juyterman1000/entroly/issues](https://github.com/juyterman1000/entroly/issues)
or drop it in [Discussions](https://github.com/juyterman1000/entroly/discussions).
The most-requested recipes get added to this cookbook.
