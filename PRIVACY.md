# Privacy Policy

**Entroly has no hosted backend and does not phone home.** Local indexing,
selection, dashboards, deterministic verification, and learning run on your
machine. If you configure Entroly as a proxy for a cloud LLM API, the selected
prompt content is still forwarded to the provider you chose, under that
provider's terms and data-handling policy.

## What Entroly Does

Entroly is a **local reverse proxy** that sits between your IDE and your chosen
LLM API. It compresses prompts, tracks local value metrics, and checks responses
for hallucinations using local deterministic paths unless you explicitly enable
an optional cloud-backed feature.

## What Stays On Your Machine

| Component | What It Processes | External Calls |
|---|---|---|
| Indexing and token compression | Your local files and prompts | **None** |
| WITNESS/EICV deterministic verification | LLM responses and selected context | **None** |
| RAVS observation and learning | Local request/outcome metadata | **None** |
| Learning loop and autotune | Optimization weights and benchmark cases | **None** |
| Belief vault | Learned code facts | **None** |
| Dashboard | Usage statistics | **None** (localhost only) |

## What Goes To Your LLM API

Entroly forwards your optimized prompt to the LLM API **you configured** through
the proxy base URL. This is the API you already selected; Entroly changes the
request shape and context size, not the provider's data policy.

The gateway control plane does not automatically move a request to a different
provider. Its executable plans are restricted to models belonging to the
original provider. A source-provider failure fails closed instead of selecting
another data recipient. See
[`docs/gateway-provider-boundary.md`](docs/gateway-provider-boundary.md) for the
code-backed boundary and test evidence.

```text
Without Entroly:  IDE -> OpenAI (50K tokens)
With Entroly:     IDE -> localhost:9377 -> compress -> OpenAI (12K tokens)
```

Entroly is designed to send less context to that API than the IDE would alone.
Review the configured provider's terms before sending confidential, regulated,
or customer data.

## Zero Phone-Home

- No Entroly-owned servers are ever contacted.
- No analytics, telemetry, or crash reporting SDKs are imported.
- No Entroly usage tracking is sent anywhere.
- `grep -r "entroly\.(com|io|dev|cloud)" entroly/` returns zero matches.

The CLI performs a best-effort version check against PyPI at most once every
24 hours. This is a normal HTTPS request to PyPI, not to an Entroly-operated
server. Failures are ignored. Set `ENTROLY_DISABLE_UPDATE_CHECK=1` to disable
the request. The proxy does not probe LLM provider hosts at startup unless you
explicitly set `ENTROLY_CHECK_UPSTREAM=1`.

When the native engine (`entroly-core`) is missing, commands that measure
context — `simulate`, `demo`, `perf`, and the MCP/proxy services at startup —
install it from PyPI before running. Without it, context selection cannot read
your query, so any reported saving would be budget arithmetic rather than a
measured result. This is an ordinary package install from PyPI, not an
Entroly-operated server: no code, prompts, file contents, queries, or telemetry
are transmitted. It is skipped entirely when the engine is already present, it
never runs from a library import path, and `ENTROLY_NO_SELF_HEAL=1` disables it —
in which case Entroly reports the figure explicitly labelled as unearned.

`entroly telemetry on` stores a local preference only. No outbound telemetry
uploader is included in this release.

## Optional Features (OFF by Default)

### WITNESS NLI (`--witness-nli` or `ENTROLY_WITNESS_NLI=1`)

Uses OpenAI as an optional entailment checker when `OPENAI_API_KEY` is present.
This sends the verification prompt, selected evidence, and extracted claims to
OpenAI. If the call fails or the option is not enabled, WITNESS falls back to
local deterministic verification.

### RAVS Model Routing (`ENTROLY_RAVS_ROUTER=1`)

Can substitute a lower-cost model after local evidence gates pass. It is off by
default because changing models can also change capabilities and cost. The live
proxy restricts this substitution to the same provider; it does not use RAVS as
an automatic cross-provider fallback.

### Federation (`ENTROLY_FEDERATION=1`)

Shares differential-privacy-noised weight vectors (numerical arrays) with other
Entroly users via file exchange or GitHub Issues. Contribution payloads do not
include prompts, code, or file paths. They include a random persisted client ID
and the noised weights. GitHub transport requests are visible to GitHub. If you
use the `ENTROLY_GITHUB_TOKEN` personal-token fallback instead of a shared
`ENTROLY_FEDERATION_BOT`, the posting GitHub account is visible on the issue.
Protected by Renyi Differential Privacy (epsilon=1.0) with a finite privacy
budget.

### Integration Gateways (Slack/Discord/Telegram)

Slack and Discord use user-configured webhooks for alerts. Telegram uses the
Telegram Bot API for outbound notifications and bot-command interactions.
These integrations are disabled unless you configure the corresponding
webhook URL or Telegram bot token and chat ID.

### Promoted Skill Execution (`ENTROLY_EXECUTE_PROMOTED_SKILLS=1`)

Generated skill tools are stored in the local writable vault. Entroly does not
execute promoted `tool.py` files automatically unless you explicitly enable
this setting. Enabled tools run as local subprocesses with a timeout; treat the
vault as executable local state and review imported or manually edited tools.

## Local Storage

All data is stored in `.entroly/` in your project directory:

```text
.entroly/
|-- vault/           # Learned code facts
|-- weights.json     # Optimization weights
|-- dashboard.db     # Request statistics
`-- certificates/    # Hallucination proof certificates
```

Delete `.entroly/` to remove all stored data.

## Verification

Run `entroly doctor --privacy` to verify that no Entroly-owned external servers
or analytics SDKs are used. If you configure a cloud LLM API, that provider
still receives the optimized prompt content you send through the proxy.

## Secret Detection

Entroly automatically detects and redacts secrets (API keys, passwords, tokens)
from internal logs. Patterns detected:

- OpenAI keys (`sk-...`)
- GitHub PATs (`ghp_...`)
- AWS access keys (`AKIA...`)
- Password/secret assignments

## Contact

If you have privacy concerns, please open an issue at
[github.com/juyterman1000/entroly](https://github.com/juyterman1000/entroly/issues).
