# Privacy Policy

**Entroly processes everything locally. Your prompts, code, and data never leave your machine through Entroly.**

## What Entroly Does

Entroly is a **local reverse proxy** that sits between your IDE and your chosen LLM API. It compresses your prompts (saving tokens and money) and checks responses for hallucinations — all using deterministic math on your local machine.

## What Stays On Your Machine

| Component | What It Processes | External Calls |
|---|---|---|
| Token compression | Your full prompts | **None** |
| Hallucination detection | LLM responses | **None** |
| Model routing | Request metadata | **None** |
| Learning loop | Optimization weights | **None** |
| Belief vault | Learned code facts | **None** |
| Dashboard | Usage statistics | **None** (localhost only) |

## What Goes To Your LLM API

Entroly forwards your (compressed) prompt to the LLM API **you configured** via `ENTROLY_TARGET_URL`. This is the API you're already using — Entroly just makes the request smaller.

```
Without Entroly:  IDE → OpenAI (50K tokens)
With Entroly:     IDE → localhost:9377 → compress → OpenAI (12K tokens)
```

Entroly sends **less** data to the API than your IDE would alone.

## Zero Phone-Home

- No Entroly-owned servers are ever contacted
- No analytics, telemetry, or crash reporting SDKs
- No usage tracking of any kind
- `grep -r "entroly\.(com|io|dev|cloud)" entroly/` returns zero matches

## Optional Features (OFF by Default)

### Federation (`ENTROLY_FEDERATION=1`)
Shares **only** differential-privacy-noised weight vectors (numerical arrays) with other Entroly users via file exchange or GitHub Issues. **Never shares prompts, code, file paths, or any identifiable information.** Protected by Rényi Differential Privacy (ε=1.0) with a finite privacy budget.

### Integration Gateways (Slack/Discord/Telegram)
User-configured webhooks for alerts. Only sends messages you explicitly configure. Disabled unless you set webhook URLs.

## Local Storage

All data is stored in `.entroly/` in your project directory:
```
.entroly/
├── vault/           # Learned code facts
├── weights.json     # Optimization weights
├── dashboard.db     # Request statistics
└── certificates/    # Hallucination proof certificates
```

Delete `.entroly/` to remove all stored data.

## Verification

Run `entroly doctor --privacy` to verify that no external connections are made beyond your configured LLM API.

## Secret Detection

Entroly automatically detects and redacts secrets (API keys, passwords, tokens) from internal logs. Patterns detected:
- OpenAI keys (`sk-...`)
- GitHub PATs (`ghp_...`)
- AWS access keys (`AKIA...`)
- Password/secret assignments

## Contact

If you have privacy concerns, please open an issue at [github.com/juyterman1000/entroly](https://github.com/juyterman1000/entroly/issues).
