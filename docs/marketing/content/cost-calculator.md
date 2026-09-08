# Entroly Cost Estimation Framework

**How much can Entroly actually save you?** The honest answer: it depends on
your codebase, your agent, your model, and how much context is genuinely
unnecessary. Anyone who gives you a universal percentage is guessing.

Here's how to estimate **your own** numbers.

---

## Step 1: Know your current spend

Before estimating savings, understand your baseline:

```
monthly_input_token_spend = requests_per_day × avg_input_tokens × 30 × price_per_token
```

| Model | Input price (per 1M tokens) |
|---|---|
| GPT-4o | $2.50 |
| GPT-4o-mini | $0.15 |
| Claude 4 Sonnet | $3.00 |
| Claude 4 Opus | $15.00 |
| Gemini 2.5 Pro | $1.25 – $2.50 |

> These prices change. Check your provider's current pricing page.

### Example baseline

A team of 5 developers using Claude 4 Sonnet with coding agents:

```
  5 developers
× 50 requests/day each
× 80,000 avg input tokens per request
× 30 days
× $3.00 / 1M tokens
= $1,800/month in input tokens alone
```

---

## Step 2: Measure YOUR context profile

Run this on your actual repository — no API key needed:

```bash
pip install -U entroly
cd /path/to/your/repo
entroly simulate
```

This tells you:
- **Source tokens:** total context your agent would normally send
- **Selected tokens:** what Entroly would select under the default budget
- **Reduction:** the ratio between them

> **This is a local estimate, not a billing guarantee.** Provider-observed usage
> depends on cache hit rates, output tokens, retries, and model-specific
> tokenization. Use a proxy pilot for real numbers.

---

## Step 3: Estimate savings range

```
estimated_monthly_savings = monthly_input_token_spend × measured_reduction
```

### Using the example above

If `entroly simulate` shows a 60% context reduction on your repo:

```
$1,800/month × 0.60 = $1,080/month estimated input savings
```

### Additional savings levers (harder to estimate)

| Lever | How it helps | Measurement |
|---|---|---|
| **Cache alignment** | Keeps stable prompt prefixes → better provider cache hits | Provider cache-hit headers |
| **Session rescue** | Prevents context overflow → fewer retries and failed requests | Proxy `X-Entroly-Action` headers |
| **Model routing (RAVS)** | Routes simpler tasks to cheaper models | Dashboard routing breakdown |

These compound but are workload-specific. Don't add them to your estimate
without measuring them through a proxy pilot.

---

## Step 4: Run a real proxy pilot

For actual billing-grade numbers:

```bash
entroly proxy
```

Point your agent at `localhost:9377` and let it run for a few days. The
dashboard shows cumulative savings with real provider-observed token counts.

Use `ENTROLY_PRICING_FILE` to input your negotiated rates for accurate
dollar estimates.

---

## What this framework does NOT claim

- ❌ Universal savings percentage
- ❌ Guaranteed quality retention on every workload
- ❌ Output token savings (Entroly optimizes inputs)
- ❌ Provider cache hit guarantees
- ❌ Zero overhead on every request

## What it DOES provide

- ✅ Your repo-specific context reduction via `simulate`
- ✅ Your actual proxy-observed savings via `dashboard`
- ✅ Honest measurement boundaries
- ✅ A framework for team-level ROI estimation

---

*Run `entroly simulate` to get your own numbers. That's the only way to
know for your workload.*

*Repository: [github.com/juyterman1000/entroly](https://github.com/juyterman1000/entroly) · Apache-2.0*
