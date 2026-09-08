# Entroly in 5 Minutes: See What Your AI Agent Actually Needs

**Your AI coding agent is reading your entire codebase every time you ask it a question.** Most of that context is noise — files and code the model never needed to see. You're paying for those tokens, and the model gets distracted by them.

Entroly fixes this. Here's how to see it work on **your own code** in under 5 minutes. No API key, no account, nothing to configure.

---

## Step 1: Install (30 seconds)

```bash
pip install -U entroly
```

Or, if you prefer npm:

```bash
npm install -g entroly
```

That's it. Nothing runs in the background. No daemon, no signup.

## Step 2: Verify the install does what it claims (60 seconds)

```bash
cd /path/to/your/project
entroly verify-claims
```

This runs bounded local checks — compression, receipts, recovery, routing — all on your machine, no API key. You'll see each check pass or fail with evidence.

**What you'll see:** A terminal report showing 12+ verification checks with pass/fail status. Every check is reproducible and self-contained.

## Step 3: See your numbers (60 seconds)

```bash
entroly simulate
```

This estimates how much of your project's context is actually needed versus how much is noise. It runs entirely locally.

**What you'll see:** Source tokens vs. selected tokens for your specific repository. These are *your* numbers, not a generic benchmark.

> **Important:** The numbers from `simulate` are local estimates, not a billing guarantee. Provider-observed usage depends on your model, cache behavior, and request patterns. Run a proxy pilot to measure real savings.

## Step 4: Connect to your editor (60 seconds)

```bash
entroly go
```

This auto-detects your editor (Claude Code, Cursor, VS Code, etc.), configures MCP or proxy integration, and opens a dashboard showing before/after token counts.

**Already using Claude Code?** You can also attach directly:

```bash
entroly attach create --client claude --project . --ttl 4h --install
```

This creates a scoped MCP attachment that expires on its own. Zero code changes.

## Step 5: See the dashboard (30 seconds)

After `entroly go`, your browser opens a dashboard showing:

- **Source tokens** — what your agent would normally send
- **Selected tokens** — what Entroly actually sends
- **Recovery status** — everything omitted is kept and recoverable
- **Receipt** — an audit trail of what was included and why

---

## What just happened?

In 5 minutes you've:

1. ✅ **Installed** a local-first context layer (nothing leaves your machine for analysis)
2. ✅ **Verified** the install does what it claims with reproducible checks
3. ✅ **Measured** your own repo's context profile — no synthetic benchmark
4. ✅ **Connected** it to your editor with zero code changes
5. ✅ **Seen** real token selection on your actual codebase

---

## Next steps

| What you want to do | How |
|---|---|
| Use the Python SDK | `from entroly import compress, optimize` |
| Run the proxy for any agent | `entroly proxy` → point `OPENAI_BASE_URL` at `localhost:9377` |
| Check model claims against evidence | Built-in WITNESS verifier (local, no API) |
| See the full command reference | `entroly --help` or [docs](https://juyterman1000.github.io/entroly/docs/index.html) |
| Report a problem | `entroly doctor` → [GitHub Issues](https://github.com/juyterman1000/entroly/issues) |

---

## Honest limitations

- **Small repos** that already fit the context window may see zero improvement — Entroly passes them through unchanged and tells you.
- **Compression trades quality for size** — the SQuAD benchmark shows 80% → 72% accuracy with aggressive compression. Use `simulate` to see your specific trade-off.
- **Proxy mode still sends to your provider** — Entroly reduces what's sent, it doesn't add a new destination.

---

*Entroly is Apache-2.0 and local-first. Repository: [github.com/juyterman1000/entroly](https://github.com/juyterman1000/entroly)*
