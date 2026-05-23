/**
 * Chat gateways — Telegram, Discord, Slack.
 *
 * Zero hard dependencies. Uses native fetch (Node ≥18).
 * Attach a source with `.stats()` method (e.g. an EvolutionDaemon or
 * any object exposing skills_promoted / skills_pruned / structural_successes /
 * dream_cycles counters) and the gateway will surface deltas.
 *
 * Env:
 *   ENTROLY_TG_TOKEN + ENTROLY_TG_CHAT_ID   (Telegram, 2-way)
 *   ENTROLY_DISCORD_WEBHOOK                 (Discord, webhook)
 *   ENTROLY_SLACK_WEBHOOK                   (Slack, webhook)
 */

'use strict';

class _BaseGateway {
  constructor(pollIntervalS = 30) {
    this._pollS = pollIntervalS;
    this._source = null;
    this._timer = null;
    this._last = {};
  }

  attach(source) { this._source = source; return this; }

  start() {
    if (this._timer) return;
    this.send(this._onlineMsg()).catch(() => {});
    this._timer = setInterval(() => this._tick().catch(() => {}), this._pollS * 1000);
    this._timer.unref && this._timer.unref();
  }

  stop() {
    if (this._timer) { clearInterval(this._timer); this._timer = null; }
  }

  async _tick() {
    if (!this._source) return;
    const now = this._source.stats();
    await this._surfaceDelta(this._last, now);
    this._last = JSON.parse(JSON.stringify(now));
  }

  async _surfaceDelta(prev, now) {
    const diff = k => (Number(now[k]) || 0) - (Number(prev[k]) || 0);
    if (diff('skills_promoted') > 0) await this.send(this._fmt('promoted', diff('skills_promoted'), now.skills_promoted));
    if (diff('skills_pruned') > 0)   await this.send(this._fmt('pruned',   diff('skills_pruned')));
    if (diff('structural_successes') > 0) await this.send(this._fmt('structural', diff('structural_successes')));
    if (diff('dream_cycles') > 0)    await this.send(this._fmt('dream',    diff('dream_cycles')));
  }

  // overridden by subclasses
  _onlineMsg() { return 'Entroly gateway online.'; }
  _fmt() { return ''; }
  async send() { throw new Error('send() not implemented'); }
}

// ── Telegram ──────────────────────────────────────────────────

class TelegramGateway extends _BaseGateway {
  constructor({ token, chatId, pollIntervalS = 30 } = {}) {
    super(pollIntervalS);
    this._token = token;
    this._chatId = String(chatId);
  }

  _onlineMsg() { return '🧬 *Entroly gateway online* — watching the self-evolution loop.'; }

  _fmt(kind, delta, total) {
    if (kind === 'promoted')   return `✅ *Skill promoted* — +${delta}. Total: ${total}.`;
    if (kind === 'pruned')     return `🗑️ *Skill pruned* — +${delta}.`;
    if (kind === 'structural') return `🧠 *Structural synthesis* — +${delta} ($0, deterministic).`;
    if (kind === 'dream')      return `💭 *Dream cycle complete* — +${delta}.`;
    return '';
  }

  async send(text) {
    const url = `https://api.telegram.org/bot${this._token}/sendMessage`;
    const body = new URLSearchParams({
      chat_id: this._chatId, text, parse_mode: 'Markdown',
      disable_web_page_preview: 'true',
    });
    const r = await fetch(url, { method: 'POST', body });
    return { ok: r.ok, status: r.status };
  }
}

// ── Discord ───────────────────────────────────────────────────

class DiscordGateway extends _BaseGateway {
  constructor({ webhookUrl, pollIntervalS = 30, username = 'Entroly' } = {}) {
    super(pollIntervalS);
    this._url = webhookUrl;
    this._username = username;
  }

  _onlineMsg() { return '🧬 **Entroly gateway online** — watching the self-evolution loop.'; }

  _fmt(kind, delta, total) {
    if (kind === 'promoted')   return `✅ **Skill promoted** — +${delta}. Total: ${total}.`;
    if (kind === 'pruned')     return `🗑️ **Skill pruned** — +${delta}.`;
    if (kind === 'structural') return `🧠 **Structural synthesis** — +${delta} ($0, deterministic).`;
    if (kind === 'dream')      return `💭 **Dream cycle complete** — +${delta}.`;
    return '';
  }

  async send(content) {
    const r = await fetch(this._url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: this._username, content }),
    });
    return { ok: r.ok, status: r.status };
  }
}

// ── Slack ─────────────────────────────────────────────────────

class SlackGateway extends _BaseGateway {
  constructor({ webhookUrl, pollIntervalS = 30 } = {}) {
    super(pollIntervalS);
    this._url = webhookUrl;
  }

  _onlineMsg() { return ':dna: *Entroly gateway online* — watching the self-evolution loop.'; }

  _fmt(kind, delta, total) {
    if (kind === 'promoted')   return `:white_check_mark: *Skill promoted* — +${delta}. Total: ${total}.`;
    if (kind === 'pruned')     return `:wastebasket: *Skill pruned* — +${delta}.`;
    if (kind === 'structural') return `:brain: *Structural synthesis* — +${delta} ($0, deterministic).`;
    if (kind === 'dream')      return `:thought_balloon: *Dream cycle complete* — +${delta}.`;
    return '';
  }

  async send(text) {
    const r = await fetch(this._url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    return { ok: r.ok, status: r.status };
  }
}

module.exports = { TelegramGateway, DiscordGateway, SlackGateway };

// Standalone entrypoint: picks a gateway based on env vars.
if (require.main === module) {
  (async () => {
    const tg = process.env.ENTROLY_TG_TOKEN && process.env.ENTROLY_TG_CHAT_ID
      ? new TelegramGateway({ token: process.env.ENTROLY_TG_TOKEN, chatId: process.env.ENTROLY_TG_CHAT_ID })
      : null;
    const dc = process.env.ENTROLY_DISCORD_WEBHOOK
      ? new DiscordGateway({ webhookUrl: process.env.ENTROLY_DISCORD_WEBHOOK })
      : null;
    const sl = process.env.ENTROLY_SLACK_WEBHOOK
      ? new SlackGateway({ webhookUrl: process.env.ENTROLY_SLACK_WEBHOOK })
      : null;

    const gws = [tg, dc, sl].filter(Boolean);
    if (!gws.length) {
      console.error('Set ENTROLY_TG_TOKEN+ENTROLY_TG_CHAT_ID, ENTROLY_DISCORD_WEBHOOK, or ENTROLY_SLACK_WEBHOOK.');
      process.exit(2);
    }
    for (const gw of gws) gw.start();
    console.log(`Started ${gws.length} gateway(s). Ctrl+C to exit.`);
    process.stdin.resume();
  })().catch(err => { console.error(err); process.exit(1); });
}
