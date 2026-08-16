<p align="center">
  <img src="../../docs/assets/entroly_wordmark.svg" width="820" alt="Entroly">
</p>

<h1 align="center">Entroly — Drop-In Kontextförsäkring för Lägre AI-Driftskostnader</h1>

<p align="center"><b>Minska onödig kontext utan att förlora kontrollen över kritiska bevis.</b><br>
Välj de mest värdefulla bevisen först, komprimera dem, håll originalen återställningsbara och utfärda ett kvitto — utan att skriva om din kodbas eller agentarkitektur.</p>
<p align="center">
  <sub>Entroly är ett local-first Context OS: innehållsadresserade bevis, återställningsbar komprimering och granskningsbara kvitton. Fungerar via proxy, MCP, insticksprogram, wrapper och SDK med Claude Code, Codex, OpenClaw, GitHub Copilot, Cursor, Aider och OpenAI/Anthropic-kompatibla appar.</sub>
</p>
<p align="center">
  <a href="https://pypi.org/project/entroly/"><img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI" alt="Entroly on PyPI"></a>
  <a href="https://www.npmjs.com/package/entroly"><img src="https://img.shields.io/npm/v/entroly?color=red&label=npm" alt="Entroly on npm"></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="Apache-2.0 license"></a>
  <a href="https://github.com/juyterman1000/entroly"><img src="https://img.shields.io/github/stars/juyterman1000/entroly?style=social" alt="Entroly GitHub stars"></a>
</p>

<p align="center"><b>100 438 nedladdningar · växer för varje dag</b><br>
<sub>Uppmätt över olika distributionskällor.</sub></p>

<p align="center"><b>⭐ Om Entroly är användbart för dig, ge projektet en stjärna på GitHub.</b><br>
<a href="https://github.com/juyterman1000/entroly">⭐ Ge Entroly en stjärna på GitHub</a> — det hjälper projektet att växa och nå fler utvecklare.</p>

<p align="center">
  <b><a href="../../README.md">English</a> · <a href="README.zh.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a> · <a href="README.es.md">Español</a> · <a href="README.hi.md">हिन्दी</a> · <a href="README.fr.md">Français</a> · <a href="README.de.md">Deutsch</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.it.md">Italiano</a> · <a href="README.tr.md">Türkçe</a> · <a href="README.vi.md">Tiếng Việt</a> · <a href="README.id.md">Bahasa Indonesia</a> · <a href="README.pl.md">Polski</a> · <a href="README.nl.md">Nederlands</a> · <a href="README.th.md">ไทย</a> · Svenska · <a href="README.cs.md">Čeština</a> · <a href="README.tl.md">Tagalog</a> · <a href="README.ro.md">Română</a></b>
</p>

---

## Vad är Entroly? (På enkel svenska)

AI-kodningsassistenter har en minnesgräns. Ger du dem hela din kodbas blir de långsamma, dyra och distraherade — som att ge någon en 500-sidig manual när de bara behövde sidan 47.

**Entroly hittar sidan 47.**

Det sitter mellan din kod och AI:n, läser allt och skickar endast vidare de delar som är relevanta för frågan som faktiskt ställs.
|  |  |
|---|---|
| 💰 **Din faktura minskar** | Mindre text till AI:n innebär en lägre faktura. |
| 🔍 **Inget går förlorat** | Allt Entroly lägger åt sidan bevaras och kan återställas *exakt* som det var. |
| 🧾 **Du kan kontrollera arbetet** | Varje beslut kommer med ett kvitto: vad som behölls, vad som uteslöts och varför. |

**Måste jag ändra min kod?** Nej. Entroly fungerar med de verktyg du redan använder — Claude Code, Cursor, Copilot och 30+ andra.

---
## Installation

| Plattform | Installation | Vad du får |
|---|---|---|
| 🐍 **Python** (pip) — *rekommenderas* | `pip install -U entroly` | Allt: CLI-verktyg, AI-editorserver, kodbibliotek |
| 📦 **Node / npm** | `npm install -g entroly` | Samma motor, kräver inte Python |
| 🦀 **Rust** (källkodsbygge) | `cd entroly-core && cargo build --release --bin entroly-rs --features proxy` | Ett fristående program |
| 🍺 **Homebrew** | `brew install juyterman1000/entroly/entroly` | CLI-verktyg på macOS/Linux |
| 🐳 **Docker** | `docker pull ghcr.io/juyterman1000/entroly:latest` | Körs i en container |

```bash
cd /your/repo
entroly verify-claims
entroly simulate
```

---
## Snabbstart

> `pip install -U entroly && entroly go` — det är allt.

```python
from entroly import compress, compress_messages, optimize
compressed = compress(api_response, budget=2000)
messages   = compress_messages(messages, budget=30000)
context    = optimize(fragments, budget=8000, query="fixa inloggningsbuggen")
```

---
## Riktmärken (Benchmarks)

| Riktmärke | Baslinje | Med Entroly | Bevarande | Tokenbesparing |
|---|---|---|---|---|
| NeedleInAHaystack | 100% | 100% | **100%** | **99.5%** |
| LongBench (HotpotQA) | 64% | 66% | **103%** | **85.3%** |
| Berkeley Function Calling | 100% | 100% | **100%** | **79.3%** |
| SQuAD 2.0 | 80% | 72% | **90%** | **43.8%** |

---
## Funktioner

- **Välj först, komprimera sen** — identifierar vilka filer som svarar på din fråga och komprimerar *därefter*.
- **Återställer originalet exakt** — allt som utesluts kan återställas tecken för tecken.
- **Full transparens** — kvitto för varje beslut.
- **Faktagranskar svar** — jämför AI-svaret mot de tillhandahållna bevisen, lokalt.

---

<p align="center"><sub>Apache-2.0 · local-first · ingen extern analys som standard</sub></p>
<p align="center"><code>pip install entroly && entroly go</code></p>
