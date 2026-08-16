<p align="center">
  <img src="../../docs/assets/entroly_wordmark.svg" width="820" alt="Entroly">
</p>

<h1 align="center">Entroly — Drop-In Contextgarantie om AI-Operationele Kosten te Verlagen</h1>

<p align="center"><b>Verminder onnodige context zonder de controle over kritisch bewijs te verliezen.</b><br>
Selecteer eerst het meest waardevolle bewijs, comprimeer het, houd originelen herstelbaar en geef een ontvangstbewijs af — zonder je codebase of agent-architectuur te herschrijven.</p>
<p align="center">
  <sub>Entroly is een local-first Context OS: content-geadresseerd bewijs, herstelbare compressie en controleerbare ontvangstbewijzen. Werkt via proxy, MCP, plug-in, wrapper en SDK-paden met Claude Code, Codex, OpenClaw, GitHub Copilot, Cursor, Aider en OpenAI/Anthropic-compatibele apps.</sub>
</p>
<p align="center">
  <a href="https://pypi.org/project/entroly/"><img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI" alt="Entroly on PyPI"></a>
  <a href="https://www.npmjs.com/package/entroly"><img src="https://img.shields.io/npm/v/entroly?color=red&label=npm" alt="Entroly on npm"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="Apache-2.0 license"></a>
  <a href="https://github.com/juyterman1000/entroly"><img src="https://img.shields.io/github/stars/juyterman1000/entroly?style=social" alt="Entroly GitHub stars"></a>
</p>

<p align="center"><b>100.438 downloads · groeit elke dag</b><br>
<sub>Gemeten over verschillende distributiebronnen.</sub></p>

<p align="center"><b>⭐ Als Entroly nuttig voor je is, geef de repository dan een ster op GitHub.</b><br>
<a href="https://github.com/juyterman1000/entroly">⭐ Geef Entroly een ster op GitHub</a> — het helpt het project te groeien en meer ontwikkelaars te bereiken.</p>

<p align="center">
  <b><a href="../../README.md">English</a> · <a href="README.zh.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a> · <a href="README.es.md">Español</a> · <a href="README.hi.md">हिन्दी</a> · <a href="README.fr.md">Français</a> · <a href="README.de.md">Deutsch</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.it.md">Italiano</a> · <a href="README.tr.md">Türkçe</a> · <a href="README.vi.md">Tiếng Việt</a> · <a href="README.id.md">Bahasa Indonesia</a> · <a href="README.uk.md">Українська</a> · <a href="README.pl.md">Polski</a> · Nederlands · <a href="README.th.md">ไทย</a> · <a href="README.sv.md">Svenska</a> · <a href="README.cs.md">Čeština</a> · <a href="README.tl.md">Tagalog</a> · <a href="README.ro.md">Română</a></b>
</p>

---

## Wat is Entroly? (In gewoon Nederlands)

AI-codeerassistenten hebben een geheugenlimiet. Geef ze je hele codebase en ze worden traag, duur en afgeleid — alsof je iemand een handleiding van 500 pagina's geeft terwijl ze alleen pagina 47 nodig hadden.

**Entroly vindt pagina 47.**

Het bevindt zich tussen jouw code en de AI, leest alles en geeft alleen de delen door die ertoe doen voor de vraag die daadwerkelijk wordt gesteld.
|  |  |
|---|---|
| 💰 **Je factuur daalt** | Minder tekst naar de AI betekent een lagere factuur. |
| 🔍 **Er gaat niets verloren** | Alles wat Entroly opzij zet, wordt bewaard en kan *exact* worden hersteld. |
| 🧾 **Je kunt het werk controleren** | Elke beslissing gaat vergezeld van een ontvangstbewijs: wat is bewaard, wat is weggelaten en waarom. |

**Moet ik mijn code aanpassen?** Nee. Entroly werkt met de tools die je al gebruikt — Claude Code, Cursor, Copilot en 30+ andere.

---
## Installatie

| Platform | Installatie | Wat je krijgt |
|---|---|---|
| 🐍 **Python** (pip) — *aanbevolen* | `pip install -U entroly` | Alles: CLI-tool, server, codebibliotheek |
| 📦 **Node / npm** | `npm install -g entroly` | Dezelfde engine, geen Python vereist |
| 🦀 **Rust** (broncode build) | `cd entroly-core && cargo build --release --bin entroly-rs --features proxy` | Eén op zichzelf staand programma |
| 🍺 **Homebrew** | `brew install juyterman1000/entroly/entroly` | CLI-tool op macOS/Linux |
| 🐳 **Docker** | `docker pull ghcr.io/juyterman1000/entroly:latest` | Draait in een container |

```bash
cd /your/repo
entroly verify-claims
entroly simulate
```

---
## Snelle Start

> `pip install -U entroly && entroly go` — dat is alles.

```python
from entroly import compress, compress_messages, optimize
compressed = compress(api_response, budget=2000)
messages   = compress_messages(messages, budget=30000)
context    = optimize(fragments, budget=8000, query="los de login bug op")
```

---
## Benchmarks

| Benchmark | Basislijn | Met Entroly | Behoud | Tokenbesparing |
|---|---|---|---|---|
| NeedleInAHaystack | 100% | 100% | **100%** | **99.5%** |
| LongBench (HotpotQA) | 64% | 66% | **103%** | **85.3%** |
| Berkeley Function Calling | 100% | 100% | **100%** | **79.3%** |
| SQuAD 2.0 | 80% | 72% | **90%** | **43.8%** |

---
## Functies

- **Eerst selecteren, dan comprimeren** — bepaalt welke bestanden de vraag beantwoorden, *daarna* pas comprimeren.
- **Geeft het origineel exact terug** — weggelaten inhoud kan karakter voor karakter worden hersteld.
- **Transparantie** — een ontvangstbewijs voor elke beslissing.
- **Feitencontrole** — vergelijkt het antwoord van de AI met het geleverde bewijs, lokaal.

---

<p align="center"><sub>Apache-2.0 · local-first · standaard geen externe analyses</sub></p>
<p align="center"><code>pip install entroly && entroly go</code></p>
