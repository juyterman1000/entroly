<p align="center">
  <img src="../../docs/assets/entroly_wordmark.svg" width="820" alt="Entroly">
</p>

<h1 align="center">Entroly — Drop-In Garance Kontextu pro Snížení Provozních Nákladů na AI</h1>

<p align="center"><b>Snižte zbytečný kontext bez ztráty kontroly nad kritickými důkazy.</b><br>
Nejprve vyberte nejhodnotnější důkazy, zkomprimujte je, udržujte originály obnovitelné a vygenerujte potvrzení — bez nutnosti přepisovat kódovou základnu nebo architekturu agentů.</p>
<p align="center">
  <sub>Entroly je lokální Context OS: obsahově adresované důkazy, obnovitelná komprese a auditovatelná potvrzení. Funguje přes proxy, MCP, pluginy, wrappery a SDK s Claude Code, Codex, OpenClaw, GitHub Copilot, Cursor, Aider a aplikacemi kompatibilními s OpenAI/Anthropic.</sub>
</p>
<p align="center">
  <a href="https://pypi.org/project/entroly/"><img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI" alt="Entroly on PyPI"></a>
  <a href="https://www.npmjs.com/package/entroly"><img src="https://img.shields.io/npm/v/entroly?color=red&label=npm" alt="Entroly on npm"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="Apache-2.0 license"></a>
  <a href="https://github.com/juyterman1000/entroly"><img src="https://img.shields.io/github/stars/juyterman1000/entroly?style=social" alt="Entroly GitHub stars"></a>
</p>

<p align="center"><b>100 438 stažení · roste každým dnem</b><br>
<sub>Měřeno z různých distribučních zdrojů.</sub></p>

<p align="center"><b>⭐ Pokud je pro vás Entroly užitečné, dejte projektu hvězdičku na GitHubu.</b><br>
<a href="https://github.com/juyterman1000/entroly">⭐ Dát Entroly hvězdičku na GitHubu</a> — pomáhá projektu růst a oslovit více vývojářů.</p>

<p align="center">
  <b><a href="../../README.md">English</a> · <a href="README.zh.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a> · <a href="README.es.md">Español</a> · <a href="README.hi.md">हिन्दी</a> · <a href="README.fr.md">Français</a> · <a href="README.de.md">Deutsch</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.it.md">Italiano</a> · <a href="README.tr.md">Türkçe</a> · <a href="README.vi.md">Tiếng Việt</a> · <a href="README.id.md">Bahasa Indonesia</a> · <a href="README.uk.md">Українська</a> · <a href="README.pl.md">Polski</a> · <a href="README.nl.md">Nederlands</a> · <a href="README.th.md">ไทย</a> · <a href="README.sv.md">Svenska</a> · Čeština · <a href="README.tl.md">Tagalog</a> · <a href="README.ro.md">Română</a></b>
</p>

---

## Co je Entroly? (Jednoduše)

AI asistenti pro programování mají limit paměti. Pokud jim předáte celý repozitář, stanou se pomalými, drahými a nepozornými — jako dát někomu 500stránkový manuál, když potřeboval pouze stranu 47.

**Entroly najde stranu 47.**

Stojí mezi vaším kódem a AI, čte vše a předává pouze ty části, které jsou relevantní pro zadanou otázku.
|  |  |
|---|---|
| 💰 **Váš účet klesne** | Méně textu odeslaného do AI znamená nižší fakturu. |
| 🔍 **Nic se neztratí** | Vše, co Entroly odloží, je zachováno a lze to *přesně* obnovit. |
| 🧾 **Můžete zkontrolovat jeho práci** | Každé rozhodnutí má potvrzení: co bylo zachováno, co vynecháno a proč. |

**Musím měnit svůj kód?** Ne. Entroly funguje s nástroji, které již používáte — Claude Code, Cursor, Copilot a 30+ dalšími.

---
## Instalace

| Platforma | Instalace | Co získáte |
|---|---|---|
| 🐍 **Python** (pip) — *doporučeno* | `pip install -U entroly` | Vše: CLI nástroj, server pro AI editor, knihovnu |
| 📦 **Node / npm** | `npm install -g entroly` | Stejný engine, bez Pythonu |
| 🦀 **Rust** (sestavení ze zdrojů) | `cd entroly-core && cargo build --release --bin entroly-rs --features proxy` | Jeden samostatný spustitelný soubor |
| 🍺 **Homebrew** | `brew install juyterman1000/entroly/entroly` | CLI nástroj pro macOS/Linux |
| 🐳 **Docker** | `docker pull ghcr.io/juyterman1000/entroly:latest` | Běh v kontejneru |

```bash
cd /your/repo
entroly verify-claims
entroly simulate
```

---
## Rychlý Start

> `pip install -U entroly && entroly go` — to je vše.

```python
from entroly import compress, compress_messages, optimize
compressed = compress(api_response, budget=2000)
messages   = compress_messages(messages, budget=30000)
context    = optimize(fragments, budget=8000, query="oprav chybu přihlášení")
```

---
## Benchmarky

| Benchmark | Základ | S Entroly | Zachování | Úspora tokenů |
|---|---|---|---|---|
| NeedleInAHaystack | 100% | 100% | **100%** | **99.5%** |
| LongBench (HotpotQA) | 64% | 66% | **103%** | **85.3%** |
| Berkeley Function Calling | 100% | 100% | **100%** | **79.3%** |
| SQuAD 2.0 | 80% | 72% | **90%** | **43.8%** |

---
## Funkce

- **Nejprve vybere, pak zkomprimuje** — zjistí, které soubory odpovídají na otázku, a *poté* je zkomprimuje.
- **Přesně vrátí originál** — vše vynechané lze obnovit znak po znaku.
- **Plná transparentnost** — potvrzení pro každé rozhodnutí.
- **Faktická kontrola** — lokálně porovnává odpověď AI s poskytnutými důkazy.

---

<p align="center"><sub>Apache-2.0 · lokální priorita · ve výchozím nastavení žádná externí analytika</sub></p>
<p align="center"><code>pip install entroly && entroly go</code></p>
