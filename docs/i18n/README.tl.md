<p align="center">
  <img src="../../docs/assets/entroly_wordmark.svg" width="820" alt="Entroly">
</p>

<h1 align="center">Entroly — Drop-In Garantiya ng Konteksto para Mabawasan ang Gastos sa AI</h1>

<p align="center"><b>Bawasan ang hindi kinakailangang konteksto nang hindi nawawala ang kontrol sa mahahalagang ebidensya.</b><br>
Piliin muna ang pinakamahalagang ebidensya, i-compress ito, panatilihing maibabalik ang orihinal, at maglabas ng resibo — nang hindi muling sinusulat ang iyong codebase o arkitektura ng agent.</p>
<p align="center">
  <sub>Ang Entroly ay isang local-first Context OS: content-addressed na ebidensya, recoverable compression, at ma-audit na mga resibo. Gumagana sa pamamagitan ng proxy, MCP, plugin, wrapper, at SDK pathways kasama ang Claude Code, Codex, OpenClaw, GitHub Copilot, Cursor, Aider, at mga app na katugma ng OpenAI/Anthropic.</sub>
</p>
<p align="center">
  <a href="https://pypi.org/project/entroly/"><img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI" alt="Entroly on PyPI"></a>
  <a href="https://www.npmjs.com/package/entroly"><img src="https://img.shields.io/npm/v/entroly?color=red&label=npm" alt="Entroly on npm"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="Apache-2.0 license"></a>
  <a href="https://github.com/juyterman1000/entroly"><img src="https://img.shields.io/github/stars/juyterman1000/entroly?style=social" alt="Entroly GitHub stars"></a>
</p>

<p align="center"><b>100,438 na download · lumalaki araw-araw</b><br>
<sub>Sinukat sa iba't ibang pinagmulan ng pamamahagi.</sub></p>

<p align="center"><b>⭐ Kung kapaki-pakinabang sa iyo ang Entroly, mangyaring i-star ang repository sa GitHub.</b><br>
<a href="https://github.com/juyterman1000/entroly">⭐ I-star ang Entroly sa GitHub</a> — nakakatulong ito sa paglago ng proyekto at pag-abot sa mas maraming developer.</p>

<p align="center">
  <b><a href="../../README.md">English</a> · <a href="README.zh.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a> · <a href="README.es.md">Español</a> · <a href="README.hi.md">हिन्दी</a> · <a href="README.fr.md">Français</a> · <a href="README.de.md">Deutsch</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.it.md">Italiano</a> · <a href="README.tr.md">Türkçe</a> · <a href="README.vi.md">Tiếng Việt</a> · <a href="README.id.md">Bahasa Indonesia</a> · <a href="README.uk.md">Українська</a> · <a href="README.pl.md">Polski</a> · <a href="README.nl.md">Nederlands</a> · <a href="README.th.md">ไทย</a> · <a href="README.sv.md">Svenska</a> · <a href="README.cs.md">Čeština</a> · Tagalog · <a href="README.ro.md">Română</a></b>
</p>

---

## Ano ang Entroly? (Sa simpleng salita)

May limitasyon sa memorya ang mga AI coding assistant. Ibigay ang buong codebase at babagal ito, magiging mahal, at mawawalan ng pokus — tulad ng pagbibigay sa isang tao ng 500-pahinang manual kung ang kailangan lang nila ay pahina 47.

**Hinahanap ng Entroly ang pahina 47.**

Nasa pagitan ito ng iyong code at ng AI, binabasa ang lahat, at ipinapasa lamang ang mga bahaging mahalaga para sa itinanong.
|  |  |
|---|---|
| 💰 **Bumababa ang iyong bayarin** | Mas kaunting teksto na ipinadala sa AI ay nangangahulugang mas maliit na bayarin. |
| 🔍 **Walang nawawala** | Ang anumang itinabi ng Entroly ay napananatili at maaaring maibalik *nang eksakto* tulad ng dati. |
| 🧾 **Maaari mong suriin ang trabaho nito** | Bawat desisyon ay may kasamang resibo: kung ano ang itinabi, kung ano ang iniwan at bakit. |

**Kailangan ko bang baguhin ang aking code?** Hindi. Gumagana ang Entroly sa mga tool na ginagamit mo na — Claude Code, Cursor, Copilot at 30+ pang iba.

---
## Pag-install

| Platform | Pag-install | Ano ang Makukuha Mo |
|---|---|---|
| 🐍 **Python** (pip) — *inirerekomenda* | `pip install -U entroly` | Lahat: CLI tool, AI editor server, code library |
| 📦 **Node / npm** | `npm install -g entroly` | Parehong engine, hindi kailangan ng Python |
| 🦀 **Rust** (source build) | `cd entroly-core && cargo build --release --bin entroly-rs --features proxy` | Isang solong self-contained program |
| 🍺 **Homebrew** | `brew install juyterman1000/entroly/entroly` | CLI tool sa macOS/Linux |
| 🐳 **Docker** | `docker pull ghcr.io/juyterman1000/entroly:latest` | Tumatakbo sa container |

```bash
cd /your/repo
entroly verify-claims
entroly simulate
```

---
## Quickstart

> `pip install -U entroly && entroly go` — 'yun lang.

```python
from entroly import compress, compress_messages, optimize
compressed = compress(api_response, budget=2000)
messages   = compress_messages(messages, budget=30000)
context    = optimize(fragments, budget=8000, query="ayusin ang login bug")
```

---
## Mga Benchmark

| Benchmark | Baseline | Kasama ang Entroly | Pagpapanatili | Pagtitipid sa Token |
|---|---|---|---|---|
| NeedleInAHaystack | 100% | 100% | **100%** | **99.5%** |
| LongBench (HotpotQA) | 64% | 66% | **103%** | **85.3%** |
| Berkeley Function Calling | 100% | 100% | **100%** | **79.3%** |
| SQuAD 2.0 | 80% | 72% | **90%** | **43.8%** |

---
## Mga Tampok

- **Piliin muna, i-compress pagkatapos** — alamin kung aling mga file ang talagang sumasagot sa iyong tanong, *pagkatapos* ay i-compress ang mga ito.
- **Ibalik ang orihinal nang eksakto** — ang anumang iniwan ay maaaring maibalik bawat karakter.
- **Ipinapakita ang trabaho nito** — isang resibo para sa bawat desisyon.
- **Fact-check ng mga sagot** — inihahambing ang sagot ng AI sa ibinigay na ebidensya, nang lokal.

---

<p align="center"><sub>Apache-2.0 · local-first · walang palabas na analytics ayon sa default</sub></p>
<p align="center"><code>pip install entroly && entroly go</code></p>
