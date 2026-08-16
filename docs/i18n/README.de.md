<p align="center">
  <img src="../../docs/assets/entroly_wordmark.svg" width="820" alt="Entroly">
</p>

<h1 align="center">Entroly — Drop-In Context-Assurance zur Senkung der KI-Betriebskosten</h1>

<p align="center"><b>Reduzieren Sie unnötigen Kontext, ohne die Kontrolle über kritische Beweise zu verlieren.</b><br>
Wählen Sie zuerst die wertvollsten Beweise aus, komprimieren Sie sie, halten Sie Originale wiederherstellbar und erstellen Sie eine Quittung — ohne Ihre Codebasis oder Agenten-Architektur umzuschreiben.</p>
<p align="center"><sub>Entroly ist ein Local-First Context OS: inhaltsadressierte Beweise, wiederherstellbare Kompression und prüfbare Quittungen.</sub></p>
<p align="center">
  <a href="https://pypi.org/project/entroly/"><img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI" alt="Entroly on PyPI"></a>
  <a href="https://www.npmjs.com/package/entroly"><img src="https://img.shields.io/npm/v/entroly?color=red&label=npm" alt="Entroly on npm"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="Apache-2.0 license"></a>
  <a href="https://github.com/juyterman1000/entroly"><img src="https://img.shields.io/github/stars/juyterman1000/entroly?style=social" alt="Entroly GitHub stars"></a>
</p>

<p align="center"><b>100.438 Downloads · täglich wachsend</b><br>
<sub>Gemessen über verschiedene Vertriebsquellen.</sub></p>

<p align="center">
  <b><a href="../../README.md">English</a> · <a href="README.zh.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a> · <a href="README.es.md">Español</a> · <a href="README.hi.md">हिन्दी</a> · <a href="README.fr.md">Français</a> · Deutsch · <a href="README.pt-BR.md">Português</a> · <a href="README.it.md">Italiano</a> · <a href="README.tr.md">Türkçe</a> · <a href="README.vi.md">Tiếng Việt</a> · <a href="README.id.md">Bahasa Indonesia</a> · <a href="README.uk.md">Українська</a> · <a href="README.pl.md">Polski</a></b>
</p>

---

## Was ist Entroly? (einfach erklärt)

KI-Coding-Assistenten haben ein Gedächtnislimit. Übergeben Sie ihnen die gesamte Codebasis und sie werden langsam, teuer und abgelenkt — wie jemandem ein 500-seitiges Handbuch zu geben, wenn nur Seite 47 gebraucht wird.

**Entroly findet Seite 47.**

|  |  |
|---|---|
| 💰 **Ihre Rechnung sinkt** | Weniger Text an die KI bedeutet eine kleinere Rechnung. |
| 🔍 **Nichts geht verloren** | Alles, was Entroly beiseitelegt, wird aufbewahrt und kann *exakt* wiederhergestellt werden. |
| 🧾 **Sie können die Arbeit überprüfen** | Jede Entscheidung kommt mit einer Quittung. |

**Muss ich meinen Code ändern?** Nein. Entroly funktioniert mit den Tools, die Sie bereits verwenden.

---
## Installation

| Plattform | Installation | Was Sie erhalten |
|---|---|---|
| 🐍 **Python** (pip) — *empfohlen* | `pip install -U entroly` | Alles: CLI-Tool, Server, Code-Bibliothek |
| 📦 **Node / npm** | `npm install -g entroly` | Gleiche Engine, kein Python nötig |
| 🦀 **Rust** | `cd entroly-core && cargo build --release --bin entroly-rs --features proxy` | Eigenständiges Programm |
| 🍺 **Homebrew** | `brew install juyterman1000/entroly/entroly` | CLI-Tool auf macOS/Linux |
| 🐳 **Docker** | `docker pull ghcr.io/juyterman1000/entroly:latest` | Läuft im Container |

```bash
cd /your/repo
entroly verify-claims
entroly simulate
```

---
## Schnellstart

> `pip install -U entroly && entroly go` — das ist alles.

```python
from entroly import compress, compress_messages, optimize
compressed = compress(api_response, budget=2000)
context    = optimize(fragments, budget=8000, query="Login-Bug beheben")
```

---
## Benchmarks

| Benchmark | Baseline | Mit Entroly | Erhaltung | Token-Einsparung |
|---|---|---|---|---|
| NeedleInAHaystack | 100% | 100% | **100%** | **99.5%** |
| LongBench (HotpotQA) | 64% | 66% | **103%** | **85.3%** |
| SQuAD 2.0 | 80% | 72% | **90%** | **43.8%** |

**Ehrlich gesagt:** Kompression ist ein Kompromiss, keine Magie.

---
## Funktionen

- **Erst auswählen, dann komprimieren** — bestimmt welche Dateien die Frage beantworten, *dann* komprimiert sie.
- **Gibt das Original exakt zurück** — Ausgelassenes kann zeichengenau wiederhergestellt werden.
- **Zeigt seine Arbeit** — eine Quittung für jede Entscheidung.
- **Überprüft Antworten** — vergleicht KI-Antworten mit den bereitgestellten Beweisen, lokal.

---

<p align="center"><sub>Apache-2.0 · Local-First · keine externen Analysen standardmäßig</sub></p>
<p align="center"><code>pip install entroly && entroly go</code></p>
