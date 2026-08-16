<p align="center">
  <img src="../../docs/assets/entroly_wordmark.svg" width="820" alt="Entroly">
</p>

<h1 align="center">Entroly — Gwarancja Kontekstu Drop-In Obniżająca Koszty Operacyjne AI</h1>

<p align="center"><b>Zmniejsz niepotrzebny kontekst bez utraty kontroli nad krytycznymi dowodami.</b><br>
Wybierz najpierw najważniejsze dowody, skompresuj je, zachowaj oryginały do odzyskania i wygeneruj potwierdzenie — bez konieczności przepisywania bazy kodu lub architektury agenta.</p>
<p align="center">
  <sub>Entroly to lokalny Context OS: dowody adresowane zawartością, odwracalna kompresja i audytowalne potwierdzenia. Działa przez proxy, MCP, wtyczki, wrappery i SDK z Claude Code, Codex, OpenClaw, GitHub Copilot, Cursor, Aider oraz aplikacjami zgodnymi z OpenAI/Anthropic.</sub>
</p>
<p align="center">
  <a href="https://pypi.org/project/entroly/"><img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI" alt="Entroly on PyPI"></a>
  <a href="https://www.npmjs.com/package/entroly"><img src="https://img.shields.io/npm/v/entroly?color=red&label=npm" alt="Entroly on npm"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="Apache-2.0 license"></a>
  <a href="https://github.com/juyterman1000/entroly"><img src="https://img.shields.io/github/stars/juyterman1000/entroly?style=social" alt="Entroly GitHub stars"></a>
</p>

<p align="center"><b>100 438 pobrań · rośnie z dnia na dzień</b><br>
<sub>Zmierzone z różnych źródeł dystrybucji.</sub></p>

<p align="center"><b>⭐ Jeśli Entroly jest dla Ciebie przydatne, dodaj gwiazdkę repozytorium na GitHubie.</b><br>
<a href="https://github.com/juyterman1000/entroly">⭐ Dodaj gwiazdkę Entroly na GitHubie</a> — pomaga to w rozwoju projektu i dotarciu do większej liczby programistów.</p>

<p align="center">
  <b><a href="../../README.md">English</a> · <a href="README.zh.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a> · <a href="README.es.md">Español</a> · <a href="README.hi.md">हिन्दी</a> · <a href="README.fr.md">Français</a> · <a href="README.de.md">Deutsch</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.it.md">Italiano</a> · <a href="README.tr.md">Türkçe</a> · <a href="README.vi.md">Tiếng Việt</a> · <a href="README.id.md">Bahasa Indonesia</a> · <a href="README.uk.md">Українська</a> · Polski</b>
</p>

---

## Czym jest Entroly? (Prostym językiem)

Asystenci programistyczni AI mają ograniczenie pamięci. Podaj im całą bazę kodu, a staną się powolni, kosztowni i rozproszeni — to jak dać komuś 500-stronicowy podręcznik, gdy potrzebował tylko strony 47.

**Entroly znajduje stronę 47.**

Znajduje się między Twoim kodem a AI, czyta wszystko i przekazuje tylko te części, które mają znaczenie dla zadanego pytania.
|  |  |
|---|---|
| 💰 **Twój rachunek maleje** | Mniej tekstu wysłanego do AI oznacza mniejszą fakturę. |
| 🔍 **Nic nie ginie** | Wszystko, co Entroly odkłada, jest zachowane i może być odzyskane *dokładnie* w pierwotnej postaci. |
| 🧾 **Możesz sprawdzić jego pracę** | Każda decyzja zawiera potwierdzenie: co zachowano, co pominięto i dlaczego. |

**Czy muszę zmieniać kod?** Nie. Entroly działa z narzędziami, których już używasz — Claude Code, Cursor, Copilot i ponad 30 innymi.

---
## Instalacja

| Platforma | Instalacja | Co otrzymujesz |
|---|---|---|
| 🐍 **Python** (pip) — *zalecane* | `pip install -U entroly` | Wszystko: narzędzie CLI, serwer edytora AI, biblioteka |
| 📦 **Node / npm** | `npm install -g entroly` | Ten sam silnik, bez Pythona |
| 🦀 **Rust** (kompilacja ze źródeł) | `cd entroly-core && cargo build --release --bin entroly-rs --features proxy` | Jeden samodzielny program |
| 🍺 **Homebrew** | `brew install juyterman1000/entroly/entroly` | Narzędzie CLI na macOS/Linux |
| 🐳 **Docker** | `docker pull ghcr.io/juyterman1000/entroly:latest` | Działa w kontenerze |

```bash
cd /your/repo
entroly verify-claims
entroly simulate
```

---
## Szybki Start

> `pip install -U entroly && entroly go` — to wszystko.

```python
from entroly import compress, compress_messages, optimize
compressed = compress(api_response, budget=2000)
messages   = compress_messages(messages, budget=30000)
context    = optimize(fragments, budget=8000, query="napraw błąd logowania")
```

---
## Benchmarki

| Benchmark | Baza | Z Entroly | Retencja | Oszczędność tokenów |
|---|---|---|---|---|
| NeedleInAHaystack | 100% | 100% | **100%** | **99.5%** |
| LongBench (HotpotQA) | 64% | 66% | **103%** | **85.3%** |
| Berkeley Function Calling | 100% | 100% | **100%** | **79.3%** |
| SQuAD 2.0 | 80% | 72% | **90%** | **43.8%** |

---
## Funkcje

- **Najpierw wybiera, potem kompresuje** — ustala, które pliki odpowiadają na pytanie, *następnie* je kompresuje.
- **Zwraca dokładnie oryginał** — wszystko pominięte można przywrócić znak po znaku.
- **Pokazuje swoją pracę** — potwierdzenie dla każdej decyzji.
- **Weryfikuje faktycznie odpowiedzi** — porównuje odpowiedź AI z dostarczonymi dowodami, lokalnie.

---

<p align="center"><sub>Apache-2.0 · lokalnie · domyślnie brak zewnętrznej analityki</sub></p>
<p align="center"><code>pip install entroly && entroly go</code></p>
