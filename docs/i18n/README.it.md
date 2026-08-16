<p align="center">
  <img src="../../docs/assets/entroly_wordmark.svg" width="820" alt="Entroly">
</p>

<h1 align="center">Entroly — Garanzia di Contesto Drop-In per Ridurre i Costi Operativi dell'IA</h1>

<p align="center"><b>Riduci il contesto non necessario senza perdere il controllo delle prove critiche.</b><br>
Seleziona prima le prove di maggior valore, comprimile, mantieni gli originali recuperabili ed emetti una ricevuta — senza riscrivere la tua base di codice o l'architettura degli agenti.</p>
<p align="center">
  <sub>Entroly è un Context OS local-first: prove indirizzate per contenuto, compressione recuperabile e ricevute verificabili. Funziona tramite proxy, MCP, plugin, wrapper e percorsi SDK con Claude Code, Codex, OpenClaw, GitHub Copilot, Cursor, Aider e app compatibili OpenAI/Anthropic.</sub>
</p>
<p align="center">
  <a href="https://pypi.org/project/entroly/"><img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI" alt="Entroly on PyPI"></a>
  <a href="https://www.npmjs.com/package/entroly"><img src="https://img.shields.io/npm/v/entroly?color=red&label=npm" alt="Entroly on npm"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="Apache-2.0 license"></a>
  <a href="https://github.com/juyterman1000/entroly"><img src="https://img.shields.io/github/stars/juyterman1000/entroly?style=social" alt="Entroly GitHub stars"></a>
</p>

<p align="center"><b>100.438 download · in crescita giorno dopo giorno</b><br>
<sub>Misurato su diverse fonti di distribuzione.</sub></p>

<p align="center"><b>⭐ Se Entroly ti è utile, aggiungi una stella al repository su GitHub.</b><br>
<a href="https://github.com/juyterman1000/entroly">⭐ Metti una stella a Entroly su GitHub</a> — aiuta il progetto a crescere e a raggiungere più sviluppatori.</p>

<p align="center">
  <b><a href="../../README.md">English</a> · <a href="README.zh.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a> · <a href="README.es.md">Español</a> · <a href="README.hi.md">हिन्दी</a> · <a href="README.fr.md">Français</a> · <a href="README.de.md">Deutsch</a> · <a href="README.pt-BR.md">Português</a> · Italiano · <a href="README.tr.md">Türkçe</a> · <a href="README.vi.md">Tiếng Việt</a> · <a href="README.id.md">Bahasa Indonesia</a> · <a href="README.uk.md">Українська</a> · <a href="README.pl.md">Polski</a></b>
</p>

---

## Cos'è Entroly? (In parole semplici)

Gli assistenti di programmazione IA hanno un limite di memoria. Fornisci l'intera base di codice e l'IA diventa lenta, costosa e distratta — come dare a qualcuno un manuale di 500 pagine quando aveva bisogno solo di pagina 47.

**Entroly trova la pagina 47.**

Si posiziona tra il tuo codice e l'IA, legge tutto e passa solo le parti che contano per la domanda effettivamente posta.
|  |  |
|---|---|
| 💰 **La tua fattura si riduce** | Meno testo inviato all'IA significa una fattura più leggera. |
| 🔍 **Nulla viene perso** | Tutto ciò che Entroly mette da parte viene conservato e può essere recuperato *esattamente* come era. |
| 🧾 **Puoi verificare il lavoro** | Ogni decisione è accompagnata da una ricevuta: cosa è stato mantenuto, cosa è stato escluso e perché. |

**Devo modificare il mio codice?** No. Entroly funziona con gli strumenti che già utilizzi — Claude Code, Cursor, Copilot e oltre 30 altri.

---
## Installazione

| Piattaforma | Installazione | Cosa ottieni |
|---|---|---|
| 🐍 **Python** (pip) — *consigliato* | `pip install -U entroly` | Tutto: strumento CLI, server editor IA, libreria di codice |
| 📦 **Node / npm** | `npm install -g entroly` | Stesso motore, nessun Python richiesto |
| 🦀 **Rust** (compilazione sorgente) | `cd entroly-core && cargo build --release --bin entroly-rs --features proxy` | Un unico programma autonomo |
| 🍺 **Homebrew** | `brew install juyterman1000/entroly/entroly` | Strumento CLI su macOS/Linux |
| 🐳 **Docker** | `docker pull ghcr.io/juyterman1000/entroly:latest` | Esecuzione in container |

```bash
cd /your/repo
entroly verify-claims
entroly simulate
```

---
## Guida Rapida

> `pip install -U entroly && entroly go` — tutto qui.

```python
from entroly import compress, compress_messages, optimize
compressed = compress(api_response, budget=2000)
messages   = compress_messages(messages, budget=30000)
context    = optimize(fragments, budget=8000, query="risolvi il bug di login")
```

---
## Benchmark

| Benchmark | Baseline | Con Entroly | Ritenzione | Risparmio Token |
|---|---|---|---|---|
| NeedleInAHaystack | 100% | 100% | **100%** | **99.5%** |
| LongBench (HotpotQA) | 64% | 66% | **103%** | **85.3%** |
| Berkeley Function Calling | 100% | 100% | **100%** | **79.3%** |
| SQuAD 2.0 | 80% | 72% | **90%** | **43.8%** |

---
## Funzionalità

- **Prima seleziona, poi comprime** — individua i file che rispondono alla domanda, *poi* li comprime.
- **Restituisce l'originale in modo esatto** — tutto ciò che viene omesso può essere ripristinato carattere per carattere.
- **Trasparenza decisionale** — una ricevuta per ogni decisione.
- **Verifica fattuale delle risposte** — confronta la risposta dell'IA con le prove fornite, localmente.

---

<p align="center"><sub>Apache-2.0 · local-first · nessun dato analitico esterno per impostazione predefinita</sub></p>
<p align="center"><code>pip install entroly && entroly go</code></p>
