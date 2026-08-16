<p align="center">
  <img src="../../docs/assets/entroly_wordmark.svg" width="820" alt="Entroly">
</p>

<h1 align="center">Entroly — Garanție de Context Drop-In pentru Reducerea Costurilor Operaționale AI</h1>

<p align="center"><b>Reduceți contextul inutil fără a pierde controlul asupra dovezilor critice.</b><br>
Selectați mai întâi dovezile cele mai valoroase, comprimați-le, păstrați originalele recuperabile și emiteți o chitanță — fără a rescrie baza de cod sau arhitectura agenților.</p>
<p align="center">
  <sub>Entroly este un Context OS local-first: dovezi adresate după conținut, compresie recuperabilă și chitanțe auditabile. Funcționează prin proxy, MCP, pluginuri, wrappere și căi SDK cu Claude Code, Codex, OpenClaw, GitHub Copilot, Cursor, Aider și aplicații compatibile OpenAI/Anthropic.</sub>
</p>
<p align="center">
  <a href="https://pypi.org/project/entroly/"><img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI" alt="Entroly on PyPI"></a>
  <a href="https://www.npmjs.com/package/entroly"><img src="https://img.shields.io/npm/v/entroly?color=red&label=npm" alt="Entroly on npm"></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="Apache-2.0 license"></a>
  <a href="https://github.com/juyterman1000/entroly"><img src="https://img.shields.io/github/stars/juyterman1000/entroly?style=social" alt="Entroly GitHub stars"></a>
</p>

<p align="center"><b>100.438 descărcări · în creștere în fiecare zi</b><br>
<sub>Măsurat pe diferite surse de distribuție.</sub></p>

<p align="center"><b>⭐ Dacă Entroly vă este util, acordați o stea depozitului pe GitHub.</b><br>
<a href="https://github.com/juyterman1000/entroly">⭐ Acordați o stea pentru Entroly pe GitHub</a> — ajută proiectul să crească și să ajungă la mai mulți dezvoltatori.</p>

<p align="center">
  <b><a href="../../README.md">English</a> · <a href="README.zh.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a> · <a href="README.es.md">Español</a> · <a href="README.hi.md">हिन्दी</a> · <a href="README.fr.md">Français</a> · <a href="README.de.md">Deutsch</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.it.md">Italiano</a> · <a href="README.tr.md">Türkçe</a> · <a href="README.vi.md">Tiếng Việt</a> · <a href="README.id.md">Bahasa Indonesia</a> · <a href="README.pl.md">Polski</a> · <a href="README.nl.md">Nederlands</a> · <a href="README.th.md">ไทย</a> · <a href="README.sv.md">Svenska</a> · <a href="README.cs.md">Čeština</a> · <a href="README.tl.md">Tagalog</a> · Română</b>
</p>

---

## Ce este Entroly? (Pe înțelesul tuturor)

Asistenții de codare AI au o limită de memorie. Dacă le trimiteți întreaga bază de cod, devin lenți, scumpi și distrași — ca și cum ați da cuiva un manual de 500 de pagini când avea nevoie doar de pagina 47.

**Entroly găsește pagina 47.**

Se plasează între codul dumneavoastră și AI, citește totul și transmite doar părțile care contează pentru întrebarea adresată.
|  |  |
|---|---|
| 💰 **Factura dumneavoastră scade** | Mai puțin text trimis către AI înseamnă o factură mai mică. |
| 🔍 **Nimic nu se pierde** | Tot ce lasă deoparte Entroly este păstrat și poate fi recuperat *exact* așa cum a fost. |
| 🧾 **Puteți verifica activitatea** | Fiecare decizie vine cu o chitanță: ce s-a păstrat, ce s-a omis și de ce. |

**Trebuie să îmi modific codul?** Nu. Entroly funcționează cu instrumentele pe care le utilizați deja — Claude Code, Cursor, Copilot și peste 30 de altele.

---
## Instalare

| Platformă | Instalare | Ce obțineți |
|---|---|---|
| 🐍 **Python** (pip) — *recomandat* | `pip install -U entroly` | Totul: instrument CLI, server editor AI, bibliotecă de cod |
| 📦 **Node / npm** | `npm install -g entroly` | Același motor, fără Python |
| 🦀 **Rust** (compilare din sursă) | `cd entroly-core && cargo build --release --bin entroly-rs --features proxy` | Un singur program autonom |
| 🍺 **Homebrew** | `brew install juyterman1000/entroly/entroly` | Instrument CLI pe macOS/Linux |
| 🐳 **Docker** | `docker pull ghcr.io/juyterman1000/entroly:latest` | Rulează în container |

```bash
cd /your/repo
entroly verify-claims
entroly simulate
```

---
## Pornire Rapidă

> `pip install -U entroly && entroly go` — atât de simplu.

```python
from entroly import compress, compress_messages, optimize
compressed = compress(api_response, budget=2000)
messages   = compress_messages(messages, budget=30000)
context    = optimize(fragments, budget=8000, query="repară bug-ul de autentificare")
```

---
## Teste Comparative (Benchmarks)

| Benchmark | Bază | Cu Entroly | Retenție | Economie Tokeni |
|---|---|---|---|---|
| NeedleInAHaystack | 100% | 100% | **100%** | **99.5%** |
| LongBench (HotpotQA) | 64% | 66% | **103%** | **85.3%** |
| Berkeley Function Calling | 100% | 100% | **100%** | **79.3%** |
| SQuAD 2.0 | 80% | 72% | **90%** | **43.8%** |

---
## Funcționalități

- **Selectează mai întâi, comprimă după** — determină ce fișiere răspund cu adevărat la întrebare, *apoi* le comprimă.
- **Restituie originalul exact** — orice este omis poate fi restaurat caracter cu caracter.
- **Transparență completă** — o chitanță pentru fiecare decizie.
- **Verificare factuală a răspunsurilor** — compară răspunsul AI cu dovezile furnizate, local.

---

<p align="center"><sub>Apache-2.0 · local-first · fără analize externe în mod implicit</sub></p>
<p align="center"><code>pip install entroly && entroly go</code></p>
