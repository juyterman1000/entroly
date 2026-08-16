<p align="center">
  <img src="../../docs/assets/entroly_wordmark.svg" width="820" alt="Entroly">
</p>

<h1 align="center">Entroly — Assurance de Contexte Drop-In pour Réduire les Coûts Opérationnels de l'IA</h1>

<p align="center"><b>Réduisez le contexte inutile sans perdre le contrôle des preuves critiques.</b><br>
Sélectionnez d'abord les preuves les plus pertinentes, compressez-les, gardez les originaux récupérables et émettez un reçu — sans réécrire votre code ou votre architecture d'agent.</p>
<p align="center">
  <sub>Entroly est un Context OS local-first : preuves adressées par contenu, compression récupérable et reçus auditables. Fonctionne via proxy, MCP, plugin, wrapper et SDK avec Claude Code, Codex, OpenClaw, GitHub Copilot, Cursor, Aider et les apps compatibles OpenAI/Anthropic.</sub>
</p>
<p align="center">
  <a href="https://pypi.org/project/entroly/"><img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI" alt="Entroly on PyPI"></a>
  <a href="https://www.npmjs.com/package/entroly"><img src="https://img.shields.io/npm/v/entroly?color=red&label=npm" alt="Entroly on npm"></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="Apache-2.0 license"></a>
  <a href="https://github.com/juyterman1000/entroly"><img src="https://img.shields.io/github/stars/juyterman1000/entroly?style=social" alt="Entroly GitHub stars"></a>
</p>

<p align="center"><b>100 438 téléchargements · en croissance chaque jour</b><br>
<sub>Mesuré à travers différentes sources de distribution.</sub></p>

<p align="center">
  <b><a href="../../README.md">English</a> · <a href="README.zh.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a> · <a href="README.es.md">Español</a> · <a href="README.hi.md">हिन्दी</a> · Français · <a href="README.de.md">Deutsch</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.it.md">Italiano</a> · <a href="README.tr.md">Türkçe</a> · <a href="README.vi.md">Tiếng Việt</a> · <a href="README.id.md">Bahasa Indonesia</a> · <a href="README.pl.md">Polski</a> · <a href="README.nl.md">Nederlands</a> · <a href="README.th.md">ไทย</a> · <a href="README.sv.md">Svenska</a> · <a href="README.cs.md">Čeština</a> · <a href="README.tl.md">Tagalog</a> · <a href="README.ro.md">Română</a></b>
</p>

---

## Qu'est-ce qu'Entroly ? (en langage simple)

Les assistants de codification IA ont une limite de mémoire. Donnez-leur tout votre code et ils deviennent lents, coûteux et distraits — comme donner un manuel de 500 pages à quelqu'un qui n'avait besoin que de la page 47.

**Entroly trouve la page 47.**

|  |  |
|---|---|
| 💰 **Votre facture diminue** | Moins de texte envoyé à l'IA signifie une facture plus petite. |
| 🔍 **Rien n'est perdu** | Tout ce qu'Entroly met de côté est conservé et peut être récupéré *exactement* tel quel. |
| 🧾 **Vous pouvez vérifier son travail** | Chaque décision s'accompagne d'un reçu. |

---
## Installation

| Plateforme | Installation | Ce que vous obtenez |
|---|---|---|
| 🐍 **Python** (pip) — *recommandé* | `pip install -U entroly` | Tout : outil CLI, serveur, bibliothèque |
| 📦 **Node / npm** | `npm install -g entroly` | Même moteur, sans Python |
| 🦀 **Rust** | `cd entroly-core && cargo build --release --bin entroly-rs --features proxy` | Programme autonome |
| 🍺 **Homebrew** | `brew install juyterman1000/entroly/entroly` | Outil CLI sur macOS/Linux |
| 🐳 **Docker** | `docker pull ghcr.io/juyterman1000/entroly:latest` | Exécution en conteneur |

```bash
cd /your/repo
entroly verify-claims
entroly simulate
```

---
## Démarrage rapide

> `pip install -U entroly && entroly go` — c'est tout.

```python
from entroly import compress, compress_messages, optimize
compressed = compress(api_response, budget=2000)
messages   = compress_messages(messages, budget=30000)
context    = optimize(fragments, budget=8000, query="corriger le bug de connexion")
```

---
## Benchmarks

| Benchmark | Base | Avec Entroly | Rétention | Économie de tokens |
|---|---|---|---|---|
| NeedleInAHaystack | 100% | 100% | **100%** | **99.5%** |
| LongBench (HotpotQA) | 64% | 66% | **103%** | **85.3%** |
| SQuAD 2.0 | 80% | 72% | **90%** | **43.8%** |

**En toute honnêteté :** la compression est un compromis, pas de la magie.

---
## Fonctionnalités

- **Sélectionne d'abord, compresse ensuite** — détermine quels fichiers répondent à votre question, *puis* les compresse.
- **Restitue l'original exactement** — tout ce qui est omis peut être restauré caractère par caractère.
- **Montre son travail** — un reçu pour chaque décision.
- **Vérifie les réponses** — compare la réponse de l'IA avec les preuves fournies, localement.

---

<p align="center"><sub>Apache-2.0 · local-first · pas d'analytiques externes par défaut</sub></p>
<p align="center"><code>pip install entroly && entroly go</code></p>
