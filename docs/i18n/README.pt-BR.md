<p align="center">
  <img src="../../docs/assets/entroly_wordmark.svg" width="820" alt="Entroly">
</p>

<h1 align="center">Entroly — Garantia de Contexto Drop-In para Reduzir Custos Operacionais de IA</h1>

<p align="center"><b>Reduza contexto desnecessário sem perder o controle de evidências críticas.</b><br>
Selecione primeiro as evidências de maior valor, comprima-as, mantenha os originais recuperáveis e emita um recibo — sem reescrever seu código ou arquitetura de agente.</p>
<p align="center"><sub>Entroly é um Context OS local-first: evidências endereçadas por conteúdo, compressão recuperável e recibos auditáveis.</sub></p>
<p align="center">
  <a href="https://pypi.org/project/entroly/"><img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI" alt="Entroly on PyPI"></a>
  <a href="https://www.npmjs.com/package/entroly"><img src="https://img.shields.io/npm/v/entroly?color=red&label=npm" alt="Entroly on npm"></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="Apache-2.0 license"></a>
  <a href="https://github.com/juyterman1000/entroly"><img src="https://img.shields.io/github/stars/juyterman1000/entroly?style=social" alt="Entroly GitHub stars"></a>
</p>

<p align="center"><b>100.438 downloads · crescendo dia a dia</b><br>
<sub>Medido em diferentes fontes de distribuição.</sub></p>

<p align="center">
  <b><a href="../../README.md">English</a> · <a href="README.zh.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a> · <a href="README.es.md">Español</a> · <a href="README.hi.md">हिन्दी</a> · <a href="README.fr.md">Français</a> · <a href="README.de.md">Deutsch</a> · Português · <a href="README.it.md">Italiano</a> · <a href="README.tr.md">Türkçe</a> · <a href="README.vi.md">Tiếng Việt</a> · <a href="README.id.md">Bahasa Indonesia</a> · <a href="README.pl.md">Polski</a> · <a href="README.nl.md">Nederlands</a> · <a href="README.th.md">ไทย</a> · <a href="README.sv.md">Svenska</a> · <a href="README.cs.md">Čeština</a> · <a href="README.tl.md">Tagalog</a> · <a href="README.ro.md">Română</a></b>
</p>

---

## O que é o Entroly? (em linguagem simples)

Assistentes de codificação com IA têm limite de memória. Entregue todo o seu código e ele fica lento, caro e distraído — como dar a alguém um manual de 500 páginas quando só precisava da página 47.

**O Entroly encontra a página 47.**

|  |  |
|---|---|
| 💰 **Sua conta diminui** | Menos texto enviado à IA significa uma fatura menor. |
| 🔍 **Nada se perde** | Tudo que o Entroly separa é mantido e pode ser recuperado *exatamente* como estava. |
| 🧾 **Você pode verificar o trabalho** | Cada decisão vem com um recibo. |

---
## Instalação

| Plataforma | Instalação | O que você obtém |
|---|---|---|
| 🐍 **Python** (pip) — *recomendado* | `pip install -U entroly` | Tudo: ferramenta CLI, servidor, biblioteca |
| 📦 **Node / npm** | `npm install -g entroly` | Mesmo motor, sem Python |
| 🦀 **Rust** | `cd entroly-core && cargo build --release --bin entroly-rs --features proxy` | Programa autônomo |
| 🍺 **Homebrew** | `brew install juyterman1000/entroly/entroly` | CLI no macOS/Linux |
| 🐳 **Docker** | `docker pull ghcr.io/juyterman1000/entroly:latest` | Executa em contêiner |

```bash
cd /your/repo
entroly verify-claims
entroly simulate
```

---
## Início rápido

> `pip install -U entroly && entroly go` — é só isso.

```python
from entroly import compress, compress_messages, optimize
compressed = compress(api_response, budget=2000)
context    = optimize(fragments, budget=8000, query="corrigir o bug de login")
```

---
## Benchmarks

| Benchmark | Base | Com Entroly | Retenção | Economia de tokens |
|---|---|---|---|---|
| NeedleInAHaystack | 100% | 100% | **100%** | **99.5%** |
| LongBench (HotpotQA) | 64% | 66% | **103%** | **85.3%** |
| SQuAD 2.0 | 80% | 72% | **90%** | **43.8%** |

**Sendo honesto:** compressão é uma troca, não mágica.

---
## Funcionalidades

- **Seleciona primeiro, comprime depois** — descobre quais arquivos respondem sua pergunta, *depois* os comprime.
- **Devolve o original exatamente** — o que foi omitido pode ser restaurado caractere por caractere.
- **Mostra seu trabalho** — um recibo para cada decisão.
- **Verifica respostas** — compara a resposta da IA com as evidências fornecidas, localmente.

---

<p align="center"><sub>Apache-2.0 · local-first · sem analytics externo por padrão</sub></p>
<p align="center"><code>pip install entroly && entroly go</code></p>
