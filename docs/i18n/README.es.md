<p align="center">
  <img src="../../docs/assets/entroly_wordmark.svg" width="820" alt="Entroly">
</p>

<h1 align="center">Entroly — Garantía de Contexto Drop-In para Reducir el Costo Operativo de IA</h1>

<p align="center"><b>Reduce el contexto innecesario sin perder el control de la evidencia crítica.</b><br>
Selecciona primero la evidencia de mayor valor, comprímela, mantén los originales recuperables y emite un recibo — sin reescribir tu código o arquitectura de agente.</p>
<p align="center">
  <sub>Entroly es un Context OS local-first: evidencia direccionada por contenido, compresión recuperable y recibos auditables. Funciona a través de proxy, MCP, plugin, wrapper y rutas SDK con Claude Code, Codex, OpenClaw, GitHub Copilot, Cursor, Aider y apps compatibles con OpenAI/Anthropic.</sub>
</p>
<p align="center">
  <a href="https://pypi.org/project/entroly/"><img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI" alt="Entroly on PyPI"></a>
  <a href="https://www.npmjs.com/package/entroly"><img src="https://img.shields.io/npm/v/entroly?color=red&label=npm" alt="Entroly on npm"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="Apache-2.0 license"></a>
  <a href="https://github.com/juyterman1000/entroly"><img src="https://img.shields.io/github/stars/juyterman1000/entroly?style=social" alt="Entroly GitHub stars"></a>
</p>

<p align="center"><b>100,438 descargas · creciendo día a día</b><br>
<sub>Medido a través de diferentes fuentes de distribución.</sub></p>

<p align="center"><b>⭐ Si Entroly te es útil, dale una estrella al repositorio en GitHub.</b><br>
<a href="https://github.com/juyterman1000/entroly">⭐ Dale estrella a Entroly en GitHub</a> — ayuda al proyecto a crecer y llegar a más desarrolladores.</p>

<p align="center">
  <b><a href="../../README.md">English</a> · <a href="README.zh.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a> · Español · <a href="README.hi.md">हिन्दी</a> · <a href="README.fr.md">Français</a> · <a href="README.de.md">Deutsch</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.it.md">Italiano</a> · <a href="README.tr.md">Türkçe</a> · <a href="README.vi.md">Tiếng Việt</a> · <a href="README.id.md">Bahasa Indonesia</a> · <a href="README.uk.md">Українська</a> · <a href="README.pl.md">Polski</a></b>
</p>

---

## ¿Qué es Entroly? (en español sencillo)

Los asistentes de codificación con IA tienen un límite de memoria. Dale todo tu código y se vuelve lento, caro y distraído — como darle a alguien un manual de 500 páginas cuando solo necesitaba la página 47.

**Entroly encuentra la página 47.**

Se sitúa entre tu código y la IA, lee todo y pasa solo las partes relevantes para la pregunta.
|  |  |
|---|---|
| 💰 **Tu factura baja** | Menos texto enviado a la IA significa una factura más pequeña. |
| 🔍 **Nada se pierde** | Todo lo que Entroly aparta se conserva y puede recuperarse *exactamente* como estaba. |
| 🧾 **Puedes verificar su trabajo** | Cada decisión viene con un recibo: qué se conservó, qué se omitió y por qué. |

**¿Tengo que cambiar mi código?** No. Entroly funciona con las herramientas que ya usas — Claude Code, Cursor, Copilot y más de 30 otras.

**¿Necesito pagar algo para probarlo?** No. Los dos comandos en la sección de [Instalación](#instalación) se ejecutan completamente en tu máquina local, sin clave API.

---
## Instalación

| Plataforma | Instalación | Lo que obtienes |
|---|---|---|
| 🐍 **Python** (pip) — *recomendado* | `pip install -U entroly` | Todo: herramienta CLI, servidor de comunicación con tu editor IA, y librería de código |
| 📦 **Node / npm** | `npm install -g entroly` | Mismo motor, sin necesidad de Python |
| 🦀 **Rust** (compilación desde fuente) | `cd entroly-core && cargo build --release --bin entroly-rs --features proxy` | Un programa independiente |
| 🍺 **Homebrew** | `brew install juyterman1000/entroly/entroly` | Herramienta CLI en macOS/Linux |
| 🐳 **Docker** | `docker pull ghcr.io/juyterman1000/entroly:latest` | Se ejecuta en contenedor |

```bash
cd /your/repo
entroly verify-claims
entroly simulate
```

---
## Inicio rápido

> `pip install -U entroly && entroly go` — eso es todo.

```python
from entroly import compress, compress_messages, optimize
compressed = compress(api_response, budget=2000)
messages   = compress_messages(messages, budget=30000)
context    = optimize(fragments, budget=8000, query="arreglar el bug de login")
```

---
## Benchmarks

| Benchmark | Base | Con Entroly | Retención | Ahorro de tokens |
|---|---|---|---|---|
| NeedleInAHaystack | 100% | 100% | **100%** | **99.5%** |
| LongBench (HotpotQA) | 64% | 66% | **103%** | **85.3%** |
| Berkeley Function Calling | 100% | 100% | **100%** | **79.3%** |
| SQuAD 2.0 | 80% | 72% | **90%** | **43.8%** |

**Siendo honestos:** mira la fila de SQuAD 2.0 — la precisión *bajó* (80% → 72%). La compresión es un intercambio, no magia.

---
## Características

- **Primero selecciona, después comprime** — determina qué archivos responden tu pregunta, *luego* los comprime.
- **Devuelve el original exactamente** — lo omitido se puede restaurar carácter por carácter y se verifica con una huella digital.
- **Muestra su trabajo** — un recibo por cada decisión.
- **Verifica las respuestas** — compara lo que dijo la IA contra la evidencia proporcionada, en tu máquina.

---
## Preguntas frecuentes

<details>
<summary><b>¿Esto cambiará mi código o mis archivos?</b></summary>
<br>
No. Entroly lee tus archivos y decide qué enviar a la IA. Nunca edita, mueve ni elimina nada en tu proyecto.
</details>

<details>
<summary><b>¿Mi código se sube a algún lugar?</b></summary>
<br>
No. Todo se ejecuta en tu propia máquina. No hay analíticas activadas por defecto.
</details>

---

<p align="center"><sub>Apache-2.0 · local-first · sin analíticas externas por defecto</sub></p>
<p align="center"><code>pip install entroly && entroly go</code></p>
