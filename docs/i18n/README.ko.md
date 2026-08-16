<p align="center">
  <img src="../../docs/assets/entroly_wordmark.svg" width="820" alt="Entroly">
</p>

<h1 align="center">Entroly — AI 운영 비용을 절감하는 드롭인 컨텍스트 보증</h1>

<p align="center"><b>중요한 증거에 대한 통제를 잃지 않으면서 불필요한 컨텍스트를 줄입니다.</b><br>
가장 가치 있는 증거를 먼저 선택하고, 압축하고, 원본을 복원 가능하게 유지하며, 영수증을 발행합니다 — 코드베이스나 에이전트 아키텍처를 다시 작성할 필요가 없습니다.</p>
<p align="center">
  <sub>Entroly는 로컬 우선 Context OS입니다: 컨텐츠 주소 방식의 증거, 복원 가능한 압축, 감사 가능한 영수증. 프록시, MCP, 플러그인, 래퍼, SDK 경로를 통해 Claude Code, Codex, OpenClaw, GitHub Copilot, Cursor, Aider 및 OpenAI/Anthropic 호환 앱과 함께 작동합니다.</sub>
</p>
<p align="center">
  <a href="https://pypi.org/project/entroly/"><img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI" alt="Entroly on PyPI"></a>
  <a href="https://www.npmjs.com/package/entroly"><img src="https://img.shields.io/npm/v/entroly?color=red&label=npm" alt="Entroly on npm"></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="Apache-2.0 license"></a>
  <a href="https://github.com/juyterman1000/entroly"><img src="https://img.shields.io/github/stars/juyterman1000/entroly?style=social" alt="Entroly GitHub stars"></a>
</p>

<p align="center"><b>100,438회 이상 다운로드 · 매일 성장 중</b><br>
<sub>다양한 배포 채널에서 측정됨.</sub></p>

<p align="center"><b>⭐ Entroly가 유용하다면 GitHub에서 저장소에 스타를 눌러주세요.</b><br>
<a href="https://github.com/juyterman1000/entroly">⭐ GitHub에서 Entroly 스타 누르기</a> — 프로젝트 성장과 더 많은 개발자에게 도달하는 데 도움이 됩니다.</p>

<p align="center">
  <b><a href="../../README.md">English</a> · <a href="README.zh.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.ja.md">日本語</a> · 한국어 · <a href="README.es.md">Español</a> · <a href="README.hi.md">हिन्दी</a> · <a href="README.fr.md">Français</a> · <a href="README.de.md">Deutsch</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.it.md">Italiano</a> · <a href="README.tr.md">Türkçe</a> · <a href="README.vi.md">Tiếng Việt</a> · <a href="README.id.md">Bahasa Indonesia</a> · <a href="README.pl.md">Polski</a> · <a href="README.nl.md">Nederlands</a> · <a href="README.th.md">ไทย</a> · <a href="README.sv.md">Svenska</a> · <a href="README.cs.md">Čeština</a> · <a href="README.tl.md">Tagalog</a> · <a href="README.ro.md">Română</a></b>
</p>

---

## Entroly란? (쉽게 설명)

AI 코딩 어시스턴트에는 메모리 한계가 있습니다. 전체 코드베이스를 넘기면 느리고, 비싸고, 산만해집니다 — 500페이지 매뉴얼을 건네주지만 실제로는 47페이지만 필요한 것과 같습니다.

**Entroly가 47페이지를 찾아줍니다.**

코드와 AI 사이에 위치하여 모든 것을 읽고, 실제로 묻고 있는 질문에 중요한 부분만 전달합니다.
|  |  |
|---|---|
| 💰 **비용이 줄어듭니다** | AI에 보내는 단어가 적으면 청구서가 작아집니다. |
| 🔍 **아무것도 잃어버리지 않습니다** | Entroly가 제외한 모든 것은 보존되며 *정확하게* 복원할 수 있습니다. |
| 🧾 **작업을 확인할 수 있습니다** | 모든 결정에 영수증이 따라옵니다: 무엇이 유지되었고, 무엇이 제외되었으며, 그 이유. |

**코드를 변경해야 하나요?** 아닙니다. Entroly는 이미 사용 중인 도구와 함께 작동합니다 — Claude Code, Cursor, Copilot 등 30개 이상의 도구 — 백그라운드에서 실행됩니다.

---
## 설치

| 플랫폼 | 설치 | 제공 기능 |
|---|---|---|
| 🐍 **Python** (pip) — *권장* | `pip install -U entroly` | 모든 기능: CLI 도구, AI 에디터 통신 서버, 코드 라이브러리 |
| 📦 **Node / npm** | `npm install -g entroly` | 동일 엔진, Python 불필요 |
| 🦀 **Rust** (소스 빌드) | `cd entroly-core && cargo build --release --bin entroly-rs --features proxy` | 독립 실행 프로그램 |
| 🍺 **Homebrew** | `brew install juyterman1000/entroly/entroly` | macOS/Linux CLI 도구 |
| 🐳 **Docker** | `docker pull ghcr.io/juyterman1000/entroly:latest` | 컨테이너에서 실행 |

```bash
cd /your/repo
entroly verify-claims
entroly simulate
```

---
## 빠른 시작

> `pip install -U entroly && entroly go` — 이게 전부입니다.

```python
from entroly import compress, compress_messages, optimize
compressed = compress(api_response, budget=2000)
messages   = compress_messages(messages, budget=30000)
context    = optimize(fragments, budget=8000, query="로그인 버그 수정")
```

---
## 벤치마크

| 벤치마크 | 기준선 | Entroly 사용 시 | 유지율 | 토큰 절약 |
|---|---|---|---|---|
| NeedleInAHaystack | 100% | 100% | **100%** | **99.5%** |
| LongBench (HotpotQA) | 64% | 66% | **103%** | **85.3%** |
| Berkeley Function Calling | 100% | 100% | **100%** | **79.3%** |
| SQuAD 2.0 | 80% | 72% | **90%** | **43.8%** |

**솔직히 말해서:** SQuAD 2.0 행을 보세요 — 정확도가 *떨어졌습니다* (80% → 72%). 압축은 트레이드오프이지 마법이 아닙니다.

---
## 기능

- **먼저 선택, 그다음 압축** — 어떤 파일이 질문에 실제로 답하는지 파악한 *후* 압축합니다.
- **원본을 정확하게 복원** — 제외된 것은 글자 하나하나 복원되며 핑거프린트로 검증됩니다.
- **작업 과정을 보여줌** — 모든 결정의 영수증: 유지된 것, 제외된 것과 이유, 남은 위험.
- **답변을 팩트체크** — AI의 답변을 제공된 증거와 비교합니다. 로컬에서 실행, 추가 AI 호출 비용 없음.

---
## 자주 묻는 질문

<details>
<summary><b>내 코드나 파일이 변경되나요?</b></summary>
<br>
아닙니다. Entroly는 파일을 읽고 AI에 무엇을 보낼지 결정합니다. 프로젝트의 어떤 것도 편집, 이동, 삭제하지 않습니다.
</details>

<details>
<summary><b>내 코드가 어딘가에 업로드되나요?</b></summary>
<br>
아닙니다. 모든 선택, 압축, 검사는 자신의 머신에서 수행됩니다. 기본적으로 분석 기능은 꺼져 있습니다.
</details>

---

<p align="center"><sub>Apache-2.0 · 로컬 우선 · 기본적으로 외부 분석 없음</sub></p>
<p align="center"><code>pip install entroly && entroly go</code></p>
