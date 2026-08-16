<p align="center">
  <img src="../../docs/assets/entroly_wordmark.svg" width="820" alt="Entroly">
</p>

<h1 align="center">Entroly — การรับประกันบริบทแบบ Drop-In เพื่อลดต้นทุนการดำเนินงานของ AI</h1>

<p align="center"><b>ลดบริบทที่ไม่จำเป็นโดยไม่สูญเสียการควบคุมหลักฐานสำคัญ</b><br>
เลือกหลักฐานที่มีมูลค่าสูงสุดก่อน บีบอัด เก็บรักษาต้นฉบับให้สามารถกู้คืนได้ และออกใบเสร็จ — โดยไม่ต้องเขียนโค้ดเบสหรือสถาปัตยกรรมของ Agent ใหม่</p>
<p align="center">
  <sub>Entroly เป็น Context OS ที่เน้นการทำงานในเครื่อง (local-first): ระบุหลักฐานตามเนื้อหา บีบอัดแบบกู้คืนได้ และมีใบเสร็จที่ตรวจสอบได้ ทำงานผ่าน Proxy, MCP, Plugin, Wrapper และ SDK ร่วมกับ Claude Code, Codex, OpenClaw, GitHub Copilot, Cursor, Aider และแอปที่รองรับ OpenAI/Anthropic</sub>
</p>
<p align="center">
  <a href="https://pypi.org/project/entroly/"><img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI" alt="Entroly on PyPI"></a>
  <a href="https://www.npmjs.com/package/entroly"><img src="https://img.shields.io/npm/v/entroly?color=red&label=npm" alt="Entroly on npm"></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="Apache-2.0 license"></a>
  <a href="https://github.com/juyterman1000/entroly"><img src="https://img.shields.io/github/stars/juyterman1000/entroly?style=social" alt="Entroly GitHub stars"></a>
</p>

<p align="center"><b>100,438 ดาวน์โหลด · เติบโตขึ้นทุกวัน</b><br>
<sub>วัดจากแหล่งแจกจ่ายต่างๆ</sub></p>

<p align="center"><b>⭐ หาก Entroly มีประโยชน์สำหรับคุณ โปรดให้ดาวบน GitHub</b><br>
<a href="https://github.com/juyterman1000/entroly">⭐ ให้ดาว Entroly บน GitHub</a> — ช่วยให้โครงการเติบโตและเข้าถึงนักพัฒนาได้มากขึ้น</p>

<p align="center">
  <b><a href="../../README.md">English</a> · <a href="README.zh.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a> · <a href="README.es.md">Español</a> · <a href="README.hi.md">हिन्दी</a> · <a href="README.fr.md">Français</a> · <a href="README.de.md">Deutsch</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.it.md">Italiano</a> · <a href="README.tr.md">Türkçe</a> · <a href="README.vi.md">Tiếng Việt</a> · <a href="README.id.md">Bahasa Indonesia</a> · <a href="README.pl.md">Polski</a> · <a href="README.nl.md">Nederlands</a> · ไทย · <a href="README.sv.md">Svenska</a> · <a href="README.cs.md">Čeština</a> · <a href="README.tl.md">Tagalog</a> · <a href="README.ro.md">Română</a></b>
</p>

---

## Entroly คืออะไร? (อธิบายเข้าใจง่าย)

ผู้ช่วยเขียนโค้ด AI มีขีดจำกัดหน่วยความจำ หากส่งโค้ดทั้งหมดไป AI จะทำงานช้าลง ค่าใช้จ่ายสูงขึ้น และเสียสมาธิ — เหมือนกับการส่งคู่มือ 500 หน้าให้ใครสักคนเมื่อเขาต้องการเพียงหน้า 47

**Entroly ค้นหาหน้า 47 ให้คุณ**

Entroly อยู่ระหว่างโค้ดของคุณกับ AI อ่านทุกอย่าง และส่งเฉพาะส่วนที่เกี่ยวข้องกับคำถามที่ถูกถามจริงๆ
|  |  |
|---|---|
| 💰 **ค่าบริการของคุณลดลง** | ส่งข้อความไปยัง AI น้อยลง หมายถึงค่าใช้จ่ายที่ลดลง |
| 🔍 **ไม่มีอะไรสูญหาย** | ทุกอย่างที่ Entroly พักไว้จะถูกเก็บรักษาและสามารถกู้คืนได้ *เหมือนเดิมทุกประการ* |
| 🧾 **คุณสามารถตรวจสอบการทำงานได้** | ทุกการตัดสินใจมาพร้อมกับใบเสร็จ: อะไรถูกเก็บไว้ อะไรถูกละเว้น และเพราะอะไร |

**ฉันต้องเปลี่ยนโค้ดหรือไม่?** ไม่ต้อง Entroly ทำงานร่วมกับเครื่องมือที่คุณใช้อยู่แล้ว — Claude Code, Cursor, Copilot และอื่นๆ อีกกว่า 30 รายการ

---
## การติดตั้ง

| แพลตฟอร์ม | คำสั่งติดตั้ง | สิ่งที่คุณจะได้รับ |
|---|---|---|
| 🐍 **Python** (pip) — *แนะนำ* | `pip install -U entroly` | ครบถ้วน: เครื่องมือ CLI, เซิร์ฟเวอร์สำหรับ AI Editor, ไลบรารีโค้ด |
| 📦 **Node / npm** | `npm install -g entroly` | เอ็นจินเดียวกัน ไม่ต้องใช้ Python |
| 🦀 **Rust** (คอมไพล์จากซอร์ส) | `cd entroly-core && cargo build --release --bin entroly-rs --features proxy` | โปรแกรมเดี่ยวทำงานได้ในตัวเอง |
| 🍺 **Homebrew** | `brew install juyterman1000/entroly/entroly` | เครื่องมือ CLI บน macOS/Linux |
| 🐳 **Docker** | `docker pull ghcr.io/juyterman1000/entroly:latest` | รันในคอนเทนเนอร์ |

```bash
cd /your/repo
entroly verify-claims
entroly simulate
```

---
## เริ่มต้นใช้งานด่วน

> `pip install -U entroly && entroly go` — แค่นั้นเลย

```python
from entroly import compress, compress_messages, optimize
compressed = compress(api_response, budget=2000)
messages   = compress_messages(messages, budget=30000)
context    = optimize(fragments, budget=8000, query="แก้ไขบั๊กการเข้าสู่ระบบ")
```

---
## การเปรียบเทียบประสิทธิภาพ (Benchmarks)

| การทดสอบ | ค่าพื้นฐาน | เมื่อใช้ Entroly | การรักษาความแม่นยำ | การประหยัดโทเค็น |
|---|---|---|---|---|
| NeedleInAHaystack | 100% | 100% | **100%** | **99.5%** |
| LongBench (HotpotQA) | 64% | 66% | **103%** | **85.3%** |
| Berkeley Function Calling | 100% | 100% | **100%** | **79.3%** |
| SQuAD 2.0 | 80% | 72% | **90%** | **43.8%** |

---
## คุณสมบัติเด่น

- **เลือกก่อน บีบอัดทีหลัง** — หาว่าไฟล์ใดตอบคำถามได้จริง *แล้วจึง* บีบอัด
- **กู้คืนต้นฉบับได้อย่างสมบูรณ์** — สิ่งที่ถูกละเว้นสามารถกู้คืนได้อักขระต่ออักขระ
- **โปร่งใสตรวจสอบได้** — มีใบเสร็จสำหรับทุกการตัดสินใจ
- **ตรวจสอบความถูกต้องของคำตอบ** — เปรียบเทียบคำตอบของ AI กับหลักฐานที่ให้มาในเครื่องของคุณ

---

<p align="center"><sub>Apache-2.0 · เน้นทำงานในเครื่อง · ไม่มีการส่งข้อมูลวิเคราะห์ออกภายนอกโดยค่าเริ่มต้น</sub></p>
<p align="center"><code>pip install entroly && entroly go</code></p>
