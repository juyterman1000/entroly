<p align="center">
  <img src="../../docs/assets/entroly_wordmark.svg" width="820" alt="Entroly">
</p>

<h1 align="center">Entroly — Jaminan Konteks Drop-In untuk Menurunkan Biaya Operasional AI</h1>

<p align="center"><b>Kurangi konteks yang tidak perlu tanpa kehilangan kendali atas bukti penting.</b><br>
Pilih bukti bernilai tertinggi terlebih dahulu, kompres, pertahankan dokumen asli dapat dipulihkan, dan terbitkan tanda terima — tanpa menulis ulang basis kode atau arsitektur agen Anda.</p>
<p align="center">
  <sub>Entroly adalah Context OS lokal-utama (local-first): bukti dialamatkan berdasarkan konten, kompresi yang dapat dipulihkan, dan tanda terima yang dapat diaudit. Bekerja melalui proxy, MCP, plugin, wrapper, dan jalur SDK dengan Claude Code, Codex, OpenClaw, GitHub Copilot, Cursor, Aider, dan aplikasi yang kompatibel dengan OpenAI/Anthropic.</sub>
</p>
<p align="center">
  <a href="https://pypi.org/project/entroly/"><img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI" alt="Entroly on PyPI"></a>
  <a href="https://www.npmjs.com/package/entroly"><img src="https://img.shields.io/npm/v/entroly?color=red&label=npm" alt="Entroly on npm"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="Apache-2.0 license"></a>
  <a href="https://github.com/juyterman1000/entroly"><img src="https://img.shields.io/github/stars/juyterman1000/entroly?style=social" alt="Entroly GitHub stars"></a>
</p>

<p align="center"><b>100.438 unduhan · berkembang dari hari ke hari</b><br>
<sub>Diukur di berbagai sumber distribusi.</sub></p>

<p align="center"><b>⭐ Jika Entroly berguna bagi Anda, beri bintang pada repositori di GitHub.</b><br>
<a href="https://github.com/juyterman1000/entroly">⭐ Beri bintang pada Entroly di GitHub</a> — membantu proyek berkembang dan menjangkau lebih banyak pengembang.</p>

<p align="center">
  <b><a href="../../README.md">English</a> · <a href="README.zh.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a> · <a href="README.es.md">Español</a> · <a href="README.hi.md">हिन्दी</a> · <a href="README.fr.md">Français</a> · <a href="README.de.md">Deutsch</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.it.md">Italiano</a> · <a href="README.tr.md">Türkçe</a> · <a href="README.vi.md">Tiếng Việt</a> · Bahasa Indonesia · <a href="README.uk.md">Українська</a> · <a href="README.pl.md">Polski</a></b>
</p>

---

## Apa itu Entroly? (Penjelasan sederhana)

Asisten pengkodean AI memiliki batas memori. Berikan seluruh basis kode Anda dan asisten akan menjadi lambat, mahal, dan terganggu — seperti memberi seseorang manual 500 halaman padahal mereka hanya membutuhkan halaman 47.

**Entroly menemukan halaman 47.**

Entroly berada di antara kode Anda dan AI, membaca semuanya, dan hanya meneruskan bagian yang penting untuk pertanyaan yang diajukan.
|  |  |
|---|---|
| 💰 **Tagihan Anda turun** | Lebih sedikit teks yang dikirim ke AI berarti tagihan yang lebih kecil. |
| 🔍 **Tidak ada yang hilang** | Apa pun yang disisihkan Entroly disimpan dan dapat dipulihkan *persis* seperti aslinya. |
| 🧾 **Anda dapat memeriksa pekerjaannya** | Setiap keputusan dilengkapi dengan tanda terima: apa yang disimpan, apa yang ditinggalkan, dan mengapa. |

**Apakah saya harus mengubah kode saya?** Tidak. Entroly bekerja dengan alat yang sudah Anda gunakan — Claude Code, Cursor, Copilot, dan 30+ lainnya.

---
## Instalasi

| Platform | Instalasi | Yang Anda Dapatkan |
|---|---|---|
| 🐍 **Python** (pip) — *direkomendasikan* | `pip install -U entroly` | Semuanya: alat CLI, server editor AI, perpustakaan kode |
| 📦 **Node / npm** | `npm install -g entroly` | Mesin yang sama, tanpa Python |
| 🦀 **Rust** (build dari sumber) | `cd entroly-core && cargo build --release --bin entroly-rs --features proxy` | Satu program mandiri |
| 🍺 **Homebrew** | `brew install juyterman1000/entroly/entroly` | Alat CLI di macOS/Linux |
| 🐳 **Docker** | `docker pull ghcr.io/juyterman1000/entroly:latest` | Berjalan dalam kontainer |

```bash
cd /your/repo
entroly verify-claims
entroly simulate
```

---
## Mulai Cepat

> `pip install -U entroly && entroly go` — itu saja.

```python
from entroly import compress, compress_messages, optimize
compressed = compress(api_response, budget=2000)
messages   = compress_messages(messages, budget=30000)
context    = optimize(fragments, budget=8000, query="perbaiki bug login")
```

---
## Tolok Ukur (Benchmarks)

| Tolok Ukur | Garis Dasar | Dengan Entroly | Retensi | Penghematan Token |
|---|---|---|---|---|
| NeedleInAHaystack | 100% | 100% | **100%** | **99.5%** |
| LongBench (HotpotQA) | 64% | 66% | **103%** | **85.3%** |
| Berkeley Function Calling | 100% | 100% | **100%** | **79.3%** |
| SQuAD 2.0 | 80% | 72% | **90%** | **43.8%** |

---
## Fitur

- **Pilih dulu, kompres kemudian** — mencari tahu berkas mana yang benar-benar menjawab pertanyaan Anda, *lalu* mengompresnya.
- **Mengembalikan yang asli secara tepat** — apa pun yang dihilangkan dapat dipulihkan karakter demi karakter.
- **Menunjukkan hasil kerjanya** — tanda terima untuk setiap keputusan.
- **Pemeriksaan fakta jawaban** — membandingkan jawaban AI dengan bukti yang diberikan, secara lokal.

---

<p align="center"><sub>Apache-2.0 · lokal-pertama · tanpa analitik keluar secara default</sub></p>
<p align="center"><code>pip install entroly && entroly go</code></p>
