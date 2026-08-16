<p align="center">
  <img src="../../docs/assets/entroly_wordmark.svg" width="820" alt="Entroly">
</p>

<h1 align="center">Entroly — Yapay Zeka Operasyon Maliyetlerini Düşüren Tak-Çalıştır Bağlam Güvencesi</h1>

<p align="center"><b>Kritik kanıtların kontrolünü kaybetmeden gereksiz bağlamı azaltın.</b><br>
Önce en değerli kanıtları seçin, sıkıştırın, orijinalleri kurtarılabilir tutun ve bir makbuz oluşturun — kod tabanınızı veya ajan mimarinizi yeniden yazmanıza gerek yok.</p>
<p align="center">
  <sub>Entroly yerel öncelikli (local-first) bir Context OS'tur: içerik adresli kanıt, kurtarılabilir sıkıştırma ve denetlenebilir makbuzlar. Claude Code, Codex, OpenClaw, GitHub Copilot, Cursor, Aider ve OpenAI/Anthropic uyumlu uygulamalarla proxy, MCP, eklenti, wrapper ve SDK yolları üzerinden çalışır.</sub>
</p>
<p align="center">
  <a href="https://pypi.org/project/entroly/"><img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI" alt="Entroly on PyPI"></a>
  <a href="https://www.npmjs.com/package/entroly"><img src="https://img.shields.io/npm/v/entroly?color=red&label=npm" alt="Entroly on npm"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="Apache-2.0 license"></a>
  <a href="https://github.com/juyterman1000/entroly"><img src="https://img.shields.io/github/stars/juyterman1000/entroly?style=social" alt="Entroly GitHub stars"></a>
</p>

<p align="center"><b>100.438 indirme · her geçen gün büyüyor</b><br>
<sub>Farklı dağıtım kaynaklarından ölçülmüştür.</sub></p>

<p align="center"><b>⭐ Entroly işinize yaradıysa, GitHub'da depoya yıldız verin.</b><br>
<a href="https://github.com/juyterman1000/entroly">⭐ GitHub'da Entroly'ye yıldız ver</a> — projenin büyümesine ve daha fazla geliştiriciye ulaşmasına yardımcı olur.</p>

<p align="center">
  <b><a href="../../README.md">English</a> · <a href="README.zh.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a> · <a href="README.es.md">Español</a> · <a href="README.hi.md">हिन्दी</a> · <a href="README.fr.md">Français</a> · <a href="README.de.md">Deutsch</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.it.md">Italiano</a> · Türkçe · <a href="README.vi.md">Tiếng Việt</a> · <a href="README.id.md">Bahasa Indonesia</a> · <a href="README.uk.md">Українська</a> · <a href="README.pl.md">Polski</a></b>
</p>

---

## Entroly Nedir? (Yalın dille)

Yapay zeka kodlama asistanlarının bellek sınırı vardır. Kod tabanınızın tamamını verdiğinizde yavaşlar, pahalılaşır ve dikkati dağılır — birine yalnızca 47. sayfaya ihtiyacı varken 500 sayfalık bir kılavuz vermek gibi.

**Entroly 47. sayfayı bulur.**

Kodunuzla yapay zeka arasında yer alır, her şeyi okur ve yalnızca sorulan soru için gerçekten önemli olan kısımları iletir.
|  |  |
|---|---|
| 💰 **Faturanız düşer** | Yapay zekaya daha az metin göndermek, daha küçük bir fatura anlamına gelir. |
| 🔍 **Hiçbir şey kaybolmaz** | Entroly'nin bir kenara koyduğu her şey saklanır ve *birebir* kurtarılabilir. |
| 🧾 **Çalışmasını kontrol edebilirsiniz** | Her kararın bir makbuzu vardır: ne saklandı, ne hariç tutuldu ve neden. |

**Kodumu değiştirmem gerekiyor mu?** Hayır. Entroly zaten kullandığınız araçlarla çalışır — Claude Code, Cursor, Copilot ve 30+ diğer araç.

---
## Kurulum

| Platform | Kurulum | Ne Elde Edersiniz |
|---|---|---|
| 🐍 **Python** (pip) — *önerilen* | `pip install -U entroly` | Her şey: CLI aracı, AI editör sunucusu, kod kütüphanesi |
| 📦 **Node / npm** | `npm install -g entroly` | Aynı motor, Python gerektirmez |
| 🦀 **Rust** (kaynaktan derleme) | `cd entroly-core && cargo build --release --bin entroly-rs --features proxy` | Bağımsız tek bir program |
| 🍺 **Homebrew** | `brew install juyterman1000/entroly/entroly` | macOS/Linux üzerinde CLI aracı |
| 🐳 **Docker** | `docker pull ghcr.io/juyterman1000/entroly:latest` | Konteyner içinde çalışır |

```bash
cd /your/repo
entroly verify-claims
entroly simulate
```

---
## Hızlı Başlangıç

> `pip install -U entroly && entroly go` — hepsi bu.

```python
from entroly import compress, compress_messages, optimize
compressed = compress(api_response, budget=2000)
messages   = compress_messages(messages, budget=30000)
context    = optimize(fragments, budget=8000, query="giriş hatasını düzelt")
```

---
## Kıyaslamalar (Benchmarks)

| Kıyaslama | Temel | Entroly ile | Koruma | Token Tasarrufu |
|---|---|---|---|---|
| NeedleInAHaystack | 100% | 100% | **100%** | **99.5%** |
| LongBench (HotpotQA) | 64% | 66% | **103%** | **85.3%** |
| Berkeley Function Calling | 100% | 100% | **100%** | **79.3%** |
| SQuAD 2.0 | 80% | 72% | **90%** | **43.8%** |

---
## Özellikler

- **Önce seçer, sonra sıkıştırır** — sorunuza hangi dosyaların gerçekten yanıt verdiğini belirler, *ardından* sıkıştırır.
- **Orijinali birebir geri verir** — hariç tutulan her şey karakteri karakterine kurtarılabilir.
- **Çalışmasını şeffafça gösterir** — her karar için makbuz.
- **Yanıtları doğrular** — yapay zekanın yanıtını verilen kanıtlarla yerel olarak karşılaştırır.

---

<p align="center"><sub>Apache-2.0 · yerel öncelikli · varsayılan olarak harici analiz yok</sub></p>
<p align="center"><code>pip install entroly && entroly go</code></p>
