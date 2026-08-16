<p align="center">
  <img src="../../docs/assets/entroly_wordmark.svg" width="820" alt="Entroly">
</p>

<h1 align="center">Entroly — Đảm Bảo Ngữ Cảnh Drop-In Giúp Giảm Chi Phí Vận Hành AI</h1>

<p align="center"><b>Giảm ngữ cảnh không cần thiết mà không làm mất quyền kiểm soát các bằng chứng quan trọng.</b><br>
Chọn bằng chứng có giá trị cao nhất trước, nén lại, giữ nguyên bản có thể khôi phục và phát biên nhận — không cần viết lại mã nguồn hoặc kiến trúc agent.</p>
<p align="center">
  <sub>Entroly là Context OS ưu tiên cục bộ (local-first): bằng chứng định địa chỉ theo nội dung, nén có thể phục hồi và biên nhận có thể kiểm toán. Hoạt động qua proxy, MCP, plugin, wrapper và SDK với Claude Code, Codex, OpenClaw, GitHub Copilot, Cursor, Aider và các ứng dụng tương thích OpenAI/Anthropic.</sub>
</p>
<p align="center">
  <a href="https://pypi.org/project/entroly/"><img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI" alt="Entroly on PyPI"></a>
  <a href="https://www.npmjs.com/package/entroly"><img src="https://img.shields.io/npm/v/entroly?color=red&label=npm" alt="Entroly on npm"></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="Apache-2.0 license"></a>
  <a href="https://github.com/juyterman1000/entroly"><img src="https://img.shields.io/github/stars/juyterman1000/entroly?style=social" alt="Entroly GitHub stars"></a>
</p>

<p align="center"><b>100.438 lượt tải xuống · tăng trưởng từng ngày</b><br>
<sub>Được đo lường trên các nguồn phân phối khác nhau.</sub></p>

<p align="center"><b>⭐ Nếu Entroly hữu ích với bạn, hãy gắn sao cho kho lưu trữ trên GitHub.</b><br>
<a href="https://github.com/juyterman1000/entroly">⭐ Gắn sao Entroly trên GitHub</a> — giúp dự án phát triển và tiếp cận nhiều lập trình viên hơn.</p>

<p align="center">
  <b><a href="../../README.md">English</a> · <a href="README.zh.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a> · <a href="README.es.md">Español</a> · <a href="README.hi.md">हिन्दी</a> · <a href="README.fr.md">Français</a> · <a href="README.de.md">Deutsch</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.it.md">Italiano</a> · <a href="README.tr.md">Türkçe</a> · Tiếng Việt · <a href="README.id.md">Bahasa Indonesia</a> · <a href="README.pl.md">Polski</a> · <a href="README.nl.md">Nederlands</a> · <a href="README.th.md">ไทย</a> · <a href="README.sv.md">Svenska</a> · <a href="README.cs.md">Čeština</a> · <a href="README.tl.md">Tagalog</a> · <a href="README.ro.md">Română</a></b>
</p>

---

## Entroly là gì? (Giải thích đơn giản)

Trợ lý lập trình AI có giới hạn bộ nhớ. Giao toàn bộ mã nguồn cho AI khiến nó trở nên chậm, đắt đỏ và phân tâm — giống như đưa cho ai đó cuốn sổ tay 500 trang khi họ chỉ cần trang 47.

**Entroly tìm trang 47 cho bạn.**

Nó nằm giữa mã nguồn của bạn và AI, đọc mọi thứ, và chỉ chuyển tiếp những phần quan trọng cho câu hỏi đang được hỏi.
|  |  |
|---|---|
| 💰 **Hóa đơn của bạn giảm xuống** | Gửi ít văn bản hơn đến AI đồng nghĩa với hóa đơn nhỏ hơn. |
| 🔍 **Không có gì bị mất** | Mọi thứ Entroly gác lại đều được lưu giữ và có thể phục hồi *chính xác* như ban đầu. |
| 🧾 **Bạn có thể kiểm tra công việc** | Mỗi quyết định đều đi kèm một biên nhận: những gì được giữ lại, những gì bị loại bỏ và tại sao. |

**Tôi có phải thay đổi mã nguồn không?** Không. Entroly hoạt động với các công cụ bạn đã sử dụng — Claude Code, Cursor, Copilot và hơn 30 công cụ khác.

---
## Cài đặt

| Nền tảng | Cài đặt | Những gì bạn nhận được |
|---|---|---|
| 🐍 **Python** (pip) — *khuyên dùng* | `pip install -U entroly` | Đầy đủ: công cụ CLI, máy chủ kết nối editor, thư viện mã nguồn |
| 📦 **Node / npm** | `npm install -g entroly` | Cùng engine, không cần Python |
| 🦀 **Rust** (bản dựng mã nguồn) | `cd entroly-core && cargo build --release --bin entroly-rs --features proxy` | Một chương trình độc lập duy nhất |
| 🍺 **Homebrew** | `brew install juyterman1000/entroly/entroly` | Công cụ dòng lệnh trên macOS/Linux |
| 🐳 **Docker** | `docker pull ghcr.io/juyterman1000/entroly:latest` | Chạy trong container |

```bash
cd /your/repo
entroly verify-claims
entroly simulate
```

---
## Bắt đầu nhanh

> `pip install -U entroly && entroly go` — chỉ đơn giản như vậy.

```python
from entroly import compress, compress_messages, optimize
compressed = compress(api_response, budget=2000)
messages   = compress_messages(messages, budget=30000)
context    = optimize(fragments, budget=8000, query="sửa lỗi đăng nhập")
```

---
## Benchmarks

| Benchmark | Cơ sở | Với Entroly | Tỷ lệ duy trì | Tiết kiệm token |
|---|---|---|---|---|
| NeedleInAHaystack | 100% | 100% | **100%** | **99.5%** |
| LongBench (HotpotQA) | 64% | 66% | **103%** | **85.3%** |
| Berkeley Function Calling | 100% | 100% | **100%** | **79.3%** |
| SQuAD 2.0 | 80% | 72% | **90%** | **43.8%** |

---
## Tính năng

- **Chọn trước, nén sau** — xác định tệp nào thực sự trả lời câu hỏi của bạn, *sau đó* mới nén.
- **Khôi phục bản gốc chính xác** — bất cứ điều gì bị bỏ sót đều có thể phục hồi từng ký tự.
- **Minh bạch công việc** — biên nhận cho mọi quyết định.
- **Kiểm tra tính xác thực câu trả lời** — so sánh câu trả lời của AI với bằng chứng được cung cấp, hoàn toàn cục bộ.

---

<p align="center"><sub>Apache-2.0 · ưu tiên cục bộ · mặc định không gửi dữ liệu phân tích ra ngoài</sub></p>
<p align="center"><code>pip install entroly && entroly go</code></p>
