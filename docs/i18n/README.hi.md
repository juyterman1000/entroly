<p align="center">
  <img src="../../docs/assets/entroly_wordmark.svg" width="820" alt="Entroly">
</p>

<h1 align="center">Entroly — AI परिचालन लागत कम करने के लिए ड्रॉप-इन कॉन्टेक्स्ट एश्योरेंस</h1>

<p align="center"><b>महत्वपूर्ण साक्ष्य पर नियंत्रण खोए बिना अनावश्यक कॉन्टेक्स्ट कम करें।</b><br>
सबसे मूल्यवान साक्ष्य पहले चुनें, उसे कंप्रेस करें, मूल को रिकवर करने योग्य रखें, और रसीद जारी करें — बिना कोडबेस या एजेंट आर्किटेक्चर को दोबारा लिखे।</p>
<p align="center">
  <sub>Entroly एक लोकल-फर्स्ट Context OS है: कंटेंट-एड्रेस्ड साक्ष्य, रिकवर करने योग्य कंप्रेशन, और ऑडिट योग्य रसीदें। प्रॉक्सी, MCP, प्लगइन, रैपर और SDK पाथ के माध्यम से Claude Code, Codex, OpenClaw, GitHub Copilot, Cursor, Aider और OpenAI/Anthropic-संगत ऐप्स के साथ काम करता है।</sub>
</p>
<p align="center">
  <a href="https://pypi.org/project/entroly/"><img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI" alt="Entroly on PyPI"></a>
  <a href="https://www.npmjs.com/package/entroly"><img src="https://img.shields.io/npm/v/entroly?color=red&label=npm" alt="Entroly on npm"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="Apache-2.0 license"></a>
  <a href="https://github.com/juyterman1000/entroly"><img src="https://img.shields.io/github/stars/juyterman1000/entroly?style=social" alt="Entroly GitHub stars"></a>
</p>

<p align="center"><b>100,438 डाउनलोड · दिन-प्रतिदिन बढ़ता हुआ</b><br>
<sub>विभिन्न वितरण स्रोतों से मापा गया।</sub></p>

<p align="center"><b>⭐ अगर Entroly आपके लिए उपयोगी है, तो कृपया GitHub पर रिपॉजिटरी को स्टार करें।</b><br>
<a href="https://github.com/juyterman1000/entroly">⭐ GitHub पर Entroly को स्टार करें</a> — इससे प्रोजेक्ट को बढ़ने और अधिक डेवलपर्स तक पहुंचने में मदद मिलती है।</p>

<p align="center">
  <b><a href="../../README.md">English</a> · <a href="README.zh.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a> · <a href="README.es.md">Español</a> · हिन्दी · <a href="README.fr.md">Français</a> · <a href="README.de.md">Deutsch</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.it.md">Italiano</a> · <a href="README.tr.md">Türkçe</a> · <a href="README.vi.md">Tiếng Việt</a> · <a href="README.id.md">Bahasa Indonesia</a> · <a href="README.uk.md">Українська</a> · <a href="README.pl.md">Polski</a></b>
</p>

---

## Entroly क्या है? (सरल भाषा में)

AI कोडिंग असिस्टेंट की मेमोरी सीमित होती है। पूरा कोडबेस देने पर यह धीमा, महंगा और भटका हुआ हो जाता है — जैसे किसी को 500 पन्नों की मैनुअल देना जबकि उन्हें सिर्फ पन्ना 47 चाहिए था।

**Entroly पन्ना 47 ढूंढता है।**

यह आपके कोड और AI के बीच बैठता है, सब कुछ पढ़ता है, और सिर्फ वही हिस्से भेजता है जो पूछे गए सवाल के लिए मायने रखते हैं।
|  |  |
|---|---|
| 💰 **आपका बिल कम होगा** | AI को कम शब्द भेजने का मतलब है छोटा बिल। |
| 🔍 **कुछ भी नहीं खोता** | Entroly जो भी अलग रखता है वह सुरक्षित रहता है और *हूबहू* वापस लाया जा सकता है। |
| 🧾 **आप इसका काम जांच सकते हैं** | हर फैसले के साथ एक रसीद: क्या रखा गया, क्या छोड़ा गया, और क्यों। |

**क्या मुझे अपना कोड बदलना होगा?** नहीं। Entroly उन टूल्स के साथ काम करता है जो आप पहले से इस्तेमाल कर रहे हैं।

**क्या आज़माने के लिए कुछ पे करना होगा?** नहीं। नीचे [इंस्टॉल](#इंस्टॉल) सेक्शन के दो कमांड पूरी तरह आपकी मशीन पर चलते हैं, बिना API कुंजी के।

---
## इंस्टॉल

| प्लेटफॉर्म | इंस्टॉल | आपको क्या मिलता है |
|---|---|---|
| 🐍 **Python** (pip) — *अनुशंसित* | `pip install -U entroly` | सब कुछ: CLI टूल, AI एडिटर सर्वर, कोड लाइब्रेरी |
| 📦 **Node / npm** | `npm install -g entroly` | वही इंजन, Python की जरूरत नहीं |
| 🦀 **Rust** (सोर्स बिल्ड) | `cd entroly-core && cargo build --release --bin entroly-rs --features proxy` | एक स्वतंत्र प्रोग्राम |
| 🍺 **Homebrew** | `brew install juyterman1000/entroly/entroly` | macOS/Linux पर CLI टूल |
| 🐳 **Docker** | `docker pull ghcr.io/juyterman1000/entroly:latest` | कंटेनर में चलता है |

```bash
cd /your/repo
entroly verify-claims
entroly simulate
```

---
## क्विक स्टार्ट

> `pip install -U entroly && entroly go` — बस इतना ही।

```python
from entroly import compress, compress_messages, optimize
compressed = compress(api_response, budget=2000)
messages   = compress_messages(messages, budget=30000)
context    = optimize(fragments, budget=8000, query="लॉगिन बग ठीक करें")
```

---
## बेंचमार्क

| बेंचमार्क | बेसलाइन | Entroly के साथ | रिटेंशन | टोकन बचत |
|---|---|---|---|---|
| NeedleInAHaystack | 100% | 100% | **100%** | **99.5%** |
| LongBench (HotpotQA) | 64% | 66% | **103%** | **85.3%** |
| Berkeley Function Calling | 100% | 100% | **100%** | **79.3%** |
| SQuAD 2.0 | 80% | 72% | **90%** | **43.8%** |

**ईमानदारी से:** SQuAD 2.0 की पंक्ति देखें — सटीकता *गिरी* (80% → 72%)। कंप्रेशन एक ट्रेड-ऑफ है, जादू नहीं।

---
## विशेषताएं

- **पहले चयन, फिर कंप्रेशन** — पहले पता लगाता है कौन सी फाइलें सवाल का जवाब देती हैं, *फिर* उन्हें कंप्रेस करता है।
- **मूल को हूबहू वापस देता है** — छोड़ा गया कुछ भी अक्षर-दर-अक्षर रिस्टोर हो सकता है।
- **अपना काम दिखाता है** — हर फैसले की रसीद।
- **जवाबों की फैक्ट-चेकिंग** — AI के जवाब को दिए गए साक्ष्य से तुलना करता है, आपकी मशीन पर।

---

<p align="center"><sub>Apache-2.0 · लोकल-फर्स्ट · डिफॉल्ट में कोई बाहरी एनालिटिक्स नहीं</sub></p>
<p align="center"><code>pip install entroly && entroly go</code></p>
