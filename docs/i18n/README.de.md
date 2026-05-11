<p align="center">
  <a href="../../README.md">🇬🇧 English</a> •
  <a href="README.zh-CN.md">🇨🇳 中文</a> •
  <a href="README.ja.md">🇯🇵 日本語</a> •
  <a href="README.ko.md">🇰🇷 한국어</a> •
  <a href="README.pt-BR.md">🇧🇷 Português</a> •
  <a href="README.es.md">🇪🇸 Español</a> •
  <a href="README.fr.md">🇫🇷 Français</a> •
  <a href="README.ru.md">🇷🇺 Русский</a> •
  <a href="README.hi.md">🇮🇳 हिन्दी</a> •
  <a href="README.tr.md">🇹🇷 Türkçe</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/logo.png" width="180" alt="Entroly">
</p>

<h1 align="center">Entroly — KI-Token-Kosten um 70–95% senken</h1>

<h3 align="center">Deine KI-Coding-Tools sehen nur 5% deiner Codebase.<br/>Entroly gibt ihnen das volle Bild — für einen Bruchteil der Kosten.</h3>

<p align="center">
  <a href="../../README.md#install"><b>Installation</b></a> ·
  <a href="../../cookbook/README.md"><b>Cookbook</b></a> ·
  <a href="../../README.md#benchmarks"><b>Benchmarks</b></a> ·
  <a href="../../README.md#works-with-your-stack"><b>65+ unterstützte Agenten</b></a> ·
  <a href="https://juyterman1000.github.io/entroly/docs/dashboard.html"><b>Dashboard</b></a>
</p>

<p align="center">
  <code>npm install entroly-wasm && npx entroly-wasm</code>&nbsp;&nbsp;|&nbsp;&nbsp;<code>pip install entroly && entroly go</code>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="https://juyterman1000.github.io/entroly/"><b>Live-Demo →</b></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/self_improvement.svg" alt="Entroly Selbstverbesserung" width="800">
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/token_savings.svg" alt="Entroly Profit" width="800">
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/context_quality.svg" alt="Entroly Kontextqualität" width="800">
</p>

---

## Das Problem — und die Auswirkung auf das Ergebnis

Jedes KI-Coding-Tool — Claude, Cursor, Copilot, Codex — hat denselben blinden Fleck: **Es sieht nur 5–10 Dateien gleichzeitig.** Die anderen 95% deiner Codebase sind unsichtbar.

Modelle werden immer größer — **Claude Opus 4.7** wurde gerade mit noch mehr Funktionen und noch höheren Token-Kosten veröffentlicht. Größere Kontextfenster lösen das Problem nicht; sie machen es noch schlimmer. Du bezahlst für 186.000 Tokens pro Anfrage — wovon das meiste nur duplizierter Boilerplate-Code ist.

> **Entroly behebt beide Probleme in 30 Sekunden.**

---

## Was sich am Tag 1 ändert

| Metrik | Vor Entroly | **Nach Entroly** |
|---|---|---|
| Für die KI sichtbare Dateien | 5–10 | **Dein gesamtes Repository** |
| Tokens pro Anfrage | ~186.000 | **9.300 – 55.000** |
| Monatliche KI-Ausgaben (1K Req/Tag) | ~$16.800 | **$840 – $5.040** |
| KI-Antwortgenauigkeit | Unvollständig, oft halluziniert | **Dependency-aware, korrekt** |
| Dev-Zeit für KI-Fehlerbehebung | Stunden/Woche | **Fast null** |
| Setup | Tage Prompt-Engineering | **30 Sekunden** |

> **ROI:** Ein 10-Personen-Team mit $15K/Monat KI-API-Kosten spart **$10K–$14K/Monat** am Tag 1.

---

## Was Deine Wettbewerber Bereits Wissen

Teams, die heute Entroly einsetzen, sparen nicht nur Geld — sie **bauen einen zusammengesetzten Vorteil auf**, den dein Team nicht einholen kann.

- **Woche 1:** Deren KI sieht 100% der Codebase. Deine sieht 5%. Sie liefern schneller.
- **Monat 1:** Deren Runtime hat die Codebase-Muster gelernt. Deine halluziniert noch Imports.
- **Monat 3:** Deren Installation ist an die Föderation angeschlossen — absorbiert Optimierungsstrategien von Tausenden Teams weltweit. Du weißt nicht mal, dass das existiert.
- **Monat 6:** Sie haben $80K+ API-Kosten gespart. Das Budget ging in Einstellungen. Du erklärst der Finanzabteilung noch, warum die KI-Rechnung weiter steigt.

Jeder Tag Wartezeit vergrößert die Lücke. Der Föderationseffekt bedeutet: **Early Adopters werden schneller stärker** — und dieser Vorteil wächst exponentiell.

---

## 🌐 Föderiertes Schwarmlernen — Der Teil, der wie Science-Fiction klingt

Nimm den Dreaming Loop und multipliziere ihn mit **jedem Entwickler auf der Erde, der Entroly nutzt.**

Während du schläfst, träumt dein Daemon — und 10.000 andere auch. Jeder entdeckt leicht unterschiedliche Tricks zur Code-Kompression. Jeder teilt anonym, was er gelernt hat. Jeder absorbiert, was die anderen gefunden haben.

**Du wachst auf. Deine KI ist schlauer als gestern. Nicht wegen dir — wegen dem, was der Schwarm geträumt hat.**

```
Dein Daemon träumt → entdeckt eine bessere Strategie → teilt sie (anonym)
     ↓
10.000 andere Daemons haben gestern Nacht dasselbe getan
     ↓
Du öffnest deinen Laptop → deine KI hat bereits alles absorbiert
```


**Netzwerkeffekt:**
- Jeder neue User macht die KI aller anderen besser — diese Installationsbasis kann nicht geforkt werden
- Infrastrukturkosten: **$0**. Läuft auf GitHub. Keine Server. Keine GPUs. Keine Cloud

```bash
# Opt-in — immer deine Wahl
export ENTROLY_FEDERATION=1
```

### 🔒 Läuft Lokal. Dein Code Verlässt Niemals Deine Maschine.

Funktioniert in Air-Gapped- und regulierten Umgebungen — nichts sendet jemals Daten nach außen.

---

## Benchmarks

### Genauigkeitserhalt

Komprimierung schadet der Genauigkeit nicht — mit Live-API verifiziert (gpt-4o-mini, Wilson 95% CI):

| Benchmark | n | Budget | Baseline (95% CI) | Mit Entroly (95% CI) | Erhalt | Token-Einsparung |
|---|---|---|---|---|---|---|
| NeedleInAHaystack | 20 | 2K | 100% [83.9–100%] | 100% [83.9–100%] | **100.0%** | **99.5%** |
| LongBench (HotpotQA) | 50 | 2K | 64.0% [50.1–75.9%] | 68.0% [54.2–79.2%] | **106.2%** | **85.3%** |
| Berkeley Function Calling | 50 | 500 | 100% [92.9–100%] | 100% [92.9–100%] | **100.0%** | **79.3%** |
| SQuAD 2.0 | 50 | 100 | 78.0% [64.8–87.2%] | 76.0% [62.6–85.7%] | **97.4%** | **39.3%** |
| GSM8K | 100 | 50K | 85.0% [76.7–90.7%] | 86.0% [77.9–91.5%] | **101.2%** | pass-through¹ |
| MMLU | 100 | 50K | 82.0% [73.3–88.3%] | 85.9% [77.8–91.4%] | **104.7%** | pass-through¹ |
| TruthfulQA (MC1) | 100 | 50K | 72.0% [62.5–79.9%] | 73.7% [64.3–81.4%] | **102.4%** | pass-through¹ |

> ¹ **pass-through**: Kontext passt bereits ins Budget — Entroly tut korrekterweise nichts. Die Konfidenzintervalle überlappen bei allen Benchmarks.

### Vergleich mit anderen Methoden (Langer Kontext)

| Methode | Erhalt | Token-Reduktion | Architektur / Trade-offs |
|---|---|---|---|
| **Entroly** | **100–106%** | **85–99%** | **Schnell (~80ms).** Fragment-Level Knapsack. Perfekte wortgetreue strukturelle Treue. |
| Token-Level Neural Pruning | ~98–99% | 80–95% | **Hoher Overhead.** Erfordert lokalen Transformer. Beschädigt Code-Syntax. |
| Regelbasierte Verbatim-Kompaktierung | ~100% | 50–70% | **Hohe Treue.** Aber geringere Token-Reduktion. |
| Attention-basierte Kompression | 95%+ | 26–54% | **Solide Genauigkeit.** Aber geringere Token-Reduktion. |

---

## Volle Parität: Python & Node.js

| Fähigkeit | Python | Node.js (WASM) |
|---|---|---|
| Kontextkompression | ✅ | ✅ |
| Selbstevolution | ✅ | ✅ |
| Dreaming Loop | ✅ | ✅ |
| Föderation | ✅ | ✅ |
| Antwortdestillation | ✅ | ✅ |
| Chat-Gateways | ✅ | ✅ |
| agentskills.io Export | ✅ | ✅ |

---

<p align="center">
  <b>Hör auf, für Tokens zu bezahlen, die deine KI verschwendet. Starte eine KI, die sich selbst lehrt.</b><br/>
  <code>npm install entroly-wasm && npx entroly-wasm</code>&nbsp;&nbsp;|&nbsp;&nbsp;<code>pip install entroly && entroly go</code>
</p>
