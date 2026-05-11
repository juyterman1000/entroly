<p align="center">
  <a href="../../README.md">🇬🇧 English</a> •
  <a href="README.zh-CN.md">🇨🇳 中文</a> •
  <a href="README.ja.md">🇯🇵 日本語</a> •
  <a href="README.ko.md">🇰🇷 한국어</a> •
  <a href="README.es.md">🇪🇸 Español</a> •
  <a href="README.de.md">🇩🇪 Deutsch</a> •
  <a href="README.fr.md">🇫🇷 Français</a> •
  <a href="README.ru.md">🇷🇺 Русский</a> •
  <a href="README.hi.md">🇮🇳 हिन्दी</a> •
  <a href="README.tr.md">🇹🇷 Türkçe</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/logo.png" width="180" alt="Entroly">
</p>

<h1 align="center">Entroly — Corte Custos de Tokens IA em 70–95%</h1>

<h3 align="center">Suas ferramentas de IA só enxergam 5% do seu código.<br/>O Entroly mostra o panorama completo — por uma fração do custo.</h3>

<p align="center">
  <a href="../../README.md#install"><b>Instalar</b></a> ·
  <a href="../../cookbook/README.md"><b>Cookbook</b></a> ·
  <a href="../../README.md#benchmarks"><b>Benchmarks</b></a> ·
  <a href="../../README.md#works-with-your-stack"><b>65+ agentes suportados</b></a> ·
  <a href="https://juyterman1000.github.io/entroly/docs/dashboard.html"><b>Dashboard</b></a>
</p>

<p align="center">
  <code>npm install entroly-wasm && npx entroly-wasm</code>&nbsp;&nbsp;|&nbsp;&nbsp;<code>pip install entroly && entroly go</code>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="https://juyterman1000.github.io/entroly/"><b>Demo ao vivo →</b></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/self_improvement.svg" alt="Entroly auto-melhoria" width="800">
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/token_savings.svg" alt="Entroly lucro" width="800">
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/context_quality.svg" alt="Entroly qualidade do contexto" width="800">
</p>

---

## O Problema — e o Impacto nos Resultados

Toda ferramenta de IA — Claude, Cursor, Copilot, Codex — tem o mesmo ponto cego: **só vê 5–10 arquivos por vez.** Os outros 95% do código são invisíveis.

Os modelos continuam crescendo — o **Claude Opus 4.7** acabou de ser lançado com ainda mais capacidade e custos por token ainda mais altos. Janelas de contexto maiores não resolvem o problema; elas o agravam. Você está pagando por 186.000 tokens por requisição — e a maior parte disso é código boilerplate duplicado.

> **O Entroly resolve os dois problemas em 30 segundos.**

---

## O Que Muda no Dia 1

| Métrica | Antes do Entroly | **Depois do Entroly** |
|---|---|---|
| Arquivos visíveis à IA | 5–10 | **Todo o repositório** |
| Tokens por requisição | ~186.000 | **9.300 – 55.000** |
| Gasto mensal com IA (1K req/dia) | ~$16.800 | **$840 – $5.040** |
| Precisão das respostas da IA | Incompleta, alucinações | **Ciente de dependências, correta** |
| Tempo do dev corrigindo erros da IA | Horas/semana | **Quase zero** |
| Setup | Dias de engenharia de prompts | **30 segundos** |

> **ROI:** Um time de 10 pessoas gastando $15K/mês em APIs de IA economiza **$10K–$14K/mês** no dia 1.

---

## O Que Seus Concorrentes Já Sabem

Times que adotam o Entroly hoje não estão apenas economizando — estão **acumulando uma vantagem composta** que seu time não consegue alcançar.

- **Semana 1:** A IA deles vê 100% do código. A sua vê 5%. Eles entregam mais rápido.
- **Mês 1:** O runtime deles aprendeu os padrões do código. O seu ainda alucina imports.
- **Mês 3:** A instalação deles está plugada na federação — absorvendo estratégias de otimização de milhares de times globalmente. Você nem sabe que isso existe.
- **Mês 6:** Eles economizaram $80K+ em custos de API. Esse budget foi para contratações. Você ainda está explicando pro financeiro por que a conta de IA continua subindo.

Cada dia que você espera, a diferença cresce. O efeito federação significa que **early adopters ficam mais fortes mais rápido** — e essa vantagem gera juros compostos.

---

## Como Funciona (30 Segundos)

```bash
npm install entroly-wasm && npx entroly-wasm
# ou
pip install entroly && entroly go
```

Pronto. O Entroly detecta sua IDE, conecta ao Claude/Cursor/Copilot/Codex/MiniMax e começa a otimizar.

---

## 🌐 Aprendizado Federado em Enxame — A Parte Que Parece Ficção Científica

Agora pegue o Dreaming Loop e multiplique por **cada desenvolvedor na Terra que roda Entroly.**

Enquanto você dorme, seu daemon sonha — e 10.000 outros também. Cada um descobre truques ligeiramente diferentes de compressão de código. Cada um compartilha o que aprendeu — anonimamente, com privacidade. Cada um absorve o que os outros encontraram.

**Você acorda. Sua IA está mais inteligente do que quando você saiu. Não por algo que você fez — pelo que o enxame sonhou.**

```
Seu daemon sonha → descobre uma estratégia melhor → compartilha (anonimamente)
     ↓
10.000 outros daemons fizeram o mesmo ontem à noite
     ↓
Você abre o laptop → sua IA já absorveu tudo
```


**Efeito de rede:**
- Cada novo usuário melhora a IA de todos — essa base instalada não pode ser forkada
- Custo de infra: **$0**. Roda no GitHub. Sem servidores. Sem GPUs. Sem cloud

```bash
# Opt-in — sua escolha, sempre
export ENTROLY_FEDERATION=1
```

### 🔒 Roda Localmente. Seu Código Nunca Sai da Sua Máquina.

Funciona em ambientes air-gapped e regulamentados — nenhum dado é enviado para fora.

---

## Benchmarks

### Retenção de Precisão

A compressão não prejudica a precisão — verificado com API ao vivo (gpt-4o-mini, Wilson 95% CI):

| Benchmark | n | Budget | Baseline (95% CI) | Com Entroly (95% CI) | Retenção | Economia de Tokens |
|---|---|---|---|---|---|---|
| NeedleInAHaystack | 20 | 2K | 100% [83.9–100%] | 100% [83.9–100%] | **100.0%** | **99.5%** |
| LongBench (HotpotQA) | 50 | 2K | 64.0% [50.1–75.9%] | 68.0% [54.2–79.2%] | **106.2%** | **85.3%** |
| Berkeley Function Calling | 50 | 500 | 100% [92.9–100%] | 100% [92.9–100%] | **100.0%** | **79.3%** |
| SQuAD 2.0 | 50 | 100 | 78.0% [64.8–87.2%] | 76.0% [62.6–85.7%] | **97.4%** | **39.3%** |
| GSM8K | 100 | 50K | 85.0% [76.7–90.7%] | 86.0% [77.9–91.5%] | **101.2%** | pass-through¹ |
| MMLU | 100 | 50K | 82.0% [73.3–88.3%] | 85.9% [77.8–91.4%] | **104.7%** | pass-through¹ |
| TruthfulQA (MC1) | 100 | 50K | 72.0% [62.5–79.9%] | 73.7% [64.3–81.4%] | **102.4%** | pass-through¹ |

> ¹ **pass-through**: Contexto já dentro do budget — Entroly corretamente não faz nada. Os intervalos de confiança se sobrepõem em todos os benchmarks.

### Comparação com Outros Métodos (Contexto Longo)

| Método | Retenção | Redução de Tokens | Arquitetura / Trade-offs |
|---|---|---|---|
| **Entroly** | **100–106%** | **85–99%** | **Rápido (~80ms).** Knapsack por fragmento. Fidelidade estrutural verbatim perfeita. |
| Poda neural por token | ~98–99% | 80–95% | **Alto overhead.** Requer transformer local. Degradação de sintaxe de código. |
| Compactação verbatim por regras | ~100% | 50–70% | **Alta fidelidade.** Mas menor redução de tokens. |
| Compressão por atenção | 95%+ | 26–54% | **Precisão sólida.** Mas menor redução de tokens. |

---

## Paridade Total: Python & Node.js

| Capacidade | Python | Node.js (WASM) |
|---|---|---|
| Compressão de contexto | ✅ | ✅ |
| Auto-evolução | ✅ | ✅ |
| Dreaming loop | ✅ | ✅ |
| Federação | ✅ | ✅ |
| Destilação de resposta | ✅ | ✅ |
| Gateways de chat | ✅ | ✅ |
| Export agentskills.io | ✅ | ✅ |

---

<p align="center">
  <b>Pare de pagar pelos tokens que sua IA desperdiça. Comece com uma IA que se ensina sozinha.</b><br/>
  <code>npm install entroly-wasm && npx entroly-wasm</code>&nbsp;&nbsp;|&nbsp;&nbsp;<code>pip install entroly && entroly go</code>
</p>
