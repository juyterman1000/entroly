<p align="center">
  <a href="../../README.md">🇬🇧 English</a> •
  <a href="README.zh-CN.md">🇨🇳 中文</a> •
  <a href="README.ja.md">🇯🇵 日本語</a> •
  <a href="README.ko.md">🇰🇷 한국어</a> •
  <a href="README.pt-BR.md">🇧🇷 Português</a> •
  <a href="README.es.md">🇪🇸 Español</a> •
  <a href="README.de.md">🇩🇪 Deutsch</a> •
  <a href="README.ru.md">🇷🇺 Русский</a> •
  <a href="README.hi.md">🇮🇳 हिन्दी</a> •
  <a href="README.tr.md">🇹🇷 Türkçe</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/logo.png" width="180" alt="Entroly">
</p>

<h1 align="center">Entroly — Réduisez les Coûts de Tokens IA de 70–95%</h1>

<h3 align="center">Vos outils de codage IA ne voient que 5% de votre code.<br/>Entroly leur donne la vue complète — pour une fraction du coût.</h3>

<p align="center">
  <a href="../../README.md#install"><b>Installation</b></a> ·
  <a href="../../cookbook/README.md"><b>Cookbook</b></a> ·
  <a href="../../README.md#benchmarks"><b>Benchmarks</b></a> ·
  <a href="../../README.md#works-with-your-stack"><b>65+ agents supportés</b></a> ·
  <a href="https://juyterman1000.github.io/entroly/docs/dashboard.html"><b>Dashboard</b></a>
</p>

<p align="center">
  <code>npm install entroly-wasm && npx entroly-wasm</code>&nbsp;&nbsp;|&nbsp;&nbsp;<code>pip install entroly && entroly go</code>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="https://juyterman1000.github.io/entroly/"><b>Démo en direct →</b></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/self_improvement.svg" alt="Entroly auto-amélioration" width="800">
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/token_savings.svg" alt="Entroly profit" width="800">
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/context_quality.svg" alt="Entroly qualité du contexte" width="800">
</p>

---

## Le Problème — et l'Impact sur vos Résultats

Chaque outil de codage IA — Claude, Cursor, Copilot, Codex — a le même angle mort : **il ne voit que 5–10 fichiers à la fois.** Les 95% restants de votre code sont invisibles.

Les modèles continuent de grossir — **Claude Opus 4.7** vient de sortir avec encore plus de capacités et des coûts par token encore plus élevés. Des fenêtres de contexte plus grandes ne résolvent pas le problème ; elles l'aggravent. Vous payez pour 186 000 tokens par requête — dont la majeure partie n'est que du code boilerplate dupliqué.

> **Entroly résout les deux problèmes en 30 secondes.**

---

## Ce Qui Change au Jour 1

| Métrique | Avant Entroly | **Après Entroly** |
|---|---|---|
| Fichiers visibles par l'IA | 5–10 | **Tout le dépôt** |
| Tokens par requête | ~186 000 | **9 300 – 55 000** |
| Dépense mensuelle IA (1K req/jour) | ~16 800$ | **840$ – 5 040$** |
| Précision des réponses IA | Incomplète, hallucinations | **Consciente des dépendances, correcte** |
| Temps dev à corriger les erreurs IA | Heures/semaine | **Quasi zéro** |
| Configuration | Jours d'ingénierie de prompts | **30 secondes** |

> **ROI :** Une équipe de 10 dépensant 15K$/mois en API IA économise **10K$–14K$/mois** dès le jour 1.

---

## Ce Que Vos Concurrents Savent Déjà

Les équipes qui adoptent Entroly aujourd'hui ne font pas que économiser — elles **accumulent un avantage composé** que votre équipe ne peut pas rattraper.

- **Semaine 1 :** Leur IA voit 100% du code. La vôtre en voit 5%. Ils livrent plus vite.
- **Mois 1 :** Leur runtime a appris les patterns du code. Le vôtre hallucine encore les imports.
- **Mois 3 :** Leur installation est branchée à la fédération — absorbant les stratégies d'optimisation de milliers d'équipes dans le monde. Vous ne savez même pas que ça existe.
- **Mois 6 :** Ils ont économisé 80K$+ en coûts API. Ce budget est allé dans les recrutements. Vous expliquez encore aux finances pourquoi la facture IA ne cesse d'augmenter.

Chaque jour d'attente creuse l'écart. L'effet fédération signifie que **les early adopters deviennent plus forts plus vite** — et cet avantage se compose.

---

## 🌐 Apprentissage Fédéré en Essaim — La Partie Qui Ressemble à de la Science-Fiction

Prenez la Boucle de Rêve et multipliez-la par **chaque développeur sur Terre qui utilise Entroly.**

Pendant que vous dormez, votre daemon rêve — et 10 000 autres aussi. Chacun découvre des astuces légèrement différentes de compression de code. Chacun partage ce qu'il a appris — anonymement, en privé. Chacun absorbe ce que les autres ont trouvé.

**Vous vous réveillez. Votre IA est plus intelligente que quand vous l'avez laissée. Pas grâce à vous — grâce à ce que l'essaim a rêvé.**

```
Votre daemon rêve → découvre une meilleure stratégie → la partage (anonymement)
     ↓
10 000 autres daemons ont fait la même chose cette nuit
     ↓
Vous ouvrez votre laptop → votre IA a déjà tout absorbé
```


**Effet de réseau :**
- Chaque nouvel utilisateur améliore l'IA de tous — cette base installée ne peut pas être forkée
- Coût d'infrastructure : **0$**. Tourne sur GitHub. Pas de serveurs. Pas de GPUs. Pas de cloud

```bash
# Opt-in — votre choix, toujours
export ENTROLY_FEDERATION=1
```

### 🔒 S'Exécute Localement. Votre Code Ne Quitte Jamais Votre Machine.

Fonctionne dans des environnements air-gapped et réglementés — rien ne communique vers l'extérieur.

---

## Benchmarks

### Rétention de Précision

La compression n'affecte pas la précision — vérifié avec l'API en direct (gpt-4o-mini, Wilson 95% CI) :

| Benchmark | n | Budget | Baseline (95% CI) | Avec Entroly (95% CI) | Rétention | Économie de Tokens |
|---|---|---|---|---|---|---|
| NeedleInAHaystack | 20 | 2K | 100% [83.9–100%] | 100% [83.9–100%] | **100.0%** | **99.5%** |
| LongBench (HotpotQA) | 50 | 2K | 64.0% [50.1–75.9%] | 68.0% [54.2–79.2%] | **106.2%** | **85.3%** |
| Berkeley Function Calling | 50 | 500 | 100% [92.9–100%] | 100% [92.9–100%] | **100.0%** | **79.3%** |
| SQuAD 2.0 | 50 | 100 | 78.0% [64.8–87.2%] | 76.0% [62.6–85.7%] | **97.4%** | **39.3%** |
| GSM8K | 100 | 50K | 85.0% [76.7–90.7%] | 86.0% [77.9–91.5%] | **101.2%** | pass-through¹ |
| MMLU | 100 | 50K | 82.0% [73.3–88.3%] | 85.9% [77.8–91.4%] | **104.7%** | pass-through¹ |
| TruthfulQA (MC1) | 100 | 50K | 72.0% [62.5–79.9%] | 73.7% [64.3–81.4%] | **102.4%** | pass-through¹ |

> ¹ **pass-through** : Le contexte tient déjà dans le budget — Entroly ne fait correctement rien. Les intervalles de confiance se chevauchent sur tous les benchmarks.

### Comparaison avec d'autres méthodes (contexte long)

| Méthode | Rétention | Réduction de Tokens | Architecture / Compromis |
|---|---|---|---|
| **Entroly** | **100–106%** | **85–99%** | **Rapide (~80ms).** Knapsack par fragment. Fidélité structurelle verbatim parfaite. |
| Élagage neural par token | ~98–99% | 80–95% | **Overhead élevé.** Nécessite un transformer local. Dégrade la syntaxe du code. |
| Compactage verbatim par règles | ~100% | 50–70% | **Haute fidélité.** Mais réduction de tokens plus faible. |
| Compression par attention | 95%+ | 26–54% | **Précision solide.** Mais réduction de tokens plus faible. |

---

## Parité Totale : Python & Node.js

| Capacité | Python | Node.js (WASM) |
|---|---|---|
| Compression de contexte | ✅ | ✅ |
| Auto-évolution | ✅ | ✅ |
| Boucle de rêve | ✅ | ✅ |
| Fédération | ✅ | ✅ |
| Distillation de réponse | ✅ | ✅ |
| Passerelles de chat | ✅ | ✅ |
| Export agentskills.io | ✅ | ✅ |

---

<p align="center">
  <b>Arrêtez de payer pour les tokens que votre IA gaspille. Lancez une IA qui s'enseigne elle-même.</b><br/>
  <code>npm install entroly-wasm && npx entroly-wasm</code>&nbsp;&nbsp;|&nbsp;&nbsp;<code>pip install entroly && entroly go</code>
</p>
