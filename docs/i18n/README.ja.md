<p align="center">
  <img src="../../docs/assets/entroly_wordmark.svg" width="820" alt="Entroly">
</p>

<h1 align="center">Entroly — AI運用コストを削減するドロップイン型コンテキスト保証</h1>

<p align="center"><b>重要な証拠の管理を失うことなく、不必要なコンテキストを削減します。</b><br>
最も価値の高い証拠を優先的に選択し、圧縮し、元データを復元可能な状態で保持し、レシートを発行します — コードベースやエージェントアーキテクチャを書き換える必要はありません。</p>
<p align="center">
  <sub>Entrolyはローカルファーストの Context OS です：コンテンツアドレス型の証拠、復元可能な圧縮、監査可能なレシート。プロキシ、MCP、プラグイン、ラッパー、SDKパスを通じて Claude Code、Codex、OpenClaw、GitHub Copilot、Cursor、Aider、OpenAI/Anthropic 互換アプリと連携します。</sub>
</p>
<p align="center">
  <a href="https://pypi.org/project/entroly/"><img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI" alt="Entroly on PyPI"></a>
  <a href="https://www.npmjs.com/package/entroly"><img src="https://img.shields.io/npm/v/entroly?color=red&label=npm" alt="Entroly on npm"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="Apache-2.0 license"></a>
  <a href="https://github.com/juyterman1000/entroly"><img src="https://img.shields.io/github/stars/juyterman1000/entroly?style=social" alt="Entroly GitHub stars"></a>
</p>

<p align="center"><b>100,438 回以上のダウンロード · 日々成長中</b><br>
<sub>複数の配信元から集計。</sub></p>

<p align="center"><b>⭐ Entrolyが役に立ったら、GitHubでスターをお願いします。</b><br>
<a href="https://github.com/juyterman1000/entroly">⭐ GitHubでEntrolyにスターを</a> — プロジェクトの成長と、より多くの開発者への普及に役立ちます。</p>

<p align="center">
  <b><a href="../../README.md">English</a> · <a href="README.zh.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · 日本語 · <a href="README.ko.md">한국어</a> · <a href="README.es.md">Español</a> · <a href="README.hi.md">हिन्दी</a> · <a href="README.fr.md">Français</a> · <a href="README.de.md">Deutsch</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.it.md">Italiano</a> · <a href="README.tr.md">Türkçe</a> · <a href="README.vi.md">Tiếng Việt</a> · <a href="README.id.md">Bahasa Indonesia</a> · <a href="README.uk.md">Українська</a> · <a href="README.pl.md">Polski</a></b>
</p>

---

## Entrolyとは？（わかりやすく説明）

AIコーディングアシスタントにはメモリ制限があります。コードベース全体を渡すと、遅く、高価で、注意散漫になります — 500ページのマニュアルを渡すようなものですが、実際には47ページだけが必要だったのです。

**Entrolyは47ページを見つけます。**

コードとAIの間に位置し、すべてを読み取り、実際に聞かれている質問に関係のある部分だけを渡します。これを安全に行える3つの理由：
|  |  |
|---|---|
| 💰 **請求額が下がる** | AIに送信するテキストが少なければ、請求額も小さくなります。どのくらい節約できるかは作業内容によります — 下の[実際の数値](#ベンチマーク)をご覧ください。 |
| 🔍 **何も失われない** | Entrolyが脇に置いたものはすべて保持され、*正確に*元の状態に復元できます。一文字一文字そのままです。 |
| 🧾 **作業を確認できる** | すべての判断にレシートが付きます：何が保持され、何が除外され、その理由は何か。 |

**コードを変更する必要はありますか？** いいえ。Entrolyはすでに使っているツール — Claude Code、Cursor、Copilotなど30以上のツール — と連携し、バックグラウンドで動作します。

**試すのに費用はかかりますか？** いいえ。下の[インストール](#インストール)セクションの2つのコマンドは完全にローカルマシンで動作し、APIキーは不要で、有料サービスに接続する前にあなたのプロジェクトの実際の数値を表示します。

---
## インストール

> **どれを選ぶか迷ったら？** **Python** を選んでください。最も完全なバージョンで、ほとんどの人が使用しています。

| プラットフォーム | インストール | 得られる機能 |
|---|---|---|
| 🐍 **Python** (pip) — *推奨* | `pip install -U entroly` | すべて：コマンドラインツール、AIエディタとの通信サーバー、コードライブラリ |
| 📦 **Node / npm** | `npm install -g entroly` | 同じエンジン、Python不要 |
| 🦀 **Rust**（ソースビルド） | `cd entroly-core && cargo build --release --bin entroly-rs --features proxy` | 単一の自己完結型プログラム |
| 🍺 **Homebrew** | `brew install juyterman1000/entroly/entroly` | macOS/Linuxでのコマンドラインツール |
| 🐳 **Docker** | `docker pull ghcr.io/juyterman1000/entroly:latest` | コンテナで実行、マシンへのインストール不要 |

**インストールの確認 — 無料、オフライン、APIキー不要：**

```bash
cd /your/repo
entroly verify-claims
entroly simulate
```

---
## クイックスタート

> **動かすだけ？** `pip install -U entroly && entroly go` — これだけです。エディタを見つけ、自動設定し、前後比較のダッシュボードを表示します。

```python
from entroly import compress, compress_messages, optimize
compressed = compress(api_response, budget=2000)
messages   = compress_messages(messages, budget=30000)
context    = optimize(fragments, budget=8000, query="ログインバグを修正")
```

---
## ベンチマーク

| ベンチマーク | ベースライン | Entroly使用時 | 保持率 | トークン節約 |
|---|---|---|---|---|
| NeedleInAHaystack | 100% | 100% | **100%** | **99.5%** |
| LongBench (HotpotQA) | 64% | 66% | **103%** | **85.3%** |
| Berkeley Function Calling | 100% | 100% | **100%** | **79.3%** |
| SQuAD 2.0 | 80% | 72% | **90%** | **43.8%** |

**正直に言うと：** SQuAD 2.0の行を見てください — 精度は*下がりました*（80% → 72%）。圧縮はトレードオフであり、魔法ではありません。すべての場面で勝てるわけではありません。

---
## 機能

- **まず選択、次に圧縮** — どのファイルが実際に質問に答えるかを見極めてから、*それから*圧縮します。
- **元のデータを正確に復元** — 除外されたものは一文字一文字復元でき、フィンガープリントで検証されます。
- **作業を透明化** — すべての判断のレシート：何が保持され、何が除外され、その理由、残りのリスク。
- **回答をファクトチェック** — AIの回答を提供された証拠と比較。ローカルで実行、追加のAI呼び出し費用なし。

---
## よくある質問

<details>
<summary><b>コードやファイルが変更されますか？</b></summary>
<br>
いいえ。Entrolyはファイルを読み取り、AIに何を送信するかを決定します。プロジェクト内の何も編集、移動、削除しません。
</details>

<details>
<summary><b>コードはどこかにアップロードされますか？</b></summary>
<br>
いいえ。選択、圧縮、チェックはすべてあなた自身のマシンで行われます。Entroly自体は外部呼び出しを行いません。デフォルトで分析機能はオフです。
</details>

<details>
<summary><b>どのくらい節約できますか？</b></summary>
<br>
正直なところ、プロジェクトによります。`entroly simulate` を実行してください — 無料で、APIキーは不要で、あなたのファイルでの削減量を見積もれます。
</details>

---

<p align="center"><sub>Apache-2.0 · ローカルファースト · デフォルトで外部分析なし</sub></p>
<p align="center"><code>pip install entroly && entroly go</code></p>
