<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/logo.png" width="180" alt="Entroly">
</p>

<h1 align="center">Entroly Daemon</h1>

<h3 align="center">あなたのAIは盲目です。30秒でそれを修正し、AIが自律的に学習するのを見てください。</h3>

<p align="center">
  <i>Claude、Cursor、Copilot、Codex、そしてMiniMaxはあなたのコードベースのわずか5%しか見ていません。Entrolyは彼らに<b>200万トークンの頭脳を90%オフで</b>提供します—それは<b>コンテキストを圧縮し、トークンの節約と優れた回答のみに注力して自律的に自己進化し新しいスキルを生み出すデーモンです</b>。学習が数学的に「トークンマイナス（常に節約する）」である最初のAIランタイムです。</i>
</p>

<p align="center">
  <code>npm install entroly-wasm && npx entroly-wasm</code>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="../../README.md"><b>English (Original) →</b></a>
</p>

---

## 実際に得られるもの

| | Entrolyなし | **Entrolyあり** |
|---|---|---|
| AIが見るファイル数 | 5–10 | **リポジトリ全体** |
| 1回のリクエストのトークン | 約 186,000 | **9,300 – 55,000** |
| 1000リクエストあたりのコスト | 約 $560 | **$28 – $168** |
| 実質的なコンテキスト長 | 200K | **約200万 (可変解像度圧縮)** |
| 学習コスト (時間の経過) | 増加する (トークン) | **$0 — 完全にトークンマイナス** |
| セットアップ | プロンプトハックに数時間 | **30秒** |

重要なファイルは「完全な状態」で。サポートファイルは「シグネチャ（関数定義等）」のみ。残りは「リファレンス」として送信されます。あなたのAIは全体像を把握しますが、あなたはほとんどコストを支払いません。

---

## インストール

```bash
npm install entroly-wasm && npx entroly-wasm
# または
pip install entroly && entroly go
```

以上です。IDEを検出し、Claude/Cursor/Copilot/Codexに自動接続し、圧縮を開始します。

<p align="center">
  <b>AIが浪費するトークンへの支払いを止め、自分で学習するAIを走らせましょう。</b>
</p>
