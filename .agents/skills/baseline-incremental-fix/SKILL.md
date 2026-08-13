---
name: baseline-incremental-fix
description: >-
  Guide and workflow for systematically and cautiously resolving baseline lint
  (Ruff) and type (Mypy) errors file-by-file without causing code regressions.
  Inspired by strict quality gate automation principles.
---

# Baseline Incremental Fix Workflow

This skill provides a systematic, step-by-step workflow for resolving remaining lint and type baseline errors in the codebase without breaking existing functionality or introducing regressions ("デグレ防止").

---

## 核心思想 (Core Philosophy)

> **「壊れたら赤くなって止まる」を積み上げる**  
> 手動レビューや人間の記憶に頼らず、自動テストと品質ゲート (`uv run ci`) をセーフティネットとして一歩ずつ慎重にエラーを解消します。

---

## ステップバイステップ手順 (Step-by-Step Procedure)

### Step 1: 現在の未解消エラー数の把握
まず、プロジェクト全体の未解消エラー数とターゲットファイルを特定します。

```bash
uv run stats
```

- 残りエラーが多い上位ファイルから、**1回につき1ファイル** をターゲットに選定します。

---

### Step 2: ターゲットファイルのエラー詳細の特定
選定したファイル内の行番号とエラーコードを確認します。

```bash
uv run errors --file <対象ファイル名>
```

例:
```bash
uv run errors --file runtime/vad.py
```

---

### Step 3: 慎重な修正と分類基準

エラーを以下の基準で分類し、修正します。

1. **機械的・構文的修正 (即時修正可能)**
   - 未使用インポート・未定義変数 prefix (`_var`)
   - テスト関数の返り値アノテーション (`def test_xxx(self) -> None:`)
   - 冗長な構造 (例: `list(singletons)` $\rightarrow$ `singletons` inside `sorted()`)

2. ** Any 型・構造体設計を要する修正 (ユーザー要相談)**
   - `Any` の使用や、関数の引数・戻り値の型定義が不透明な場合
   - `dict` や `Queue` の中身のデータ構造が複数の型を取り得る場合
   - **方針**: 勝手に推測で `Any` や複雑な型を入れず、ユーザーに候補（`TypedDict`, `dict[str, Any]`, `Queue[T]`）を提示して合意形成します。

---

### Step 4: 自動検証 (デグレチェック)
コードを修正したら、**必ず即座に CI 品質ゲートを実行** します。

```bash
uv run ci
```

- **全単体テスト 100% 通過**
- **新規エラー数 0 件**

上記の両方が確認できるまで、次の修正に進まないでください。

---

### Step 5: ベースラインの凍結更新
問題なく修正とテスト通過が完了したら、ベースラインを更新して成果を保存します。

```bash
uv run sync-baseline
```

---

## 遵守ルール (Strict Guidelines)

1. **一度に大量のファイルを一括修正しない**: 1ファイル・1エラー種別ずつ着実に進めます。
2. **`safe_open` などの特殊オブジェクトに注意**: 辞書風オブジェクトに対するリファクタリング（例: `.keys()` の除去など）は動作確認テストを伴うこと。
3. **`uv run ci` なしの宣言禁止**: 修正を行った後に `uv run ci` の検証なしで完了を宣言することは厳禁です。
