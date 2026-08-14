# モデル管理および Hugging Face Hub チーム共有ガイド

本プロジェクトでは、リポジトリの軽量化とチーム開発の円滑化のため、**モデルの重みバイナリファイル（*.pth, *.safetensors, *.bin）を Git Tracking から除外**し、外部ストレージ（**Hugging Face Hub**）にてチーム共有・同期管理を行っています。

---

## 1. Hugging Face Organization (braille-mate) の作成と準備

個人アカウントではなく **Organization（組織: braille-mate）** を使用することで、チームメンバー全員が同一のモデルリポジトリに対して安全に同期・自動共有を行えるようになります。

### ステップ 1: Organization の作成
1. Hugging Face（ https://huggingface.co/ ）にログイン。
2. 右上のアイコンから **[New Organization]** を選択。
3. Organization Name に `braille-mate` を入力して作成。

### ステップ 2: チームモデルリポジトリの作成
1. 作成した Organization (`braille-mate`) のページから **[New Model]** を選択。
2. Model Name に `braille-mate-hiragana-recognizer` を入力して作成（Public または Private）。

### ステップ 3: メンバーの追加と Access Token の準備
1. Organization の **[Members]** タブからチームメンバーを招待。
2. 各自の個人アカウント設定（Settings -> Access Tokens）にて Token を発行。
   - **個人アカウントで作成した Access Token で問題ありません。**（Hugging Face では個人の Access Token を使用して所属 Organization へのアクセスを行います）
   - トークン作成時、**Write（書き込み）権限** を付与してください。

---

## 2. 開発者向けローカル環境設定 (.env)

リポジトリ ID (`braille-mate/braille-mate-hiragana-recognizer`) はコード内（`config.py`）で固定されているため、一般利用者は設定不要です。

モデルを学習・更新してアップロードする開発者のみ、プロジェクト直下に `.env` を作成し、Write 権限付きのアクセストークンを記載します。

```bash
cp .env.example .env
```

`.env` 内の設定内容:
```env
# Hugging Face Hub Write トークン (開発者のみ)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 3. 学習時の完全自動同期

学習スクリプト実行時、手動でのフラグ指定なしでチームモデルとの相互同期が自動適用されます。

```powershell
# CNN モデルの学習 (チーム最新モデルの自動取得 ＆ 新記録更新時の自動同期)
uv run python models/cnn/train.py

# Wav2Vec2 モデルの学習
uv run python models/wav2vec2/train.py
```

### 動作仕様
- **開始時 (Auto-Pull)**: Hugging Face リモートのチーム共有最新モデルと手元ファイルの SHA-256 ハッシュ値を照合。手元が古い場合や未存在の場合のみ自動取得（一致していれば通信 0 秒でスキップ）。
- **完了時 (Auto-Push)**: 学習結果の Validation Macro-F1 が過去最高精度（チーム最高）を更新した場合のみ、自動的に Hugging Face へプッシュ送信。チームの既存スコアを壊す心配がありません。

---

## 4. 手動でのモデル同期コマンド

CLI スクリプト [`script/upload_to_hf.py`](file:///c:/Users/yamadarikuto/Mycode/voicerecognizer/script/upload_to_hf.py) を使用して手動同期を行うことも可能です。

```powershell
# CNN のベストモデル成果物（best_model.pth, labels.json）を同期 (デフォルト)
uv run python script/upload_to_hf.py

# Wav2Vec2 のベストモデル成果物を同期
uv run python script/upload_to_hf.py --type wav2vec2

# ローカルハッシュチェックを無視して強制再送信
uv run python script/upload_to_hf.py --force
```
