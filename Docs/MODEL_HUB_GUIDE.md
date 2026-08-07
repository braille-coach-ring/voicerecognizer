# 📦 モデル管理および Hugging Face Hub 同期ガイド

本プロジェクトでは、リポジトリの軽量化とバージョン管理の最適化のため、**モデルの重みバイナリファイル（`*.pth`, `*.safetensors`, `*.bin`）を Git Tracking から除外**し、外部ストレージ（**Hugging Face Hub**）にて同期・管理しています。

---

## 🔑 1. 事前準備 (環境設定)

プロジェクト直下に `.env` ファイルを作成し、ご自身の Hugging Face アカウント（またはチーム Org）の情報を記述します。

```bash
cp .env.example .env
```

`.env` 内の設定内容:
```env
# Hugging Face Hub 設定
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx      # Write 権限を持つアクセス Token
HF_REPO_ID=rikutoyamada01/braille-mate-hiragana-recognizer  # 対象リポジトリID
HF_AUTO_UPLOAD=false                          # 学習終了時に常に自動アップロードを有効化する場合は true
```

> ⚠️ **チーム運用時の注意点**
> メンバー各自がモデルをアップロードできるようにするには、Hugging Face 上で **Organization (組織)** を作成してリポジトリを移動するか、リポジトリの `Settings` -> `Collaborators` からチームメンバーを追加してください。

---

## 🏋️ 2. 学習時の自動同期

学習スクリプト実行時、`--upload-hf` オプションを付与すると学習終了時に Hugging Face へ成果物を同期できます。

```powershell
# CNN モデルの学習と自動同期
uv run python models/cnn/train.py --upload-hf

# Wav2Vec2 モデルの学習と自動同期
uv run python models/wav2vec2/train.py --upload-hf
```

### ⚡ 無駄のないアップロード制御 (Smart Upload)
* **ベストモデル更新時のみ実行**: 学習の結果、最高評価精度（Validation Macro-F1）が更新されたセッションでのみアップロードが起動します。精度向上がなかったセッションでは自動的にスキップされます。
* **軽量成果物のみ送信**: CNN 学習時は `best_model.pth` および `labels.json`（約1.4MB）のみを送信し、関係のない大容量モデル（Wav2Vec2等）や途中経過（`last_model.pth`）は送信されません。

---

## 🛠️ 3. 手動でのモデル同期コマンド

CLI スクリプト [`script/upload_to_hf.py`](file:///c:/Users/yamadarikuto/Mycode/voicerecognizer/script/upload_to_hf.py) を使って、いつでも手動でモデル成果物を同期できます。

```powershell
# CNN のベストモデル成果物（best_model.pth, labels.json）を同期 (デフォルト)
uv run python script/upload_to_hf.py

# Wav2Vec2 のベストモデル成果物を同期
uv run python script/upload_to_hf.py --type wav2vec2

# 全モデルの Best 成果物を同期
uv run python script/upload_to_hf.py --type best_only

# ローカルハッシュチェックを無視して強制再送信
uv run python script/upload_to_hf.py --force
```

### 🔍 SHA-256 事前判定
手動スクリプト実行時は、ローカルファイルの SHA-256 ハッシュ値をリモート側と事前照合します。すでにリモートと完全同一のファイルであれば **0秒で通信を自動スキップ** します。
