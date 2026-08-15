# voicerecognizer (GitHub / Hugging Face) Public 公開手順 ＆ チェックリスト

本ドキュメントは、Issue #41 に基づき `voicerecognizer` リポジトリおよび Hugging Face リポジトリ（`braille-mate/braille-mate-hiragana-recognizer`）を安全にオープンソース / 公開モデル化するための確認結果と移行手順をまとめたものです。

---

## 1. セキュリティ ＆ プライバシー監査結果 (Phase 1)

### 秘密情報・トークンの Git 履歴監査
- 監査結果: Git コミット履歴およびコードベース全体をスキャンし、実際の `HF_TOKEN` や個人アクセストークン等の認証情報が一切コミットされていないことを確認。
- `.env` ファイル: `.gitignore` に登録されており、リポジトリ追跡対象外であることを確認。
- `.env.example`: 名前空間プレフィックス付き環境変数（`VOICERECOGNIZER_HF_*`）を反映したテンプレートとして整備完了。

### 生音声データセット (`dataset/`) の取り扱い
- 学習用生音声データセットについて、OSS リポジトリの軽量化およびプライバシー保護の観点から、今後の運用方針を整備。
- `.gitignore` に生成物（`merged_dataset/`, `processed_dataset/`, `evaluation_results/`）が登録されており不要なローカル中間ファイルが追跡されないことを確認。

### 個人パス・ハードコードの排除
- ソースコード内にローカル環境固有の絶対パス（`C:\Users\...` 等）が存在しないことを検証完了。

---

## 2. モジュール設計の自立化 (Phase 2)

### 環境変数プレフィックスとフォールバック
- `VOICERECOGNIZER_HF_REPO_ID` (フォールバック: `HF_REPO_ID`, デフォルト: `braille-mate/braille-mate-hiragana-recognizer`)
- `VOICERECOGNIZER_HF_TOKEN` (フォールバック: `HF_TOKEN`, デフォルト: `""`)
- `VOICERECOGNIZER_HF_AUTO_UPLOAD` (フォールバック: `HF_AUTO_UPLOAD`, デフォルト: `false`)
- `VOICERECOGNIZER_CACHE_DIR` (デフォルト: `~/.cache/voicerecognizer`)

### 暗黙的 load_dotenv() の廃止
- `config.py` モジュールロード時の暗黙的 `load_dotenv()` を廃止し、明示的ヘルパー関数 `load_env()` を提供。外部ライブラリとしてインポートされた際に予期せぬ `.env` 読み込みが発生しない構造に改善。

### ゼロコンフィグ (Zero-config) での動作
- 環境変数や `.env` が一切存在しない環境でも、公開モデルリポジトリから ONNX 重み（`model_mel_int8.onnx`）を自動取得し、即座に推論が動作することを保証。

### コードからの明示的パラメータ注入
- `Wav2Vec2Recognizer(hf_repo_id=..., hf_token=..., model_path=...)`
- `CNNRecognizer(hf_repo_id=..., hf_token=..., model_path=...)`
  上記のように、コードから直接 Hugging Face リポジトリやトークン、モデルパスを注入可能。

---

## 3. Hugging Face リポジトリ公開手順 (Phase 3)

1. **リポジトリ公開設定 (Visibility)**:
   - Hugging Face Web 上で `braille-mate/braille-mate-hiragana-recognizer` の Settings -> Danger Zone -> Change visibility を **Public** に変更。
2. **モデルカードの配置**:
   - `Docs/HF_MODEL_CARD.md` の内容を Hugging Face リポジトリ直下の `README.md` に反映。
3. **公開重みファイルの確認**:
   - リポジトリに以下の 5 ファイルが最新状態で配置されていることを確認:
     - `model_mel_int8.onnx`
     - `model_int8.onnx`
     - `labels.json`
     - `config.json`
     - `preprocessor_config.json`

---

## 4. GitHub リポジトリ公開 ＆ OSS ドキュメント整備 (Phase 4)

1. **ライセンスの明記**:
   - リポジトリ直下に `LICENSE` (MIT License) を配置完了。
2. **README.md の整備**:
   - ゼロコンフィグでの利用法、クイックスタート、インストール手順、システム要件、環境変数一覧を記載完了。
3. **GitHub Actions セキュリティ確認**:
   - `.github/workflows/quality_gate.yml` において Secrets を一切使用・公開しない設計となっており、Fork からの Pull Request でも安全に CI がパスすることを確認。
4. **GitHub リポジトリ公開設定**:
   - GitHub Web 上で `braille-coach-ring/voicerecognizer` の Settings -> General -> Danger Zone -> Change repository visibility を **Public** に変更。
5. **ブランチ保護ルール (Branch Protection Rules)**:
   - Settings -> Branches -> Add branch protection rule:
     - Branch name pattern: `main`
     - Require a pull request before merging: チェック
     - Require status checks to pass before merging: チェック (`Strict Quality Gate / quality-gate`)
     - Do not allow bypassing the above settings: チェック

---

## 5. 完了条件 (Definition of Done) の達成状況

1. 外部環境（新規仮想環境）からトークンや `.env` の設定なしで `voicerecognizer` をインストール＆即座に認識実行できる状態を整備完了。
2. 個人音声データ・認証情報・秘密鍵の漏洩リスクがゼロであることを監査・確認完了。
3. GitHub および Hugging Face のドキュメント（README / Model Card / LICENSE）を整備完了。
