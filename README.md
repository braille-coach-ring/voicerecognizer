# voicerecognizer

日本語ひらがな105クラス対応の超軽量・低遅延音声認識ライブラリ。  
Wav2Vec2 + ONNX INT8 量子化モデルおよび軽量 CNN モデルを内包し、CPU 環境（Raspberry Pi / ノートPC）でもリアルタイムに動作します。

公開 Hugging Face Hub リポジトリ（`braille-mate/braille-mate-hiragana-recognizer`）と統合されており、環境変数の設定なし（ゼロコンフィグ）で即座にモデルを自動ダウンロードして推論を実行できます。

---

## 主な特徴

- ゼロコンフィグ（設定不要）: トークンや設定ファイル不要で、公開モデルを自動取得して即座に推論開始。
- 超高速推論 (ONNX INT8): ONNX Runtime + INT8 量子化により、CPU 環境でも数ミリ秒オーダーの超低遅延認識。
- 105文字ひらがな全音対応: 清音・濁音・半濁音・拗音を含む全105ラベルの母音・発音認識。
- 非同期リアルタイムストリーミング: `AudioStreamListener` によるマイク入力の常時非同期監視と文字認識。
- コードからの明示的設定注入: Python コードから直接カスタムリポジトリやローカル重みパス、認証トークンを注入可能。

---

## インストール

### pip でのインストール

```bash
pip install git+https://github.com/braille-coach-ring/voicerecognizer.git
```

### uv でのインストール

```bash
uv add git+https://github.com/braille-coach-ring/voicerecognizer.git
```

### システム依存関係

マイク入力ストリーミング機能を利用する場合は、OS ごとに以下のオーディオライブラリが必要です。

- Ubuntu / Debian: `sudo apt-get install -y libportaudio2 libsndfile1`
- macOS: `brew install portaudio libsndfile`
- Windows: 通常 `sounddevice` のホイールパッケージに PortAudio が同梱されています。

---

## クイックスタート

### 1. Wav2Vec2 ONNX による音声認識（デフォルト・推奨）

```python
import voicerecognizer as vr

# 初回実行時に Hugging Face Hub より自動ダウンロードされます
recognizer = vr.Wav2Vec2Recognizer()

# 音声ファイル（wav）または NumPy 波形配列を渡して認識
text = recognizer.recognize("sample.wav")
print(f"認識結果: {text}")

# 候補と確信度スコアの取得
candidates = recognizer.recognize_with_candidates("sample.wav", top_k=3)
for label, score in candidates:
    print(f"候補: {label} (確信度: {score:.4f})")
```

### 2. 非同期マイクストリーミング（リアルタイム認識）

```python
import asyncio
import voicerecognizer as vr


async def main():
    recognizer = vr.Wav2Vec2Recognizer()
    listener = vr.AudioStreamListener(recognizer=recognizer)

    print("マイク音声の監視を開始します (Ctrl+C で停止)...")
    async for result in listener.listen():
        print(f"認識文字: {result.text} (確信度: {result.confidence:.2f})")


if __name__ == "__main__":
    asyncio.run(main())
```

### 3. CNN 認識器の利用

```python
import voicerecognizer as vr

cnn_recognizer = vr.CNNRecognizer()
text = cnn_recognizer.recognize("sample.wav")
print(f"CNN 認識結果: {text}")
```

### 4. 明示的なパラメータ注入（カスタムリポジトリ / ローカルモデル）

```python
import voicerecognizer as vr

# 独自のリポジトリやローカルパスを指定
custom_recognizer = vr.Wav2Vec2Recognizer(
    model_path="./my_local_model_dir",
    hf_repo_id="my-org/my-voice-model",
    hf_token="optional_token",
)
```

---

## 環境変数設定

公開モデルの利用においては設定不要ですが、プライベートリポジトリの利用やモデルの自動アップロードを行う場合は以下の環境変数を設定できます。

| 環境変数名 | フォールバック | デフォルト値 | 説明 |
| --- | --- | --- | --- |
| `VOICERECOGNIZER_HF_REPO_ID` | `HF_REPO_ID` | `braille-mate/braille-mate-hiragana-recognizer` | Hugging Face Hub のリポジトリ ID |
| `VOICERECOGNIZER_HF_TOKEN` | `HF_TOKEN` | `""` (空) | Hugging Face 認証トークン (モデル公開時は不要) |
| `VOICERECOGNIZER_HF_AUTO_UPLOAD` | `HF_AUTO_UPLOAD` | `false` | 学習時に最良モデルを自動アップロードするかどうか |
| `VOICERECOGNIZER_CACHE_DIR` | なし | `~/.cache/voicerecognizer` | モデル重みファイルのキャッシュ保存先 |

設定ファイル（`.env`）を使用する場合は、リポジトリ直下の `.env.example` を `.env` にコピーして設定してください。

---

## 開発・学習手順

### リポジトリのクローンと環境構築

```bash
git clone https://github.com/braille-coach-ring/voicerecognizer.git
cd voicerecognizer
uv sync --all-extras --dev
```

### モデルの学習

```bash
# CNN モデルの学習 (自動前処理込み)
uv run python train.py --model cnn

# Wav2Vec2 モデルのファインチューニング
uv run python train.py --model wav2vec2
```

### テストおよび品質ゲートの実行

```bash
# Pytest 全テスト実行
uv run pytest

# 静的解析・フォーマット・型チェック
uv run python script/check_quality_gate.py
```

---

## ライセンス

本プロジェクトは [MIT License](LICENSE) の下で公開されています。
