# voicelibrary 現在の構成と実行方法

## このプログラムについて

`voicelibrary` は、音声入力を受け取り、選択した認識アルゴリズムで推論し、認識した文字列を返すための音声認識ライブラリです。

現在は Strategy Pattern を中心にして、次の認識方式を切り替えられる構造になっています。

- CNN: ゼロから学習したひらがな母音認識モデル。現在動作する実装です。
- Wav2Vec2 Fine-tuning: 今後実装予定の空実装です。
- Whisper: 今後実装予定の空実装です。

現在のCNNモデルは `a`, `e`, `i`, `o`, `u` の5クラスを認識します。重みファイルは `weights/best_model.pth` を使用します。

## 依存関係の流れ

実行時の依存関係は次の形です。

```text
main.py
↓
AudioPipeline
↓
VoiceRecognizer
↓
RecognitionStrategy
↓
CNNRecognizer / Wav2Vec2Recognizer / WhisperRecognizer
```

`AudioPipeline` は具体的な認識モデルを知りません。`VoiceRecognizer.recognize(audio)` だけを呼び出します。

Recognizerの切り替えは `RecognizerFactory` だけが担当します。`if` 文による切り替えも `core/factory/recognizer_factory.py` のみにあります。

## 主要クラス

| クラス | ファイル | 役割 |
| --- | --- | --- |
| `RecognitionStrategy` | `core/interfaces.py` | すべての認識方式が実装する共通インターフェース |
| `VoiceRecognizer` | `core/services/voice_recognizer.py` | Strategyを受け取り、`recognize(audio)` を呼ぶ薄いサービス |
| `RecognizerFactory` | `core/factory/recognizer_factory.py` | `cnn`, `wav2vec2`, `whisper` からRecognizerを生成 |
| `AudioPipeline` | `core/services/audio_pipeline.py` | 録音、認識、結果表示の流れを制御 |
| `CNNRecognizer` | `recognizers/cnn_recognizer.py` | CNN用のモデルロード、前処理、Mel変換、推論、後処理 |
| `Wav2Vec2Recognizer` | `recognizers/wav2vec2_recognizer.py` | Wav2Vec2用の将来実装枠 |
| `WhisperRecognizer` | `recognizers/whisper_recognizer.py` | Whisper用の将来実装枠 |
| `AudioPreprocessor` | `preprocessing/audio_preprocessor.py` | 録音データの共通波形処理 |

## ディレクトリ構成

```text
voicelibrary/
  main.py
  config.py

  core/
    interfaces.py
    factory/
      recognizer_factory.py
    services/
      audio_pipeline.py
      voice_recognizer.py

  recognizers/
    cnn_recognizer.py
    wav2vec2_recognizer.py
    whisper_recognizer.py

  models/
    cnn/
      hiragana_cnn.py
      train.py
      evaluate.py
      export.py
    wav2vec2/
      processor.py
      train.py
      evaluate.py
      export_onnx.py
      quantize.py
      vocab_builder.py

  preprocessing/
    audio_preprocessor.py
    collect.py
    dataset_builder.py
    text_normalizer.py

  runtime/
    audio_capture.py
    vad.py
    inference_worker.py
    output_worker.py
    queues.py

  dataset/
    hiragana_dataset.py
    json_dataset.py

  utils/
    audio.py
    logger.py
    metrics.py

  weights/
    best_model.pth
    last_model.pth

  tests/
    test_architecture.py
```

## セットアップ

依存関係は `uv` で管理しています。

```powershell
uv sync
```

`python` がPATHに入っていない環境では、直接仮想環境のPythonを使えます。

```powershell
.\.venv\Scripts\python.exe --version
```

## 音声ファイルを認識する

CNNでwavファイルを認識します。

```powershell
uv run python main.py dataset\mikeryu\a\001.wav --model cnn
```

PATHの都合で `uv run python` が使えない場合は次の形でも実行できます。

```powershell
.\.venv\Scripts\python.exe main.py dataset\mikeryu\a\001.wav --model cnn
```

出力例:

```text
a
```

## マイクから録音して認識する

音声ファイルを省略すると、マイクから1回録音して認識します。

```powershell
uv run python main.py --model cnn
```

現在のデフォルト設定は次の通りです。

- サンプリングレート: `16000`
- 録音長: `1.0` 秒
- VADしきい値: `0.02`
- VAD最小アクティブ比率（閾値超えサンプル比）: `0.02`
- VAD起動時スキップチャンク数: `3`（約0.3秒）
- モデル重み: `weights/best_model.pth`

## 認識方式を切り替える

指定できる値は次の3つです。

```powershell
uv run python main.py --model cnn
uv run python main.py --model wav2vec2
uv run python main.py --model whisper
```

ただし、現在実際に認識まで動くのは `cnn` のみです。`wav2vec2` と `whisper` は将来実装用の空実装なので、認識実行時には `NotImplementedError` になります。

## CNNモデルを学習する

学習済みデータは `processed_dataset/` を使います。

```powershell
uv run python -m models.cnn.train
```

主なオプション:

```powershell
uv run python -m models.cnn.train `
  --root-dir processed_dataset `
  --epochs 150 `
  --batch-size 8 `
  --learning-rate 0.001 `
  --best-model-path weights\best_model.pth `
  --last-model-path weights\last_model.pth
```

学習後は次のファイルが更新されます。

- `weights/best_model.pth`
- `weights/last_model.pth`
- `loss.png`
- `accuracy.png`

## CNNモデルを評価する

```powershell
uv run python -m models.cnn.evaluate
```

モデルパスや評価データを指定する場合:

```powershell
uv run python -m models.cnn.evaluate `
  --root-dir processed_dataset `
  --model-path weights\best_model.pth
```

## 音声データを収集する

マイクから短い音声サンプルを収集します。

```powershell
uv run python -m preprocessing.collect speaker_id
```

例:

```powershell
uv run python -m preprocessing.collect rinry --repeat 10
```

保存先は `dataset/<speaker_id>/<label>/` です。

## テストとチェック

設計テスト:

```powershell
uv run python -m unittest discover -s tests
```

構文チェック:

```powershell
uv run python -m compileall -q main.py config.py core recognizers models preprocessing runtime dataset utils tests
```

Ruff:

```powershell
uv run python -m ruff check main.py config.py core recognizers models preprocessing runtime dataset utils tests
```

## 今後実装する箇所

Wav2Vec2では、次の処理を `Wav2Vec2Recognizer` 内に実装する予定です。

- Processor
- ONNX Runtime
- CTC Decode

Whisperでは、次の処理を `WhisperRecognizer` 内に実装する予定です。

- log-mel変換
- decode
- 必要に応じたモデルロード
## 学習データを追加して再学習する

新しい音声データを追加したい場合は、以下の流れで作業します。

### 1. 音声を収集する

既存データを消さずに、新しい音声だけを追加します。

```powershell
uv run python -m preprocessing.collect rinry --repeat 20
```

保存先は次のようになります。

```text
dataset/
└── rinry/
    ├── a/
    │   ├── 001.wav
    │   ├── 002.wav
    │   ├── ...
    │   ├── 021.wav
    │   └── 022.wav
    ├── i/
    ├── u/
    ├── e/
    └── o/
```

同じ `speaker_id` を指定すると、既存の音声は保持したまま続き番号で保存されます。

別の話者を追加する場合は、新しい `speaker_id` を指定します。

```powershell
uv run python -m preprocessing.collect yamada --repeat 20
```

すると

```text
dataset/
├── rinry/
└── yamada/
```

のように話者ごとに管理されます。

---

### 2. 学習用データセットを更新する

収集した音声を学習用データへ変換します。

```powershell
uv run python -m preprocessing.dataset_builder
```

必要に応じて入力・出力ディレクトリも指定できます。

```powershell
uv run python -m preprocessing.dataset_builder `
    --input dataset `
    --output processed_dataset
```

この処理では

- リサンプリング
- 正規化
- 無音区間除去
- メルスペクトログラム生成（必要な場合）

など、学習に必要な前処理を行い `processed_dataset/` を更新します。

---

### 3. モデルを再学習する

前処理が終わったら学習を実行します。

```powershell
uv run python -m models.cnn.train
```

学習が終了すると

```text
weights/
    best_model.pth
    last_model.pth
```

が更新されます。

---

### データ追加時の流れ

```text
collect.py
        │
        ▼
dataset/
        │
        ▼
dataset_builder.py
        │
        ▼
processed_dataset/
        │
        ▼
models.cnn.train
        │
        ▼
weights/
        │
        ▼
main.py
```

この流れを繰り返すことで、学習データを増やしながらモデルの認識精度を向上させることができます。

# voicelibrary
パッケージ管理: uv

uv syncで必要なライブラリがインストールされる
uv run python "スクリプト名" で実行