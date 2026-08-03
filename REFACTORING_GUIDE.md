# predict_from_mike.py リファクタリング

## 📋 概要

元の `predict_from_mike.py` をSOLID原則とDesign Patternに基づいてリファクタリングしました。

- **機能**: 変更なし（同じ推論結果が得られます）
- **構造**: 大幅に改善（保守性・拡張性・テスト容易性が向上）

---

## 🏗️ 新しいアーキテクチャ

### ディレクトリ構造

```
voicelibrary/
├── core/                      # リファクタリング後のコアモジュール
│   ├── __init__.py
│   ├── interfaces.py          # 抽象インターフェース定義
│   ├── implementations.py     # デフォルト実装
│   ├── pipeline.py            # メインパイプライン
│   └── factory.py             # Dependency Injectionコンテナ
├── predict_from_mike_refactored.py  # 新しいエントリーポイント
├── predict_from_mike.py       # 元のコード（参照用）
└── REFACTORING_EXAMPLES.py    # カスタマイズ例
```

---

## 🎯 SOLID原則への対応

### 1. Single Responsibility Principle（単一責任の原則）

**Before（元のコード）**:
```python
# predict_from_mike.py が全てを処理
- マイク録音
- 音声前処理
- モデルロード
- 推論
- 結果表示
```

**After（リファクタリング後）**:
```
AudioRecorder          → 音声録音のみ
AudioPreprocessor      → 前処理のみ
MelSpectrogramConverter → メル変換のみ
ModelLoader            → モデルロードのみ
Inferencer             → 推論のみ
ResultPresenter        → 結果表示のみ
```

### 2. Open/Closed Principle（開放閉鎖原則）

**拡張に対して開く**:
```python
# 新しい前処理アルゴリズムを追加する場合
class AdvancedAudioPreprocessor(AudioPreprocessor):
    def preprocess(self, audio, sample_rate, target_length, top_db):
        # カスタム実装
        ...
```

**変更に対して閉じている**:
```python
# 既存コードへの影響なし
# インターフェースを実装するだけ
```

### 3. Liskov Substitution Principle（リスコフの置換原則）

```python
# どのResultPresenter実装でも互換性を持つ
config = DefaultConfig()
# result_presenter = DefaultResultPresenter()
result_presenter = FileResultPresenter()      # 代替可能
# result_presenter = JsonResultPresenter()    # 代替可能

pipeline = PipelineFactory.create_custom_pipeline(
    ...
    result_presenter=result_presenter,  # 代替可能
    ...
)
```

### 4. Interface Segregation Principle（インターフェース分離の原則）

```python
# 各インターフェースは最小限のメソッドのみを持つ
class AudioRecorder(ABC):
    @abstractmethod
    def record(self, duration: float, sample_rate: int) -> np.ndarray:
        pass

# 不要なメソッドを持つ大きなインターフェースはない
```

### 5. Dependency Inversion Principle（依存性逆転の原則）

**Before**:
```python
# predict_from_mike.py が具体的な実装に直接依存
model = HiraganaCNN(...)
audio = sd.rec(...)
output = model(mel_spec)
```

**After**:
```python
# 抽象インターフェースに依存
recorder: AudioRecorder = DefaultAudioRecorder()
model_loader: ModelLoader = DefaultModelLoader()
inferencer: Inferencer = DefaultInferencer(model_loader)
```

---

## 🎨 Design Patterns の適用

### 1. Strategy Pattern

各処理のアルゴリズムを動的に切り替え可能：

```python
# 前処理の戦略を切り替え
preprocessor: AudioPreprocessor = DefaultAudioPreprocessor()
# または
preprocessor: AudioPreprocessor = AdvancedAudioPreprocessor()

# 結果表示の戦略を切り替え
result_presenter: ResultPresenter = DefaultResultPresenter()
# または
result_presenter: ResultPresenter = FileResultPresenter()
# または
result_presenter: ResultPresenter = JsonResultPresenter()
```

### 2. Dependency Injection Pattern

全ての依存関係をコンストラクタで注入：

```python
pipeline = MicrophonePredictionPipeline(
    config=config,
    recorder=recorder,
    preprocessor=preprocessor,
    mel_converter=mel_converter,
    model_loader=model_loader,
    inferencer=inferencer,
    audio_saver=audio_saver,
    result_presenter=result_presenter,
    countdown_display=countdown_display,
)
```

### 3. Factory Pattern

複雑なオブジェクト生成を集中管理：

```python
# デフォルト実装で自動組み立て
pipeline = PipelineFactory.create_default_pipeline()

# カスタム実装で手動組み立て
pipeline = PipelineFactory.create_custom_pipeline(
    config=custom_config,
    recorder=custom_recorder,
    ...
)
```

### 4. Template Method Pattern

処理の流れを定義し、詳細は各クラスで実装：

```python
# core/pipeline.py の run() メソッド
def run(self) -> str:
    self._show_countdown()
    raw_audio = self._record_audio()
    processed_audio = self._preprocess_audio(raw_audio)
    self._save_audio(processed_audio)
    mel_spec = self._convert_to_mel_spectrogram(processed_audio)
    pred_idx, probs = self._run_inference(mel_spec)
    self._present_results(...)
    return predicted_label
```

---

## 🚀 使用方法

### シンプルな使用方法

```python
from core import PipelineFactory

# 1. パイプラインを作成（全て自動で組み立てられる）
pipeline = PipelineFactory.create_default_pipeline()

# 2. 初期化
pipeline.setup()

# 3. 実行
result = pipeline.run()
```

### カスタマイズ例

```python
# 設定をカスタマイズ
pipeline = PipelineFactory.create_default_pipeline(
    top_db=25,
    n_mels=128,
    audio_output_file="my_audio.wav",
)
pipeline.setup()
result = pipeline.run()
```

### カスタム実装を使用

```python
from core import (
    PipelineFactory,
    DefaultConfig,
    DefaultAudioRecorder,
    DefaultAudioPreprocessor,
    DefaultMelSpectrogramConverter,
    DefaultModelLoader,
    DefaultInferencer,
    DefaultAudioSaver,
    DefaultCountdownDisplay,
)
from REFACTORING_EXAMPLES import FileResultPresenter

# カスタム実装を作成
custom_presenter = FileResultPresenter(output_file="results.txt")

# パイプラインに注入
pipeline = PipelineFactory.create_custom_pipeline(
    config=DefaultConfig(),
    recorder=DefaultAudioRecorder(),
    preprocessor=DefaultAudioPreprocessor(),
    mel_converter=DefaultMelSpectrogramConverter(),
    model_loader=DefaultModelLoader(),
    inferencer=DefaultInferencer(DefaultModelLoader()),
    audio_saver=DefaultAudioSaver(),
    result_presenter=custom_presenter,  # ← カスタム実装
    countdown_display=DefaultCountdownDisplay(),
)

pipeline.setup()
result = pipeline.run()
```

---

## 📈 改善のポイント

### 1. 責務の分離

| クラス | 責務 |
|--------|------|
| `ConfigProvider` | 設定管理 |
| `AudioRecorder` | 音声入力 |
| `AudioPreprocessor` | 音声処理 |
| `MelSpectrogramConverter` | 特徴抽出 |
| `ModelLoader` | モデル管理 |
| `Inferencer` | 推論実行 |
| `AudioSaver` | ファイル出力 |
| `ResultPresenter` | 結果表示 |
| `MicrophonePredictionPipeline` | 全体の流れ管理 |

### 2. 拡張性

**新しい前処理アルゴリズムを追加**:
```python
class CustomPreprocessor(AudioPreprocessor):
    def preprocess(self, audio, sample_rate, target_length, top_db):
        # カスタム実装
        ...
```

**新しい出力形式を追加**:
```python
class JsonResultPresenter(ResultPresenter):
    def present(self, predicted_label, probabilities):
        # JSON形式で出力
        ...
```

**新しいモデルアーキテクチャに対応**:
```python
class TransformerModelLoader(ModelLoader):
    def load_model(self, model_path, num_classes, device):
        # Transformer実装をロード
        ...
```

### 3. テスト容易性

```python
# モック実装でテスト
class MockAudioRecorder(AudioRecorder):
    def record(self, duration, sample_rate):
        return np.random.normal(0, 0.1, sample_rate)

class MockResultPresenter(ResultPresenter):
    def __init__(self):
        self.results = []
    
    def present(self, predicted_label, probabilities):
        self.results.append({"label": predicted_label})

# テスト実行
pipeline = PipelineFactory.create_custom_pipeline(
    config=DefaultConfig(),
    recorder=MockAudioRecorder(),
    ...
    result_presenter=MockResultPresenter(),
    ...
)
```

---

## 🔄 オリジナルとの比較

### コード行数

| ファイル | 行数 |
|---------|------|
| `predict_from_mike.py`（元）| ~70行 |
| `predict_from_mike_refactored.py`（新）| ~20行 |
| 全体（core + エントリーポイント）| ~400行 |

### 複雑性

- **元のコード**: 1ファイルに全ての処理が混在
- **リファクタリング後**: 責務が明確に分離

### 変更の容易さ

- **元のコード**: 各処理を変更するにはメインファイルを編集
- **リファクタリング後**: 各インターフェース実装を新規作成するだけ

---

## ✅ 確認事項

- [ ] 機能は変更されていない（同じ結果が得られることを確認）
- [ ] `core/` ディレクトリが作成されている
- [ ] 全てのインターフェースが定義されている
- [ ] デフォルト実装が提供されている
- [ ] エントリーポイントがシンプルになっている
- [ ] カスタマイズ例が提供されている

---

## 🎓 学習ポイント

このリファクタリングから学べること：

1. **責務分離の重要性**: 各クラスが1つの責務のみを持つ
2. **インターフェースの力**: 具体的な実装に依存しない設計
3. **Strategy Pattern**: アルゴリズムの動的切り替え
4. **Dependency Injection**: 依存性の明示的な管理
5. **Factory Pattern**: 複雑なオブジェクト生成の集中管理
6. **テスト駆動設計**: モック実装でテスト可能な設計

---

## 🚀 次のステップ

1. **他のプログラムもリファクタリング**: `predict.py`、`evaluate.py` など
2. **ユニットテストの追加**: 各コンポーネントのテスト
3. **統合テストの追加**: パイプライン全体のテスト
4. **さらなるStrategy実装**: 異なるモデルアーキテクチャなど
5. **設定ファイル化**: YAMLやJSONで設定管理
