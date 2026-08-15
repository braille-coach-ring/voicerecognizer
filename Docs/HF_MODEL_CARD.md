---
language:
- ja
license: mit
tags:
- audio
- speech-recognition
- audio-classification
- onnx
- wav2vec2
- japanese
- hiragana
pipeline_tag: audio-classification
---

# braille-mate-hiragana-recognizer

日本語ひらがな105文字（清音・濁音・半濁音・拗音）を高精度かつ超低遅延に分類認識するための Wav2Vec2 + ONNX INT8 量子化モデルです。

## モデル概要

- ベースモデル: `facebook/wav2vec2-base`
- 分類クラス数: 105 クラス（日本語ひらがな全音）
- 入力サンプリングレート: 16,000 Hz (モノラル)
- 最適化: ONNX INT8 Dynamic Quantization + 前処理内包型モデル (`model_mel_int8.onnx`)
- 推論エンジン: ONNX Runtime (CPU 最適化)

## ファイル構成

本リポジトリには以下のファイルが含まれています。

- `model_mel_int8.onnx`: 前処理（Mel/Waveform）を内包した INT8 量子化 ONNX モデル（推奨・デフォルト）
- `model_int8.onnx`: 標準 INT8 量子化 ONNX モデル
- `labels.json`: 105 クラスのひらがなラベル対応表
- `config.json`: Wav2Vec2 モデル設定
- `preprocessor_config.json`: 音声特徴量抽出設定

## 利用方法

`voicerecognizer` パッケージを使用することで、認証トークンや設定ファイル不要で即座にモデルをロードして推論を実行できます。

### インストール

```bash
pip install git+https://github.com/braille-coach-ring/voicerecognizer.git
```

### Python コード例

```python
import voicerecognizer as vr

# リポジトリから自動的に model_mel_int8.onnx がダウンロードされます
recognizer = vr.Wav2Vec2Recognizer()

# 音声ファイルまたは波形配列から推論
result = recognizer.recognize("audio.wav")
print(f"認識文字: {result}")

# 候補と確信度スコアの取得
candidates = recognizer.recognize_with_candidates("audio.wav", top_k=3)
for label, score in candidates:
    print(f"候補: {label} (スコア: {score:.4f})")
```

## ライセンス

MIT License
