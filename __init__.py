"""
Voice Recognizer Package API

外部モジュールや他アプリケーションから簡単なコードで音声認識機能を呼び出せるよう、
主要クラスおよびショートカット関数をパッケージトップレベルでエクスポートします。

使い方:
  import voicerecognizer as vr

  # クラスを生成して利用
  recognizer = vr.Wav2Vec2Recognizer()
  result = recognizer.recognize("audio.wav")

  # リアルタイム非同期ストリーミング聴取
  listener = vr.AudioStreamListener()
  async for text in listener.listen():
      print(text)
"""

from pathlib import Path
from typing import Any

from core.exceptions import (
    AudioPreprocessingError,
    DeviceNotFoundError,
    ModelNotFoundError,
    VoiceRecognizerError,
)
from core.interfaces import RecognitionStrategy
from recognizers import CNNRecognizer, Wav2Vec2Recognizer, WhisperRecognizer
from runtime.stream_listener import AudioStreamListener, RecognitionResult

__version__ = "0.1.0"

__all__ = [
    "AudioPreprocessingError",
    "AudioStreamListener",
    "CNNRecognizer",
    "DeviceNotFoundError",
    "ModelNotFoundError",
    "RecognitionResult",
    "RecognitionStrategy",
    "VoiceRecognizerError",
    "Wav2Vec2Recognizer",
    "WhisperRecognizer",
    "recognize",
    "recognize_async",
]

_default_recognizer: Wav2Vec2Recognizer | None = None


def get_default_recognizer() -> Wav2Vec2Recognizer:
    """シングルトン感覚で利用できるデフォルト識別器インスタンスを取得"""
    global _default_recognizer
    if _default_recognizer is None:
        _default_recognizer = Wav2Vec2Recognizer()
    return _default_recognizer


def recognize(audio: str | Path | Any, model_type: str = "wav2vec2") -> str:
    """
    ワンライナーで音声をテキスト認識するショートカット関数。

    Args:
        audio: 音声ファイルパス (.wav) または numpy 配列の音声波形
        model_type: "wav2vec2" (標準・高速) または "cnn"

    Returns:
        認識されたひらがな文字列
    """
    if model_type == "wav2vec2":
        rec: RecognitionStrategy = get_default_recognizer()
        return rec.recognize(audio)
    elif model_type == "cnn":
        cnn_rec: RecognitionStrategy = CNNRecognizer()
        return cnn_rec.recognize(audio)
    else:
        raise ValueError(f"未対応のモデルタイプです: {model_type}")


async def recognize_async(audio: str | Path | Any, model_type: str = "wav2vec2") -> str:
    """非同期 (async/await) でノンブロッキング音声認識を行うショートカット関数"""
    if model_type == "wav2vec2":
        rec: RecognitionStrategy = get_default_recognizer()
        return await rec.recognize_async(audio)
    elif model_type == "cnn":
        cnn_rec: RecognitionStrategy = CNNRecognizer()
        return await cnn_rec.recognize_async(audio)
    else:
        raise ValueError(f"未対応のモデルタイプです: {model_type}")
