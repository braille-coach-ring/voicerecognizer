"""
Voice Recognizer Package API

外部モジュールや他アプリケーションから簡単なコードで音声認識機能を呼び出せるよう、
主要クラスおよびショートカット関数をパッケージトップレベルでエクスポートします。

使い方:
  import voicerecognizer as vr

  # クラスを生成して利用
  recognizer = vr.Wav2Vec2Recognizer()
  result = recognizer.recognize("audio.wav")

  # ショートカット関数 (初回呼び出し時に weights 未存在なら Hugging Face より自動ダウンロード)
  text = vr.recognize("audio.wav", model_type="wav2vec2")

  # リアルタイム非同期ストリーミング聴取
  listener = vr.AudioStreamListener()
  async for text in listener.listen():
      print(text)
"""

import logging
from pathlib import Path
from typing import Any

from voicerecognizer.core.exceptions import (
    AudioPreprocessingError,
    DeviceNotFoundError,
    ModelNotFoundError,
    VoiceRecognizerError,
)
from voicerecognizer.core.interfaces import RecognitionStrategy
from voicerecognizer.recognizers.cnn_recognizer import CNNRecognizer
from voicerecognizer.recognizers.wav2vec2_recognizer import Wav2Vec2Recognizer
from voicerecognizer.recognizers.whisper_recognizer import WhisperRecognizer
from voicerecognizer.runtime.stream_listener import AudioStreamListener, RecognitionResult

logger = logging.getLogger(__name__)

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
_default_cnn_recognizer: CNNRecognizer | None = None


def _ensure_weights_downloaded(model_type: str = "wav2vec2") -> None:
    """必要に応じて Hugging Face Hub より最新重みファイルを自動ダウンロードしてキャッシュに配置"""
    try:
        from voicerecognizer.utils.model_uploader import download_latest_team_weights_if_needed

        if model_type in ("wav2vec2", "cnn"):
            download_latest_team_weights_if_needed(model_type=model_type)  # type: ignore[arg-type]
    except Exception as e:
        logger.debug("モデル重みの自動同期チェックをスキップしました: %s", e)


def get_default_recognizer() -> Wav2Vec2Recognizer:
    """シングルトン感覚で利用できるデフォルト Wav2Vec2 識別器インスタンスを取得"""
    global _default_recognizer
    if _default_recognizer is None:
        _ensure_weights_downloaded("wav2vec2")
        _default_recognizer = Wav2Vec2Recognizer()
    return _default_recognizer


def get_default_cnn_recognizer() -> CNNRecognizer:
    """シングルトン感覚で利用できるデフォルト CNN 識別器インスタンスを取得"""
    global _default_cnn_recognizer
    if _default_cnn_recognizer is None:
        _ensure_weights_downloaded("cnn")
        _default_cnn_recognizer = CNNRecognizer()
    return _default_cnn_recognizer


def recognize(audio: str | Path | Any, model_type: str = "wav2vec2") -> str:
    """
    ワンライナーで音声をテキスト認識するショートカット関数。

    Args:
        audio: 音声ファイルパス (.wav) または numpy 配列の音声波形
        model_type: "wav2vec2" (標準・高精度) または "cnn"

    Returns:
        認識されたひらがな文字列
    """
    if model_type == "wav2vec2":
        rec: RecognitionStrategy = get_default_recognizer()
        return rec.recognize(audio)
    elif model_type == "cnn":
        cnn_rec: RecognitionStrategy = get_default_cnn_recognizer()
        return cnn_rec.recognize(audio)
    else:
        raise ValueError(f"未対応のモデルタイプです: {model_type}")


async def recognize_async(audio: str | Path | Any, model_type: str = "wav2vec2") -> str:
    """非同期 (async/await) でノンブロッキング音声認識を行うショートカット関数"""
    if model_type == "wav2vec2":
        rec: RecognitionStrategy = get_default_recognizer()
        return await rec.recognize_async(audio)
    elif model_type == "cnn":
        cnn_rec: RecognitionStrategy = get_default_cnn_recognizer()
        return await cnn_rec.recognize_async(audio)
    else:
        raise ValueError(f"未対応のモデルタイプです: {model_type}")
