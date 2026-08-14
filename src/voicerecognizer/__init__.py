"""
Voice Recognizer Package API

外部モジュールや他アプリケーションから簡単に音声認識・ストリーミング機能を呼び出せるよう、
主要クラスおよび例外をパッケージトップレベルでエクスポートします。

使い方:
  import voicerecognizer as vr

  # 1. Wav2Vec2 ONNX 認識器（クラス初期化時に重みがなければ Hugging Face より自動ダウンロード）
  recognizer = vr.Wav2Vec2Recognizer()
  text = recognizer.recognize("audio.wav")

  # 2. CNN 認識器
  cnn_recognizer = vr.CNNRecognizer()
  text = cnn_recognizer.recognize("audio.wav")

  # 3. リアルタイム非同期ストリーミング聴取
  listener = vr.AudioStreamListener(recognizer=recognizer)
  async for result in listener.listen():
      print(f"認識文字: {result.text} (確信度: {result.confidence:.2f})")
"""

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
]
