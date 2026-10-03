"""Strategy 02: Wav2Vec2 + Whisper Hidden Feature Knowledge Distillation."""

from voicerecognizer.strategies.whisper_kd.model import Wav2Vec2ForWhisperKDClassification
from voicerecognizer.strategies.whisper_kd.teacher import WhisperTeacher

__all__ = ["Wav2Vec2ForWhisperKDClassification", "WhisperTeacher"]
