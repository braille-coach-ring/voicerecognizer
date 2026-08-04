from pathlib import Path
from typing import Any

import librosa
import numpy as np

from config import (
    DEFAULT_AUDIO_CONFIG,
    DEFAULT_PREPROCESS_CONFIG,
    DEFAULT_RECOGNITION_CONFIG,
)
from preprocessing.threshold_calculator import (
    AbstractSilenceThresholdCalculator,
    FixedSilenceThresholdCalculator,
)


class AudioPreprocessor:
    def __init__(
        self,
        sample_rate: int = DEFAULT_AUDIO_CONFIG.sample_rate,
        target_length_seconds: float = DEFAULT_RECOGNITION_CONFIG.target_length_seconds,
        top_db: float = DEFAULT_PREPROCESS_CONFIG.top_db,
        threshold_calculator: AbstractSilenceThresholdCalculator | None = None,
    ):
        self.sample_rate = sample_rate
        self.target_length_seconds = target_length_seconds
        self.top_db = top_db
        self.threshold_calculator = (
            threshold_calculator or FixedSilenceThresholdCalculator(top_db=float(top_db))
        )

    def load(self, audio: str | Path | np.ndarray) -> np.ndarray:
        if isinstance(audio, np.ndarray):
            return audio.astype(np.float32).reshape(-1)

        waveform, _ = librosa.load(Path(audio), sr=self.sample_rate, mono=True)
        return waveform.astype(np.float32)

    def preprocess_waveform(self, audio: Any) -> np.ndarray:
        waveform = self.load(audio)
        self.threshold_calculator.update(waveform)
        current_top_db = self.threshold_calculator.get_silence_threshold()
        waveform, _ = librosa.effects.trim(waveform, top_db=current_top_db)
        waveform = self._normalize_volume(waveform)
        return self._fit_length(waveform)

    def _normalize_volume(self, waveform: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(waveform)) if waveform.size else 0
        if peak > 0:
            return waveform / peak
        return waveform

    def _fit_length(self, waveform: np.ndarray) -> np.ndarray:
        target_samples = int(self.target_length_seconds * self.sample_rate)
        if len(waveform) > target_samples:
            return waveform[:target_samples]
        return np.pad(waveform, (0, target_samples - len(waveform)))
