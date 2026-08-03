from pathlib import Path
from typing import Any

import librosa
import numpy as np


class AudioPreprocessor:
    def __init__(
        self,
        sample_rate: int = 16000,
        target_length_seconds: float = 1.0,
        top_db: int = 30,
    ):
        self.sample_rate = sample_rate
        self.target_length_seconds = target_length_seconds
        self.top_db = top_db

    def load(self, audio: str | Path | np.ndarray) -> np.ndarray:
        if isinstance(audio, np.ndarray):
            return audio.astype(np.float32).reshape(-1)

        waveform, _ = librosa.load(Path(audio), sr=self.sample_rate, mono=True)
        return waveform.astype(np.float32)

    def preprocess_waveform(self, audio: Any) -> np.ndarray:
        waveform = self.load(audio)
        waveform, _ = librosa.effects.trim(waveform, top_db=self.top_db)
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
