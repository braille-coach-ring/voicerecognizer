from pathlib import Path
from typing import Any

import librosa
import numpy as np
import logging
logger = logging.getLogger(__name__)

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
            threshold_calculator
            or FixedSilenceThresholdCalculator(top_db=float(top_db))
        )
        logger.info("AudioPreprocessorの初期化完了")

    def load(self, audio: str | Path | np.ndarray) -> np.ndarray:
        if isinstance(audio, np.ndarray):
            logger.info("AudioPreprocessor is instance")
            return audio.astype(np.float32).reshape(-1)

        waveform, _ = librosa.load(Path(audio), sr=self.sample_rate, mono=True)
        logger.info("AudioPreprocessor is loaded from file")
        return waveform.astype(np.float32)

    def preprocess_waveform(self, audio: Any) -> np.ndarray:
        waveform = self.load(audio)
        self.threshold_calculator.update(waveform)
        current_top_db = self.threshold_calculator.get_silence_threshold()

        # 1. 無音境界の検索
        intervals = librosa.effects.split(waveform, top_db=current_top_db)
        if len(intervals) > 0:
            start_idx = intervals[0][0]
            end_idx = intervals[-1][1]

            # 2. 「頭切れ」防止マージン (前後に約50msの余裕を確保)
            margin_samples = int(self.sample_rate * 0.05)  # 50ms
            start_idx = max(0, start_idx - margin_samples)
            end_idx = min(len(waveform), end_idx + margin_samples)
            waveform = waveform[start_idx:end_idx]

        # 3. 「ブツッ」という波形不連続ノイズ（クリック音）を抑えるソフトフェード処理 (5ms)
        fade_samples = int(self.sample_rate * 0.005)  # 5ms
        if len(waveform) > fade_samples * 2:
            fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
            waveform[:fade_samples] *= fade_in
            waveform[-fade_samples:] *= fade_out

        waveform = self._normalize_volume(waveform)
        logger.info("AudioPreprocessor is normalized volume")
        return self._fit_length(waveform)

    def _normalize_volume(self, waveform: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(waveform)) if waveform.size else 0
        if peak > 0:
            logger.info("AudioPreprocessor is normalized volume")
            return waveform / peak
        return waveform

    def _fit_length(self, waveform: np.ndarray) -> np.ndarray:
        target_samples = int(self.target_length_seconds * self.sample_rate)
        logger.info("AudioPreprocessor._fit_length: %s", target_samples)
        if len(waveform) > target_samples:
            return waveform[:target_samples]
        return np.pad(waveform, (0, target_samples - len(waveform)))
