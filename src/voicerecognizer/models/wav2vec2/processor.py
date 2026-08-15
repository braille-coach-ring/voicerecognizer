import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from voicerecognizer.config import (
    DEFAULT_AUDIO_CONFIG,
    DEFAULT_PREPROCESS_CONFIG,
    DEFAULT_RECOGNITION_CONFIG,
)
from voicerecognizer.core.exceptions import ModelNotFoundError
from voicerecognizer.preprocessing.audio_preprocessor import AudioPreprocessor

logger = logging.getLogger(__name__)


class Wav2Vec2Processor:
    def __init__(
        self,
        model_name_or_path: str | Path = DEFAULT_RECOGNITION_CONFIG.wav2vec2_best_model_dir,
        sample_rate: int = DEFAULT_AUDIO_CONFIG.sample_rate,
        target_length_seconds: float = DEFAULT_RECOGNITION_CONFIG.target_length_seconds,
        top_db: float = DEFAULT_PREPROCESS_CONFIG.top_db,
    ):
        self.model_name_or_path = model_name_or_path
        self.sample_rate = sample_rate
        self.audio_preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            target_length_seconds=target_length_seconds,
            top_db=top_db,
        )
        self.feature_extractor: Any | None = None

    def prepare(self, audio: Any) -> dict[str, torch.Tensor]:
        waveform = self.audio_preprocessor.preprocess_waveform(audio)
        return self.prepare_waveform(waveform)

    def prepare_waveform(self, waveform: np.ndarray) -> dict[str, torch.Tensor]:
        self._ensure_feature_extractor()
        if self.feature_extractor is None:
            logger.error(
                "Wav2Vec2 FeatureExtractor が初期化されていません (None): %s",
                self.model_name_or_path,
            )
            raise ModelNotFoundError(
                f"Wav2Vec2 FeatureExtractor がロードできませんでした: {self.model_name_or_path}"
            )
        return self.feature_extractor(
            waveform,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )

    def _ensure_feature_extractor(self) -> None:
        if self.feature_extractor is not None:
            return

        try:
            from transformers import AutoFeatureExtractor
        except ImportError as exc:
            raise ImportError(
                "Wav2Vec2 preprocessing requires the 'transformers' package. "
                "Install project dependencies with: uv sync"
            ) from exc

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name_or_path)
