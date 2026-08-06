import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import DEFAULT_RECOGNITION_CONFIG
from core.interfaces import RecognitionStrategy
from preprocessing.audio_preprocessor import AudioPreprocessor

logger = logging.getLogger(__name__)


class Wav2Vec2Recognizer(RecognitionStrategy):
    def __init__(
        self,
        model_path: str | Path = DEFAULT_RECOGNITION_CONFIG.wav2vec2_best_model_dir,
        labels: tuple[str, ...] | list[str] = DEFAULT_RECOGNITION_CONFIG.labels,
        sample_rate: int = DEFAULT_RECOGNITION_CONFIG.sample_rate,
        target_length_seconds: float = DEFAULT_RECOGNITION_CONFIG.target_length_seconds,
        top_db: float = DEFAULT_RECOGNITION_CONFIG.top_db,
        device: torch.device | None = None,
    ):
        self.model_path = Path(model_path)
        self.labels = tuple(labels)
        self.sample_rate = sample_rate
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.audio_preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            target_length_seconds=target_length_seconds,
            top_db=top_db,
        )
        self.feature_extractor: Any | None = None
        self.model: Any | None = None
        self.last_confidence: float | None = None
        logger.info("Wav2Vec2Recognizer initialized: %s", self.model_path)

    def recognize(self, audio: Any) -> str:
        self._ensure_model_loaded()
        waveform = self.audio_preprocessor.preprocess_waveform(audio)
        inputs = self._prepare_inputs(waveform)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        predicted_index = int(torch.argmax(probabilities).item())
        self.last_confidence = float(probabilities[predicted_index].item())
        logger.info("Wav2Vec2 probabilities: %s", probabilities)
        return self._label_for_index(predicted_index)

    def _ensure_model_loaded(self) -> None:
        if self.model is not None and self.feature_extractor is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Fine-tuned Wav2Vec2 model was not found at "
                f"{self.model_path}. Train it with: "
                "python train.py --model wav2vec2"
            )

        try:
            from transformers import (
                AutoFeatureExtractor,
                Wav2Vec2ForSequenceClassification,
            )
        except ImportError as exc:
            raise ImportError(
                "Wav2Vec2 support requires the 'transformers' package. "
                "Install project dependencies with: uv sync"
            ) from exc

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.model_path
        )
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            self.model_path
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info("Loaded fine-tuned Wav2Vec2 model from %s", self.model_path)

    def _prepare_inputs(self, waveform: np.ndarray) -> dict[str, torch.Tensor]:
        inputs = self.feature_extractor(
            waveform,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        return {
            key: value.to(self.device)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }

    def _label_for_index(self, index: int) -> str:
        id2label = getattr(self.model.config, "id2label", None)
        if id2label is not None:
            label = id2label.get(index, id2label.get(str(index)))
            if label is not None:
                return str(label)
        return self.labels[index]
