import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch

from config import (
    DEFAULT_AUDIO_CONFIG,
    DEFAULT_PREPROCESS_CONFIG,
    DEFAULT_RECOGNITION_CONFIG,
)
from core.interfaces import RecognitionStrategy
from models.cnn.hiragana_cnn import HiraganaCNN
from preprocessing.audio_preprocessor import AudioPreprocessor
from preprocessing.threshold_calculator import AbstractSilenceThresholdCalculator

logger = logging.getLogger(__name__)


class CNNRecognizer(RecognitionStrategy):
    def __init__(
        self,
        model_path: str | Path,
        labels: tuple[str, ...] | list[str],
        sample_rate: int = DEFAULT_AUDIO_CONFIG.sample_rate,
        target_length_seconds: float = DEFAULT_RECOGNITION_CONFIG.target_length_seconds,
        top_db: float = DEFAULT_PREPROCESS_CONFIG.top_db,
        n_mels: int = DEFAULT_PREPROCESS_CONFIG.n_mels,
        n_fft: int = DEFAULT_PREPROCESS_CONFIG.n_fft,
        hop_length: int = DEFAULT_PREPROCESS_CONFIG.hop_length,
        device: torch.device | None = None,
        threshold_calculator: AbstractSilenceThresholdCalculator | None = None,
    ):
        self.model_path = Path(model_path)
        labels_json_path = self.model_path.parent / "labels.json"
        if labels_json_path.exists():
            import json

            with open(labels_json_path, encoding="utf-8") as f:
                self.labels = tuple(json.load(f))
        else:
            self.labels = tuple(labels)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.audio_preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            target_length_seconds=target_length_seconds,
            top_db=top_db,
            threshold_calculator=threshold_calculator,
        )
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.model = self._load_model()
        logger.info("CNNレコグナイザーの初期化完了")

    def recognize(self, audio: Any) -> str:
        import time

        t_start = time.perf_counter()

        t_prep_start = time.perf_counter()
        waveform = self._preprocess(audio)
        mel_tensor = self._create_mel(waveform)
        t_prep_end = time.perf_counter()

        t_inf_start = time.perf_counter()
        probabilities = self._predict(mel_tensor)
        t_inf_end = time.perf_counter()

        predicted_index = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_index].item())

        prep_stats = getattr(self.audio_preprocessor, "last_stats", {})
        self.last_timing_stats = {
            "onset_ms": prep_stats.get("onset_ms", 0.0),
            "offset_ms": prep_stats.get("offset_ms", 0.0),
            "speech_duration_ms": prep_stats.get("speech_duration_ms", 0.0),
            "preprocess_latency_ms": (t_prep_end - t_prep_start) * 1000.0,
            "inference_latency_ms": (t_inf_end - t_inf_start) * 1000.0,
            "total_latency_ms": (t_inf_end - t_start) * 1000.0,
            "confidence": confidence,
        }

        logger.info(f"推論結果: {probabilities}")
        return self._postprocess(probabilities)

    def _load_model(self) -> HiraganaCNN:
        try:
            model = HiraganaCNN(num_classes=len(self.labels))
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
        except Exception as e:
            logger.error(e)
        model.to(self.device)
        model.eval()
        return model

    def _preprocess(self, audio: Any) -> np.ndarray:
        return self.audio_preprocessor.preprocess_waveform(audio)

    def _create_mel(self, waveform: np.ndarray) -> torch.Tensor:
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        return torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

    def _predict(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model(mel_tensor)
            return torch.softmax(logits, dim=1)[0]

    def _postprocess(self, probabilities: torch.Tensor) -> str:
        predicted_index = int(torch.argmax(probabilities).item())
        return self.labels[predicted_index]
