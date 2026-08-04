from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch

from core.interfaces import RecognitionStrategy
from models.cnn.hiragana_cnn import HiraganaCNN
from preprocessing.audio_preprocessor import AudioPreprocessor
from preprocessing.threshold_calculator import AbstractSilenceThresholdCalculator


class CNNRecognizer(RecognitionStrategy):
    def __init__(
        self,
        model_path: str | Path,
        labels: tuple[str, ...] | list[str],
        sample_rate: int = 16000,
        target_length_seconds: float = 1.0,
        top_db: int = 30,
        n_mels: int = 64,
        device: torch.device | None = None,
        threshold_calculator: AbstractSilenceThresholdCalculator | None = None,
    ):
        self.model_path = Path(model_path)
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
        self.model = self._load_model()

    def recognize(self, audio: Any) -> str:
        waveform = self._preprocess(audio)
        mel_tensor = self._create_mel(waveform)
        probabilities = self._predict(mel_tensor)
        return self._postprocess(probabilities)

    def _load_model(self) -> HiraganaCNN:
        model = HiraganaCNN(num_classes=len(self.labels))
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _preprocess(self, audio: Any) -> np.ndarray:
        return self.audio_preprocessor.preprocess_waveform(audio)

    def _create_mel(self, waveform: np.ndarray) -> torch.Tensor:
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=400,
            hop_length=160,
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
