from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.audio_preprocessor import AudioPreprocessor


class HiraganaDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path = "processed_dataset",
        sample_rate: int = 16000,
        n_mels: int = 64,
    ):
        self.root = Path(root_dir)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate)
        self.labels = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.data = self._collect_files()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        wav_path, label = self.data[idx]
        waveform = self.preprocessor.preprocess_waveform(wav_path)
        mel = self._create_mel(waveform)
        return mel, torch.tensor(label, dtype=torch.long)

    def _collect_files(self) -> list[tuple[Path, int]]:
        data = []
        for label in self.labels:
            folder = self.root / label
            for wav_path in folder.glob("*.wav"):
                data.append((wav_path, self.label_to_idx[label]))
        return data

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
        return torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
