from pathlib import Path

import numpy as np
import librosa
import torch
from torch.utils.data import Dataset


class HiraganaDataset(Dataset):
    def __init__(self, root_dir="processed_dataset", sample_rate=16000, n_mels=64):

        self.root = Path(root_dir)
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        # ラベル一覧
        self.labels = sorted([d.name for d in self.root.iterdir() if d.is_dir()])

        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}

        self.data = []

        for label in self.labels:
            folder = self.root / label

            for wav in folder.glob("*.wav"):
                self.data.append((wav, self.label_to_idx[label]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        wav_path, label = self.data[idx]

        y, sr = librosa.load(wav_path, sr=self.sample_rate)

        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=400, hop_length=160, n_mels=self.n_mels
        )

        mel = librosa.power_to_db(mel, ref=np.max)

        mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        mel = torch.tensor(mel, dtype=torch.float32)

        mel = mel.unsqueeze(0)

        label = torch.tensor(label, dtype=torch.long)

        return mel, label
