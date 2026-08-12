from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

import logging
logger = logging.getLogger(__name__)

from config import (
    DEFAULT_AUDIO_CONFIG,
    DEFAULT_PREPROCESS_CONFIG,
    DEFAULT_RECOGNITION_CONFIG,
)
from preprocessing.audio_preprocessor import AudioPreprocessor


class HiraganaDataset(Dataset):
    """
    PyTorch 用のデータセットクラス。
    インデックスファイル (index.csv) またはディレクトリ構造から音声データとラベルを読み込み、
    メルスペクトログラムを抽出して出力します。
    """

    def __init__(
        self,
        root_dir: str | Path = DEFAULT_RECOGNITION_CONFIG.processed_dataset_dir,
        sample_rate: int = DEFAULT_AUDIO_CONFIG.sample_rate,
        n_mels: int = DEFAULT_PREPROCESS_CONFIG.n_mels,
        n_fft: int = DEFAULT_PREPROCESS_CONFIG.n_fft,
        hop_length: int = DEFAULT_PREPROCESS_CONFIG.hop_length,
        cache_in_memory: bool = False,
    ):
        self.root = Path(root_dir)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate)
        if (self.root / "index.csv").exists():
            self.index_file = self.root / "index.csv"
        elif self.root.is_file():
            self.index_file = self.root
        else:
            self.index_file = None

        if self.index_file:
            # index.csv からラベル一覧を自動取得
            labels_set = set()
            with open(self.index_file, "r", encoding="utf-8") as f:
                f.readline()
                for line in f:
                    parts = [p.strip() for p in line.strip().split(",")]
                    if len(parts) >= 2 and parts[1]:
                        labels_set.add(parts[1])
            self.labels = sorted(labels_set)
        else:
            self.labels = sorted(path.name for path in self.root.iterdir() if path.is_dir())

        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.data = self._collect_files()

        self.cached_mels: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        if cache_in_memory and self.data:
            logger.info("オンメモリキャッシュ作成中: %d件のメルスペクトログラムを計算中...", len(self.data))
            self.cached_mels = []
            for wav_path, label in self.data:
                waveform = self.preprocessor.preprocess_waveform(wav_path)
                mel = self._create_mel(waveform)
                self.cached_mels.append((mel, torch.tensor(label, dtype=torch.long)))

        logger.info(f"HiraganaDatasetのロード完了: 全 %d件 (クラス数 %d)", len(self.data), len(self.labels))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        if self.cached_mels is not None:
            return self.cached_mels[idx]

        wav_path, label = self.data[idx]
        waveform = self.preprocessor.preprocess_waveform(wav_path)
        mel = self._create_mel(waveform)
        return mel, torch.tensor(label, dtype=torch.long)

    def _collect_files(self) -> list[tuple[Path, int]]:
        data = []
        if self.index_file and self.index_file.exists():
            from config import PROJECT_ROOT
            with open(self.index_file, "r", encoding="utf-8") as f:
                f.readline()
                for line in f:
                    parts = [p.strip() for p in line.strip().split(",")]
                    if len(parts) < 2 or not parts[0]:
                        continue
                    wav_path = Path(parts[0])
                    if not wav_path.is_absolute():
                        wav_path = PROJECT_ROOT / wav_path
                    label = parts[1]
                    if wav_path.exists() and label in self.label_to_idx:
                        data.append((wav_path, self.label_to_idx[label]))
            return data

        for label in self.labels:
            folder = self.root / label
            if folder.is_dir():
                for wav_path in folder.glob("*.wav"):
                    data.append((wav_path, self.label_to_idx[label]))
        return data

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
        return torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
