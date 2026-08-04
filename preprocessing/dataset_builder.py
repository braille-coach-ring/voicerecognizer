from pathlib import Path
import shutil

import soundfile as sf

from config import (
    DEFAULT_AUDIO_CONFIG,
    DEFAULT_PREPROCESS_CONFIG,
    DEFAULT_RECOGNITION_CONFIG,
)
from preprocessing.audio_preprocessor import AudioPreprocessor


class DatasetBuilder:
    def __init__(
        self,
        labels: tuple[str, ...] = DEFAULT_RECOGNITION_CONFIG.labels,
        sample_rate: int = DEFAULT_AUDIO_CONFIG.sample_rate,
        target_length_seconds: float = DEFAULT_RECOGNITION_CONFIG.target_length_seconds,
        top_db: float = DEFAULT_PREPROCESS_CONFIG.top_db,
    ):
        self.labels = labels
        self.preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            target_length_seconds=target_length_seconds,
            top_db=top_db,
        )

    def merge_by_label(
        self,
        source_root: str | Path = DEFAULT_RECOGNITION_CONFIG.raw_dataset_dir,
        output_root: str | Path = DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir,
    ) -> None:
        source_root = Path(source_root)
        output_root = Path(output_root)
        self._recreate_label_dirs(output_root)

        for label in self.labels:
            output_folder = output_root / label
            count = 1

            for person in sorted(source_root.iterdir()):
                input_folder = person / label
                if not input_folder.is_dir():
                    continue

                for wav_path in sorted(input_folder.glob("*.wav")):
                    destination = output_folder / f"{count:03d}.wav"
                    shutil.copy2(wav_path, destination)
                    count += 1

    def preprocess_dataset(
        self,
        input_root: str | Path = DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir,
        output_root: str | Path = DEFAULT_RECOGNITION_CONFIG.processed_dataset_dir,
    ) -> None:
        input_root = Path(input_root)
        output_root = Path(output_root)
        if output_root.exists():
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True)

        for label_dir in sorted(input_root.iterdir()):
            if not label_dir.is_dir():
                continue

            output_dir = output_root / label_dir.name
            output_dir.mkdir(exist_ok=True)
            file_number = 1

            for wav_path in sorted(label_dir.glob("*.wav")):
                waveform = self.preprocessor.preprocess_waveform(wav_path)
                sf.write(output_dir / f"{file_number:03d}.wav", waveform, self.preprocessor.sample_rate)
                file_number += 1

    def _recreate_label_dirs(self, output_root: Path) -> None:
        output_root.mkdir(exist_ok=True)
        for label in self.labels:
            folder = output_root / label
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True)
