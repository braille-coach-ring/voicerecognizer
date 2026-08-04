import tempfile
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf

from config import DEFAULT_AUDIO_CONFIG
from preprocessing.dataset_builder import DatasetBuilder


class TestDatasetBuilderIsolated(unittest.TestCase):
    def test_merge_and_preprocess_in_temp_dirs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_root = tmp_path / "raw_dataset"
            merged_root = tmp_path / "merged_dataset"
            processed_root = tmp_path / "processed_dataset"

            # Create dummy raw data for speaker "speaker1" and label "a"
            speaker_dir = raw_root / "speaker1" / "a"
            speaker_dir.mkdir(parents=True, exist_ok=True)
            dummy_audio = np.sin(
                np.linspace(0, 440 * 2 * np.pi, DEFAULT_AUDIO_CONFIG.sample_rate)
            ).astype(np.float32)
            sf.write(
                speaker_dir / "001.wav", dummy_audio, DEFAULT_AUDIO_CONFIG.sample_rate
            )

            builder = DatasetBuilder(labels=("a",))

            # Test merge_by_label without touching repo dataset
            builder.merge_by_label(source_root=raw_root, output_root=merged_root)
            self.assertTrue((merged_root / "a" / "001.wav").exists())

            # Test preprocess_dataset without touching repo dataset
            builder.preprocess_dataset(
                input_root=merged_root, output_root=processed_root
            )
            self.assertTrue((processed_root / "a" / "001.wav").exists())

            # Verify saved audio format
            data, sr = sf.read(processed_root / "a" / "001.wav")
            self.assertEqual(sr, DEFAULT_AUDIO_CONFIG.sample_rate)
            self.assertEqual(
                len(data),
                int(
                    DEFAULT_AUDIO_CONFIG.sample_rate
                    * builder.preprocessor.target_length_seconds
                ),
            )


if __name__ == "__main__":
    unittest.main()
