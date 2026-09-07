import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from voicerecognizer.config import DEFAULT_AUDIO_CONFIG
from voicerecognizer.preprocessing.dataset_builder import DatasetBuilder


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
            sf.write(speaker_dir / "001.wav", dummy_audio, DEFAULT_AUDIO_CONFIG.sample_rate)

            builder = DatasetBuilder(labels=("a",))

            # Test merge_by_label generating index.csv without copying wav files
            builder.merge_by_label(source_root=raw_root, output_root=merged_root)
            self.assertTrue((merged_root / "index.csv").exists())

            # Test preprocess_dataset reading directly from index.csv
            builder.preprocess_dataset(input_root=merged_root, output_root=processed_root)
            processed_wav_path = processed_root / "a" / "001.wav"
            self.assertTrue(processed_wav_path.exists())

            processed_index = processed_root / "index.csv"
            self.assertTrue(processed_index.exists())
            with open(processed_index, encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["label"], "a")
            self.assertEqual(Path(rows[0]["filepath"]), processed_wav_path)
            self.assertEqual(Path(rows[0]["source_filepath"]), speaker_dir / "001.wav")
            self.assertEqual(rows[0]["speaker"], "speaker1")
            self.assertGreaterEqual(float(rows[0]["speech_duration_ms"]), 0.0)
            self.assertAlmostEqual(float(rows[0]["processed_duration_ms"]), 600.0, places=1)

            # Verify saved audio format
            data, sr = sf.read(processed_wav_path)
            self.assertEqual(sr, DEFAULT_AUDIO_CONFIG.sample_rate)
            self.assertEqual(
                len(data),
                int(DEFAULT_AUDIO_CONFIG.sample_rate * builder.preprocessor.target_length_seconds),
            )

    def test_merge_collected_dataset_skips_unlabeled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            collected_dir = tmp_path / "collected"
            merged_root = tmp_path / "merged_dataset"
            collected_dir.mkdir(parents=True, exist_ok=True)

            dummy_audio = np.zeros(16000, dtype=np.float32)
            sf.write(collected_dir / "100_1.wav", dummy_audio, 16000)
            sf.write(collected_dir / "100_2.wav", dummy_audio, 16000)

            # 100_1: ground_truth未記入 ("") -> スキップされるべき
            # 100_2: ground_truth記入 ("a") -> インデックス化されるべき
            metadata_file = collected_dir / "metadata.csv"
            with open(metadata_file, "w", encoding="utf-8") as f:
                f.write("100_1,100_1.wav,a,\n")
                f.write("100_2,100_2.wav,a,a\n")

            builder = DatasetBuilder(labels=("a",))
            builder.merge_collected_dataset(collected_dir=collected_dir, output_root=merged_root)

            index_path = merged_root / "index.csv"
            self.assertTrue(index_path.exists())

            content = index_path.read_text(encoding="utf-8")
            self.assertIn("100_2.wav", content)
            self.assertNotIn("100_1.wav", content)

    def test_merge_collected_dataset_with_subdirectories(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            collected_dir = tmp_path / "collected"
            merged_root = tmp_path / "merged_dataset"
            pc_dir = collected_dir / "pc_12345678"
            pc_dir.mkdir(parents=True, exist_ok=True)

            dummy_audio = np.zeros(16000, dtype=np.float32)
            sf.write(pc_dir / "200_1.wav", dummy_audio, 16000)

            metadata_file = pc_dir / "metadata.csv"
            with open(metadata_file, "w", encoding="utf-8") as f:
                f.write("200_1,200_1.wav,a,a\n")

            builder = DatasetBuilder(labels=("a",))
            builder.merge_collected_dataset(collected_dir=collected_dir, output_root=merged_root)

            index_path = merged_root / "index.csv"
            self.assertTrue(index_path.exists())
            self.assertIn("200_1.wav", index_path.read_text(encoding="utf-8"))

            processed_root = tmp_path / "processed_dataset"
            builder.preprocess_dataset(input_root=merged_root, output_root=processed_root)
            with open(processed_root / "index.csv", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["speaker"], "pc_12345678")


if __name__ == "__main__":
    unittest.main()
