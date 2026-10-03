import csv
import tempfile
import unittest
from pathlib import Path
from typing import Any

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


class TestPreprocessDatasetKeepsPreviousOnFailure(unittest.TestCase):
    """前処理が途中で失敗しても、既存の processed_dataset を壊さないことを保証する。"""

    def _build_raw_dataset(self, tmp_path: Path, filenames: tuple[str, ...]) -> Path:
        raw_root = tmp_path / "raw_dataset"
        speaker_dir = raw_root / "speaker1" / "a"
        speaker_dir.mkdir(parents=True, exist_ok=True)
        dummy_audio = np.zeros(DEFAULT_AUDIO_CONFIG.sample_rate, dtype=np.float32)
        for name in filenames:
            sf.write(speaker_dir / name, dummy_audio, DEFAULT_AUDIO_CONFIG.sample_rate)
        return raw_root

    def _assert_no_leftover_dirs(self, processed_root: Path) -> None:
        parent = processed_root.parent
        self.assertFalse((parent / f".{processed_root.name}.staging").exists())
        self.assertFalse((parent / f".{processed_root.name}.previous").exists())

    def test_existing_dataset_survives_midway_failure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_root = self._build_raw_dataset(tmp_path, ("001.wav", "002.wav"))
            merged_root = tmp_path / "merged_dataset"
            processed_root = tmp_path / "processed_dataset"

            builder = DatasetBuilder(labels=("a",))
            builder.merge_by_label(source_root=raw_root, output_root=merged_root)
            builder.preprocess_dataset(input_root=merged_root, output_root=processed_root)

            original_index = (processed_root / "index.csv").read_text(encoding="utf-8")
            original_wavs = sorted(p.name for p in (processed_root / "a").glob("*.wav"))
            self.assertEqual(len(original_wavs), 2)

            # 2 件目の前処理で失敗させる
            real_preprocess = builder.preprocessor.preprocess_waveform
            calls = {"n": 0}

            def failing_preprocess(audio: Any, *args: Any, **kwargs: Any) -> np.ndarray:
                calls["n"] += 1
                if calls["n"] == 2:
                    raise RuntimeError("simulated preprocessing failure")
                return real_preprocess(audio, *args, **kwargs)

            builder.preprocessor.preprocess_waveform = failing_preprocess
            with self.assertRaises(RuntimeError):
                builder.preprocess_dataset(input_root=merged_root, output_root=processed_root)

            # 既存データセットが丸ごと残っていること
            self.assertEqual(
                (processed_root / "index.csv").read_text(encoding="utf-8"), original_index
            )
            self.assertEqual(
                sorted(p.name for p in (processed_root / "a").glob("*.wav")), original_wavs
            )
            self._assert_no_leftover_dirs(processed_root)

    def test_existing_dataset_survives_missing_input(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_root = self._build_raw_dataset(tmp_path, ("001.wav",))
            merged_root = tmp_path / "merged_dataset"
            processed_root = tmp_path / "processed_dataset"

            builder = DatasetBuilder(labels=("a",))
            builder.merge_by_label(source_root=raw_root, output_root=merged_root)
            builder.preprocess_dataset(input_root=merged_root, output_root=processed_root)
            original_index = (processed_root / "index.csv").read_text(encoding="utf-8")

            with self.assertRaises(OSError):
                builder.preprocess_dataset(
                    input_root=tmp_path / "does_not_exist", output_root=processed_root
                )

            self.assertEqual(
                (processed_root / "index.csv").read_text(encoding="utf-8"), original_index
            )
            self.assertTrue((processed_root / "a" / "001.wav").exists())
            self._assert_no_leftover_dirs(processed_root)

    def test_successful_rerun_replaces_dataset_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_root = self._build_raw_dataset(tmp_path, ("001.wav", "002.wav"))
            merged_root = tmp_path / "merged_dataset"
            processed_root = tmp_path / "processed_dataset"

            builder = DatasetBuilder(labels=("a",))
            builder.merge_by_label(source_root=raw_root, output_root=merged_root)
            builder.preprocess_dataset(input_root=merged_root, output_root=processed_root)
            self.assertEqual(len(list((processed_root / "a").glob("*.wav"))), 2)

            # 入力を 1 件に減らして再実行 → 古い 2 件目は残らない
            (raw_root / "speaker1" / "a" / "002.wav").unlink()
            builder.merge_by_label(source_root=raw_root, output_root=merged_root)
            builder.preprocess_dataset(input_root=merged_root, output_root=processed_root)

            self.assertEqual(len(list((processed_root / "a").glob("*.wav"))), 1)
            with open(processed_root / "index.csv", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            # index.csv の filepath は最終パス (processed_dataset 配下) を指すこと
            self.assertNotIn(".staging", rows[0]["filepath"])
            self.assertTrue(Path(rows[0]["filepath"]).name.endswith(".wav"))
            self._assert_no_leftover_dirs(processed_root)


if __name__ == "__main__":
    unittest.main()
