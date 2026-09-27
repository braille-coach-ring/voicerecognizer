"""CSV フィールドにカンマが含まれても列がずれないことのテスト。

metadata.csv / index.csv を素朴な split(",") で読み書きすると、
カンマを 1 つ含む値だけで列がずれ、誤ったラベルで学習してしまう。
"""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from voicerecognizer.config import DEFAULT_AUDIO_CONFIG
from voicerecognizer.dataset.hiragana_dataset import read_index_rows
from voicerecognizer.preprocessing.dataset_builder import DatasetBuilder
from voicerecognizer.runtime.output_worker import OutputWorker


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        path,
        np.zeros(DEFAULT_AUDIO_CONFIG.sample_rate // 2, dtype=np.float32),
        DEFAULT_AUDIO_CONFIG.sample_rate,
    )


class TestOutputWorkerMetadataQuoting(unittest.TestCase):
    def test_comma_in_predicted_text_keeps_four_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            save_dir = Path(tmp) / "pc_test"
            worker = OutputWorker(save_dir=save_dir)
            worker.save(
                audio_data=None,
                predicted_text="a,b",
                timestamp=1700000000.5,
                ground_truth="a",
            )

            with open(save_dir / "metadata.csv", encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f))

            self.assertEqual(len(rows), 1)
            self.assertEqual(len(rows[0]), 4)
            self.assertEqual(rows[0][2], "a,b")
            self.assertEqual(rows[0][3], "a")


class TestBuildIndexMetadataQuoting(unittest.TestCase):
    def test_comma_in_predicted_text_does_not_shift_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            collected_dir = tmp_path / "collected"
            pc_dir = collected_dir / "pc_12345678"
            _write_wav(pc_dir / "100_1.wav")

            # predicted_text にカンマを含む行 (csv 引用付き)
            with open(pc_dir / "metadata.csv", "w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(["100_1", "100_1.wav", "a,b", "a"])

            merged_root = tmp_path / "merged_dataset"
            builder = DatasetBuilder(labels=("a",))
            builder.merge_collected_dataset(collected_dir=collected_dir, output_root=merged_root)

            rows = list(read_index_rows(merged_root / "index.csv"))
            self.assertEqual(len(rows), 1, "ground_truth の列ずれで行が落ちている")
            self.assertEqual(rows[0][1], "a")


class TestIndexReadingQuoting(unittest.TestCase):
    def test_comma_in_filepath_keeps_label_aligned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index_file = Path(tmp) / "index.csv"
            with open(index_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["filepath", "label", "predicted_text"])
                writer.writerow(["dataset/speaker1/a/a,1.wav", "a", ""])
                writer.writerow(["dataset/speaker1/i/plain.wav", "i", "a,b"])

            rows = list(read_index_rows(index_file))

            self.assertEqual(
                rows,
                [
                    ("dataset/speaker1/a/a,1.wav", "a"),
                    ("dataset/speaker1/i/plain.wav", "i"),
                ],
            )

    def test_bom_prefixed_header_is_handled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index_file = Path(tmp) / "index.csv"
            index_file.write_text(
                "﻿filepath,label,predicted_text\ndataset/a/001.wav,a,\n",
                encoding="utf-8",
            )
            self.assertEqual(list(read_index_rows(index_file)), [("dataset/a/001.wav", "a")])


class TestHiraganaDatasetEndToEnd(unittest.TestCase):
    def test_dataset_loads_file_whose_name_contains_comma(self) -> None:
        from voicerecognizer.dataset.hiragana_dataset import HiraganaDataset

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wav_path = tmp_path / "a" / "a,1.wav"
            _write_wav(wav_path)

            index_file = tmp_path / "index.csv"
            with open(index_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["filepath", "label", "predicted_text"])
                writer.writerow(["a/a,1.wav", "a", ""])

            dataset = HiraganaDataset(root_dir=tmp_path)

            self.assertEqual(dataset.labels, ["a"])
            self.assertEqual(len(dataset.data), 1)
            self.assertEqual(dataset.data[0][0], wav_path)


if __name__ == "__main__":
    unittest.main()
