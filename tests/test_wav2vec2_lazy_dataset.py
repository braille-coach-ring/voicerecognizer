import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from voicerecognizer.models.wav2vec2.train import (
    Wav2Vec2ClassificationDataset,
    build_collate_fn,
    build_parser,
    determine_optimal_num_workers,
    is_pagefile_or_memory_error,
    load_local_safetensors_streaming,
    load_wav2vec2_classifier,
)


class FakeFeatureExtractor:
    def __call__(
        self,
        waveforms: list[np.ndarray],
        sampling_rate: int,
        return_tensors: str,
        padding: bool,
    ) -> dict[str, torch.Tensor]:
        max_len = max(len(waveform) for waveform in waveforms)
        padded = np.zeros((len(waveforms), max_len), dtype=np.float32)
        for index, waveform in enumerate(waveforms):
            padded[index, : len(waveform)] = waveform
        return {"input_values": torch.tensor(padded)}


class TinyConfig:
    pass


class TinyConfigLoader:
    @staticmethod
    def from_pretrained(model_source: str | Path) -> TinyConfig:
        return TinyConfig()


class TinyModel(torch.nn.Linear):
    def __init__(self, config: TinyConfig) -> None:
        super().__init__(2, 2)
        self.config = config

    @classmethod
    def from_pretrained(cls, *args: object, **kwargs: object) -> "TinyModel":
        raise MemoryError()


class TestWav2Vec2LazyDataset(unittest.TestCase):
    def test_dataset_keeps_paths_only_and_dataloader_builds_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_rate = 16000
            labels = ("a", "e")
            for label in labels:
                label_dir = root / label
                label_dir.mkdir(parents=True)
                for i in range(2):
                    waveform = np.full(sample_rate, 0.01 * (i + 1), dtype=np.float32)
                    sf.write(label_dir / f"{i + 1:03d}.wav", waveform, sample_rate)

            dataset = Wav2Vec2ClassificationDataset(
                root_dir=root,
                sample_rate=sample_rate,
                target_length_seconds=1.0,
                top_db=20.0,
            )

            self.assertEqual(len(dataset), 4)
            self.assertEqual(tuple(dataset.labels), labels)
            self.assertFalse(hasattr(dataset, "cached_data"))
            self.assertIsInstance(dataset.data[0][0], Path)
            self.assertEqual(dataset.speaker_ids, ["unknown"] * 4)
            pickle.dumps(dataset)

            waveform, label_idx = dataset[0]
            self.assertEqual(waveform.shape, (sample_rate,))
            self.assertIsInstance(label_idx, int)

            collate_fn = build_collate_fn(FakeFeatureExtractor(), sample_rate)
            pickle.dumps(collate_fn)

            loader = DataLoader(
                dataset,
                batch_size=2,
                collate_fn=collate_fn,
            )
            batch = next(iter(loader))

            self.assertEqual(batch["input_values"].shape, (2, sample_rate))
            self.assertEqual(batch["labels"].shape, (2,))

    def test_epoch_alias_is_supported(self) -> None:
        args = build_parser().parse_args(["--epoch", "1"])

        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.split_mode, "speaker")

    def test_streaming_safetensors_loader_copies_matching_tensors(self) -> None:
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = torch.nn.Linear(2, 2)
            with torch.no_grad():
                model.weight.zero_()
                model.bias.zero_()

            expected_weight = torch.ones_like(model.weight)
            expected_bias = torch.arange(2, dtype=torch.float32)
            save_file(
                {
                    "weight": expected_weight,
                    "bias": expected_bias,
                    "unexpected": torch.ones(1),
                },
                root / "model.safetensors",
            )

            missing, mismatched, unexpected = load_local_safetensors_streaming(
                model,
                root,
            )

            self.assertEqual(missing, [])
            self.assertEqual(mismatched, [])
            self.assertEqual(unexpected, ["unexpected"])
            self.assertTrue(torch.equal(model.weight, expected_weight))
            self.assertTrue(torch.equal(model.bias, expected_bias))

    def test_pagefile_error_detection(self) -> None:
        self.assertTrue(
            is_pagefile_or_memory_error(
                OSError(
                    "ページング ファイルが小さすぎるため、この操作を完了できません。 (os error 1455)"
                )
            )
        )
        self.assertTrue(is_pagefile_or_memory_error(MemoryError()))
        self.assertFalse(is_pagefile_or_memory_error(RuntimeError("other")))

    def test_classifier_loader_falls_back_to_streaming_after_memory_error(self) -> None:
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected_weight = torch.ones((2, 2), dtype=torch.float32)
            expected_bias = torch.arange(2, dtype=torch.float32)
            save_file(
                {
                    "weight": expected_weight,
                    "bias": expected_bias,
                },
                root / "model.safetensors",
            )

            model = load_wav2vec2_classifier(
                TinyModel,
                TinyConfigLoader,
                root,
                labels=("a", "e"),
                label2id={"a": 0, "e": 1},
                id2label={0: "a", 1: "e"},
            )

            self.assertTrue(torch.equal(model.weight, expected_weight))
            self.assertTrue(torch.equal(model.bias, expected_bias))
            self.assertEqual(model.config.num_labels, 2)

    def test_determine_optimal_num_workers(self) -> None:
        # 手動指定がある場合はそれを優先する
        self.assertEqual(determine_optimal_num_workers(2), 2)
        self.assertEqual(determine_optimal_num_workers(0), 0)

        # 未指定 (None) の場合、1以上の有効な数値が自動計算される
        auto_workers = determine_optimal_num_workers(None)
        self.assertGreaterEqual(auto_workers, 1)
        self.assertLessEqual(auto_workers, 8)


if __name__ == "__main__":
    unittest.main()
