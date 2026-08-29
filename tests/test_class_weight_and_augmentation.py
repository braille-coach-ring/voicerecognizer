"""
Unit tests for AudioAugmentor, compute_class_weights, and CLI option defaults using unittest.
"""

import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from voicerecognizer.models.wav2vec2.train import (
    AugmentedSubset,
    build_parser,
    compute_balanced_sampler_weights,
    compute_class_weights,
    load_confusion_label_multipliers,
)
from voicerecognizer.preprocessing.audio_augmentor import AudioAugmentor


class TestClassWeightAndAugmentation(unittest.TestCase):
    def test_audio_augmentor_shapes_and_values(self) -> None:
        augmentor = AudioAugmentor(seed=42)
        sample_rate = 16000
        duration = 0.6
        waveform = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        aug_waveform = augmentor.augment(waveform)

        self.assertIsInstance(aug_waveform, np.ndarray)
        self.assertEqual(aug_waveform.dtype, np.float32)
        self.assertEqual(aug_waveform.shape, waveform.shape)
        self.assertFalse(np.isnan(aug_waveform).any())
        self.assertFalse(np.isinf(aug_waveform).any())

    def test_audio_augmentor_keeps_shape_with_speed_pitch_and_device_noise(self) -> None:
        import soundfile as sf

        sample_rate = 16000
        waveform = np.sin(np.linspace(0, 440 * 2 * np.pi, sample_rate // 2)).astype(np.float32)
        noise = np.random.default_rng(1).normal(0, 0.05, sample_rate).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            noise_path = f"{tmp_dir}/device_noise.wav"
            sf.write(noise_path, noise, sample_rate)
            augmentor = AudioAugmentor(
                noise_level=0.0,
                gain_range=(1.0, 1.0),
                shift_max_ratio=0.0,
                speed_range=(1.05, 1.05),
                pitch_shift_steps=(0.25, 0.25),
                noise_file_paths=[noise_path],
                sample_rate=sample_rate,
                p=1.0,
                seed=2,
            )
            aug_waveform = augmentor.augment(waveform)

        self.assertEqual(aug_waveform.dtype, np.float32)
        self.assertEqual(aug_waveform.shape, waveform.shape)
        self.assertFalse(np.isnan(aug_waveform).any())
        self.assertFalse(np.isinf(aug_waveform).any())

    def test_augmented_subset_applies_augmentation_when_loaded(self) -> None:
        class TinyDataset(Dataset):
            def __len__(self) -> int:
                return 2

            def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
                return np.zeros(4, dtype=np.float32), index

        subset = Subset(TinyDataset(), [0, 1])
        train_dataset = AugmentedSubset(subset, augmentor=AudioAugmentor(seed=1))

        with patch.object(AudioAugmentor, "augment", return_value=np.ones(4, dtype=np.float32)):
            loader = DataLoader(train_dataset, batch_size=1)
            waveform, _ = next(iter(loader))

        self.assertTrue(torch.all(waveform == 1.0))

    def test_compute_class_weights_balance(self) -> None:
        labels = [0] * 90 + [1] * 10
        num_classes = 2
        weights = compute_class_weights(labels, num_classes=num_classes, power=0.5)

        self.assertIsInstance(weights, torch.Tensor)
        self.assertEqual(weights.shape, (2,))
        self.assertGreater(weights[1].item(), weights[0].item())
        self.assertAlmostEqual(weights.mean().item(), 1.0, places=4)

    def test_confusion_pair_sampler_boosts_labels_from_confusion_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            evaluation_result = {
                "confusion_matrix": {
                    "shu": {"shu": 20, "chu": 6, "a": 0},
                    "chu": {"shu": 1, "chu": 20, "a": 0},
                    "a": {"shu": 0, "chu": 0, "a": 20},
                }
            }
            result_path = f"{tmp_dir}/evaluation_result.json"
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(evaluation_result, f)

            multipliers = load_confusion_label_multipliers(
                result_path,
                ["shu", "chu", "a"],
                min_count=3,
                boost=0.5,
            )

        self.assertGreater(multipliers[0], 1.0)
        self.assertGreater(multipliers[1], 1.0)
        self.assertNotIn(2, multipliers)

        sample_weights = compute_balanced_sampler_weights(
            [0, 1, 2],
            num_classes=3,
            power=0.0,
            confusion_label_multipliers=multipliers,
        )
        self.assertGreater(sample_weights[0], sample_weights[2])
        self.assertGreater(sample_weights[1], sample_weights[2])

    def test_cli_parser_defaults(self) -> None:
        parser = build_parser()
        args_default = parser.parse_args([])

        self.assertTrue(args_default.use_class_weights)
        self.assertTrue(args_default.augment)
        self.assertIsNone(args_default.augmentation_noise_dir)
        self.assertTrue(args_default.use_balanced_sampler)
        self.assertFalse(args_default.speaker_aware_split)
        self.assertTrue(args_default.use_confusion_pair_sampler)
        self.assertEqual(args_default.class_weight_power, 0.5)
        self.assertEqual(args_default.confusion_pair_min_count, 3)
        self.assertEqual(args_default.confusion_pair_boost, 0.5)

        args_opt_out = parser.parse_args(
            [
                "--no-class-weights",
                "--no-augment",
                "--no-balanced-sampler",
                "--no-confusion-pair-sampler",
                "--speaker-aware-split",
                "--class-weight-power",
                "1.0",
            ]
        )
        self.assertFalse(args_opt_out.use_class_weights)
        self.assertFalse(args_opt_out.augment)
        self.assertFalse(args_opt_out.use_balanced_sampler)
        self.assertTrue(args_opt_out.speaker_aware_split)
        self.assertFalse(args_opt_out.use_confusion_pair_sampler)
        self.assertEqual(args_opt_out.class_weight_power, 1.0)


if __name__ == "__main__":
    unittest.main()
