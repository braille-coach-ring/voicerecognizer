"""
Unit tests for AudioAugmentor, compute_class_weights, and CLI option defaults using unittest.
"""

import json
import tempfile
import unittest
from typing import override
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from voicerecognizer.models.wav2vec2.train import (
    AugmentedSubset,
    build_parser,
    build_wav2vec2_optimizer,
    compute_balanced_sampler_weights,
    compute_class_weights,
    load_confusion_label_multipliers,
    resolve_training_settings,
    seed_dataloader_worker,
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
        class TinyDataset(Dataset[tuple[np.ndarray, int]]):
            def __len__(self) -> int:
                return 2

            @override
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
        self.assertTrue(args_default.hf_upload)
        self.assertTrue(args_default.from_scratch_auto_tune)
        self.assertEqual(args_default.class_weight_power, 0.5)
        self.assertEqual(args_default.confusion_pair_min_count, 3)
        self.assertEqual(args_default.confusion_pair_boost, 0.5)

        args_opt_out = parser.parse_args(
            [
                "--no-class-weights",
                "--no-augment",
                "--no-balanced-sampler",
                "--no-confusion-pair-sampler",
                "--no-hf-upload",
                "--no-from-scratch-auto-tune",
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
        self.assertFalse(args_opt_out.hf_upload)
        self.assertFalse(args_opt_out.from_scratch_auto_tune)
        self.assertEqual(args_opt_out.class_weight_power, 1.0)

    def test_from_scratch_auto_tune_changes_only_from_scratch_settings(self) -> None:
        parser = build_parser()

        default_settings = resolve_training_settings(parser.parse_args([]))
        self.assertFalse(default_settings.from_scratch_auto_tuned)
        self.assertEqual(default_settings.learning_rate, 3e-5)
        self.assertEqual(default_settings.freeze_transformer_layers, 10)
        self.assertEqual(default_settings.patience, 5)
        self.assertEqual(default_settings.head_lr_multiplier, 1.0)
        self.assertEqual(default_settings.early_stopping_scope, "global_best")

        from_scratch_settings = resolve_training_settings(parser.parse_args(["--from-scratch"]))
        self.assertTrue(from_scratch_settings.from_scratch_auto_tuned)
        self.assertEqual(from_scratch_settings.learning_rate, 5e-5)
        self.assertEqual(from_scratch_settings.freeze_transformer_layers, 6)
        self.assertEqual(from_scratch_settings.patience, 10)
        self.assertEqual(from_scratch_settings.head_lr_multiplier, 10.0)
        self.assertEqual(from_scratch_settings.early_stopping_scope, "run_best")

        old_style_from_scratch_settings = resolve_training_settings(
            parser.parse_args(["--from-scratch", "--no-from-scratch-auto-tune"])
        )
        self.assertFalse(old_style_from_scratch_settings.from_scratch_auto_tuned)
        self.assertEqual(old_style_from_scratch_settings.learning_rate, 3e-5)
        self.assertEqual(old_style_from_scratch_settings.freeze_transformer_layers, 10)
        self.assertEqual(old_style_from_scratch_settings.patience, 5)
        self.assertEqual(old_style_from_scratch_settings.early_stopping_scope, "global_best")

        manual_from_scratch_settings = resolve_training_settings(
            parser.parse_args(
                [
                    "--from-scratch",
                    "--learning-rate",
                    "0.0001",
                    "--freeze-transformer-layers",
                    "2",
                    "--patience",
                    "3",
                    "--from-scratch-head-lr-multiplier",
                    "2.0",
                ]
            )
        )
        self.assertEqual(manual_from_scratch_settings.learning_rate, 0.0001)
        self.assertEqual(manual_from_scratch_settings.freeze_transformer_layers, 2)
        self.assertEqual(manual_from_scratch_settings.patience, 3)
        self.assertEqual(manual_from_scratch_settings.head_lr_multiplier, 2.0)
        self.assertEqual(manual_from_scratch_settings.early_stopping_scope, "run_best")

    def test_from_scratch_optimizer_boosts_projector_and_classifier_lr(self) -> None:
        class TinyClassifier(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.wav2vec2 = torch.nn.Linear(2, 2)
                self.projector = torch.nn.Linear(2, 2)
                self.classifier = torch.nn.Linear(2, 2)

        model = TinyClassifier()

        optimizer = build_wav2vec2_optimizer(
            model,
            learning_rate=5e-5,
            weight_decay=0.01,
            head_lr_multiplier=10.0,
        )

        group_lrs = sorted(group["lr"] for group in optimizer.param_groups)
        self.assertEqual(group_lrs, [5e-5, 5e-4])


class TestAugmentationRandomness(unittest.TestCase):
    """データ拡張の乱数がシードで再現でき、ワーカー間で独立していることを保証する。"""

    def _waveform(self) -> np.ndarray:
        return np.sin(np.linspace(0, 8 * np.pi, 1600, dtype=np.float32)).astype(np.float32)

    def test_same_seed_reproduces_augmentation(self) -> None:
        waveform = self._waveform()
        first = AudioAugmentor(seed=123).augment(waveform)
        second = AudioAugmentor(seed=123).augment(waveform)
        np.testing.assert_allclose(first, second)

    def test_different_seed_changes_augmentation(self) -> None:
        waveform = self._waveform()
        first = AudioAugmentor(seed=123).augment(waveform)
        second = AudioAugmentor(seed=456).augment(waveform)
        self.assertFalse(np.allclose(first, second))

    def _run_worker_init(self, augmentor: AudioAugmentor, worker_seed: int) -> None:
        """ワーカープロセス内での worker_init_fn 実行を再現する"""
        fake_worker_info = type(
            "WorkerInfo", (), {"dataset": AugmentedSubset.__new__(AugmentedSubset)}
        )()
        fake_worker_info.dataset.augmentor = augmentor
        with (
            patch(
                "voicerecognizer.models.wav2vec2.train.torch.initial_seed", return_value=worker_seed
            ),
            patch(
                "voicerecognizer.models.wav2vec2.train.get_worker_info",
                return_value=fake_worker_info,
            ),
        ):
            seed_dataloader_worker(0)

    def test_worker_init_gives_each_worker_an_independent_stream(self) -> None:
        """同一シードから fork された 2 ワーカーが同じ拡張列を引かないこと"""
        waveform = self._waveform()

        worker_a = AudioAugmentor(seed=42)
        worker_b = AudioAugmentor(seed=42)
        # worker_init_fn を通さない場合は完全に同じ列になる（修正前の挙動）
        np.testing.assert_allclose(worker_a.augment(waveform), worker_b.augment(waveform))

        worker_a = AudioAugmentor(seed=42)
        worker_b = AudioAugmentor(seed=42)
        self._run_worker_init(worker_a, worker_seed=1000)
        self._run_worker_init(worker_b, worker_seed=1001)
        self.assertFalse(
            np.allclose(worker_a.augment(waveform), worker_b.augment(waveform)),
            "worker_init_fn 適用後もワーカー間で拡張が一致している",
        )

    def test_worker_init_is_deterministic_for_a_given_worker_seed(self) -> None:
        waveform = self._waveform()

        first = AudioAugmentor(seed=42)
        self._run_worker_init(first, worker_seed=2024)
        second = AudioAugmentor(seed=99)
        self._run_worker_init(second, worker_seed=2024)

        np.testing.assert_allclose(first.augment(waveform), second.augment(waveform))

    def test_worker_init_without_worker_info_is_a_noop(self) -> None:
        with (
            patch("voicerecognizer.models.wav2vec2.train.torch.initial_seed", return_value=7),
            patch("voicerecognizer.models.wav2vec2.train.get_worker_info", return_value=None),
        ):
            seed_dataloader_worker(0)


if __name__ == "__main__":
    unittest.main()
