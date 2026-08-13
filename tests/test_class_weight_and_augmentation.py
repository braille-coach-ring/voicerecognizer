"""
Unit tests for AudioAugmentor, compute_class_weights, and CLI option defaults using unittest.
"""

import unittest

import numpy as np
import torch

from models.wav2vec2.train import build_parser, compute_class_weights
from preprocessing.audio_augmentor import AudioAugmentor


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

    def test_compute_class_weights_balance(self) -> None:
        labels = [0] * 90 + [1] * 10
        num_classes = 2
        weights = compute_class_weights(labels, num_classes=num_classes, power=0.5)

        self.assertIsInstance(weights, torch.Tensor)
        self.assertEqual(weights.shape, (2,))
        self.assertGreater(weights[1].item(), weights[0].item())
        self.assertAlmostEqual(weights.mean().item(), 1.0, places=4)

    def test_cli_parser_defaults(self) -> None:
        parser = build_parser()
        args_default = parser.parse_args([])

        self.assertTrue(args_default.use_class_weights)
        self.assertTrue(args_default.augment)
        self.assertTrue(args_default.use_balanced_sampler)
        self.assertEqual(args_default.class_weight_power, 0.5)

        args_opt_out = parser.parse_args(
            [
                "--no-class-weights",
                "--no-augment",
                "--no-balanced-sampler",
                "--class-weight-power",
                "1.0",
            ]
        )
        self.assertFalse(args_opt_out.use_class_weights)
        self.assertFalse(args_opt_out.augment)
        self.assertFalse(args_opt_out.use_balanced_sampler)
        self.assertEqual(args_opt_out.class_weight_power, 1.0)


if __name__ == "__main__":
    unittest.main()
