import unittest

import numpy as np

from voicerecognizer.preprocessing.audio_preprocessor import AudioPreprocessor


class TestAudioPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = AudioPreprocessor()

    def test_smooth_fade_and_rms_normalization(self):
        # 100Hz Sine wave simulating low-frequency vowel "o"
        sr = 16000
        t = np.linspace(0, 0.4, int(sr * 0.4), endpoint=False)
        audio = (0.05 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)  # Soft "o" sound

        processed = self.preprocessor.preprocess_waveform(audio)

        # Output shape should match target length (0.6s = 9600 samples)
        self.assertEqual(len(processed), 9600)

        # RMS-based normalization should boost soft "o" sound to target level (~0.12 RMS or near 0.2-0.5 peak)
        rms = np.sqrt(np.mean(processed**2))
        self.assertGreater(rms, 0.05)

        # Peak amplitude bounded within 0.95
        peak = np.max(np.abs(processed))
        self.assertLessEqual(peak, 0.95 + 1e-4)

        # Smooth fade-in at the beginning
        self.assertLess(abs(processed[0]), 0.05)

    def test_long_audio_truncation_has_smooth_fadeout(self):
        # 1.5s long audio exceeding target 0.6s
        sr = 16000
        t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
        audio = (0.8 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)

        processed = self.preprocessor.preprocess_waveform(audio)

        self.assertEqual(len(processed), 9600)
        # End of truncated audio should fade out smoothly to 0 without hard cut
        self.assertAlmostEqual(processed[-1], 0.0, places=3)

    def test_dynamic_trimming_without_padding(self):
        # 0.25s short audio
        sr = 16000
        t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False)
        audio = (0.3 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)

        # Preprocess with pad_to_target=False
        processed_dynamic = self.preprocessor.preprocess_waveform(audio, pad_to_target=False)

        # Should be shorter than target 0.6s (9600 samples)
        self.assertLess(len(processed_dynamic), 9600)
        # Should be at least minimum length (0.2s = 3200 samples)
        self.assertGreaterEqual(len(processed_dynamic), 3200)

    def test_dynamic_trimming_min_length_guarantee(self):
        # Very short 0.05s burst
        sr = 16000
        t = np.linspace(0, 0.05, int(sr * 0.05), endpoint=False)
        audio = (0.3 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)

        processed = self.preprocessor.preprocess_waveform(
            audio, pad_to_target=False, min_length_seconds=0.2
        )

        # Should guarantee minimum 0.2s (3200 samples)
        self.assertEqual(len(processed), 3200)

    def test_late_onset_is_aligned_to_head(self):
        # マイク推論時の 1 秒窓のように、発話が 0.5s 地点から始まるケース
        sr = 16000
        rng = np.random.default_rng(0)
        audio = (rng.standard_normal(sr) * 0.001).astype(np.float32)
        onset = int(sr * 0.5)
        t = np.arange(sr - onset) / sr
        audio[onset:] += (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

        processed = self.preprocessor.preprocess_waveform(audio)

        self.assertEqual(len(processed), 9600)
        # 発話は先頭 0.6s 内に収まり、開始位置は pre-roll (0.1s) 付近になる
        active = np.flatnonzero(np.abs(processed) > 0.1)
        self.assertGreater(len(active), 0)
        self.assertAlmostEqual(active[0] / sr, 0.1, delta=0.02)
        self.assertAlmostEqual(self.preprocessor.last_stats["onset_ms"], 500.0, delta=20.0)

    def test_early_onset_is_unchanged(self):
        # 学習データのように発話開始が pre-roll より手前なら切り出し位置は変わらない
        sr = 16000
        audio = np.zeros(int(sr * 0.5), dtype=np.float32)
        onset = int(sr * 0.05)
        t = np.arange(len(audio) - onset) / sr
        audio[onset:] = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

        aligned = self.preprocessor.preprocess_waveform(audio)
        unaligned = AudioPreprocessor(onset_pre_roll_seconds=None).preprocess_waveform(audio)

        np.testing.assert_allclose(aligned, unaligned)


if __name__ == "__main__":
    unittest.main()
