import unittest

import numpy as np

from config import PreprocessConfig
from runtime.vad import VoiceActivityDetector


class TestVoiceActivityDetector(unittest.TestCase):
    def test_requires_consecutive_chunks_before_speech(self):
        cfg = PreprocessConfig(
            vad_silence_threshold=0.005,
            vad_min_speech_chunks=2,
            vad_startup_ignore_chunks=0,
        )
        vad = VoiceActivityDetector(config=cfg)

        speech_like = np.full(1600, 0.006, dtype=np.float32)

        self.assertFalse(vad.is_speech(speech_like))
        self.assertTrue(vad.is_speech(speech_like))

    def test_streak_resets_on_silence(self):
        cfg = PreprocessConfig(
            vad_silence_threshold=0.005,
            vad_min_speech_chunks=2,
            vad_startup_ignore_chunks=0,
        )
        vad = VoiceActivityDetector(config=cfg)

        speech_like = np.full(1600, 0.006, dtype=np.float32)
        silence = np.zeros(1600, dtype=np.float32)

        self.assertFalse(vad.is_speech(speech_like))
        self.assertFalse(vad.is_speech(silence))
        self.assertFalse(vad.is_speech(speech_like))
        self.assertTrue(vad.is_speech(speech_like))

    def test_ignores_single_peak_noise(self):
        cfg = PreprocessConfig(
            vad_silence_threshold=0.005,
            vad_min_speech_chunks=1,
            vad_startup_ignore_chunks=0,
            vad_min_active_ratio=0.02,
        )
        vad = VoiceActivityDetector(config=cfg)

        impulse_like = np.zeros(1600, dtype=np.float32)
        impulse_like[800] = 0.2

        self.assertFalse(vad.is_speech(impulse_like))

    def test_accepts_when_active_ratio_is_enough(self):
        cfg = PreprocessConfig(
            vad_silence_threshold=0.005,
            vad_min_speech_chunks=1,
            vad_startup_ignore_chunks=0,
            vad_min_active_ratio=0.02,
        )
        vad = VoiceActivityDetector(config=cfg)

        speech_like = np.zeros(1600, dtype=np.float32)
        speech_like[:100] = 0.01

        self.assertTrue(vad.is_speech(speech_like))

    def test_ignores_first_chunks_as_warmup(self):
        cfg = PreprocessConfig(
            vad_silence_threshold=0.005,
            vad_min_speech_chunks=1,
            vad_startup_ignore_chunks=3,
            vad_min_active_ratio=0.0,
        )
        vad = VoiceActivityDetector(config=cfg)
        speech_like = np.full(1600, 0.01, dtype=np.float32)

        self.assertFalse(vad.is_speech(speech_like))
        self.assertFalse(vad.is_speech(speech_like))
        self.assertFalse(vad.is_speech(speech_like))
        self.assertTrue(vad.is_speech(speech_like))


if __name__ == "__main__":
    unittest.main()
