import unittest

import numpy as np

from config import PreprocessConfig
from runtime.vad import VoiceActivityDetector


class TestVoiceActivityDetector(unittest.TestCase):
    def test_requires_consecutive_chunks_before_speech(self):
        cfg = PreprocessConfig(vad_silence_threshold=0.005, vad_min_speech_chunks=2)
        vad = VoiceActivityDetector(config=cfg)

        speech_like = np.full(1600, 0.006, dtype=np.float32)

        self.assertFalse(vad.is_speech(speech_like))
        self.assertTrue(vad.is_speech(speech_like))

    def test_streak_resets_on_silence(self):
        cfg = PreprocessConfig(vad_silence_threshold=0.005, vad_min_speech_chunks=2)
        vad = VoiceActivityDetector(config=cfg)

        speech_like = np.full(1600, 0.006, dtype=np.float32)
        silence = np.zeros(1600, dtype=np.float32)

        self.assertFalse(vad.is_speech(speech_like))
        self.assertFalse(vad.is_speech(silence))
        self.assertFalse(vad.is_speech(speech_like))
        self.assertTrue(vad.is_speech(speech_like))


if __name__ == "__main__":
    unittest.main()
