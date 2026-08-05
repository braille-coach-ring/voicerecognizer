import unittest
import numpy as np
from config import PreprocessConfig
from runtime.vad import VoiceActivityDetector


class TestVoiceActivityDetector(unittest.TestCase):
    def setUp(self):
        self.config = PreprocessConfig()  # Uses defaults (vad_silence_threshold=0.03, vad_rms_threshold=0.008)
        self.vad = VoiceActivityDetector(config=self.config)

    def test_none_or_empty_audio(self):
        self.assertFalse(self.vad.is_speech(None))
        self.assertFalse(self.vad.is_speech(np.array([], dtype=np.float32)))

    def test_silence(self):
        silence = np.zeros(16000, dtype=np.float32)
        self.assertFalse(self.vad.is_speech(silence))

        low_noise = np.random.normal(0, 0.001, 16000).astype(np.float32)
        self.assertFalse(self.vad.is_speech(low_noise))

    def test_environment_noise(self):
        # ログで観察された環境雑音 (Peak=0.0134, RMS=0.0022)
        # Peakは0.03未満、RMSも0.008未満のため音声と判定されないこと
        env_noise = np.random.normal(0, 0.0022, 16000).astype(np.float32)
        self.assertFalse(self.vad.is_speech(env_noise))

    def test_spike_noise(self):
        # 1サンプルのスパイクノイズ (Peak=0.05 だが RMSが極めて小さい)
        audio = np.zeros(16000, dtype=np.float32)
        audio[100] = 0.05  # スパイク
        # Peakは0.05 (>=0.03) だが RMS は sqrt(0.05^2 / 16000) = 0.000395 (< 0.008)
        self.assertFalse(self.vad.is_speech(audio))

    def test_speech_signal(self):
        # 継続的な音声シグナル (正弦波: Peak=0.1, RMS=0.1/sqrt(2) = ~0.07)
        t = np.linspace(0, 1, 16000, endpoint=False)
        speech = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        self.assertTrue(self.vad.is_speech(speech))


if __name__ == "__main__":
    unittest.main()
