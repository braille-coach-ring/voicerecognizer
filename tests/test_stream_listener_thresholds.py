import unittest
from unittest.mock import MagicMock, patch

from voicerecognizer.runtime.stream_listener import AudioStreamListener


class TestStreamListenerThresholds(unittest.TestCase):
    @patch("voicerecognizer.recognizers.wav2vec2_recognizer.Wav2Vec2Recognizer")
    def test_listener_thresholds_initialization(self, mock_rec_cls: MagicMock) -> None:
        mock_rec = MagicMock()
        mock_rec_cls.return_value = mock_rec

        listener = AudioStreamListener(
            recognizer=mock_rec,
            min_confidence=0.25,
            silence_threshold=0.08,
            rms_threshold=0.015,
            speech_settle_seconds=0.5,
        )

        self.assertEqual(listener.min_confidence, 0.25)
        self.assertEqual(listener.vad.silence_threshold, 0.08)
        self.assertEqual(listener.vad.rms_threshold, 0.015)
        self.assertEqual(listener.speech_settle_seconds, 0.5)

    @patch("voicerecognizer.recognizers.wav2vec2_recognizer.Wav2Vec2Recognizer")
    def test_listener_dynamic_set_thresholds(self, mock_rec_cls: MagicMock) -> None:
        mock_rec = MagicMock()
        mock_rec_cls.return_value = mock_rec

        listener = AudioStreamListener(
            recognizer=mock_rec,
            min_confidence=0.35,
        )

        # 初期値の確認
        self.assertEqual(listener.min_confidence, 0.35)

        # 動的更新
        listener.set_thresholds(
            min_confidence=0.12,
            silence_threshold=0.05,
            rms_threshold=0.02,
            speech_settle_seconds=0.6,
        )

        self.assertEqual(listener.min_confidence, 0.12)
        self.assertEqual(listener.vad.silence_threshold, 0.05)
        self.assertEqual(listener.vad.rms_threshold, 0.02)
        self.assertEqual(listener.speech_settle_seconds, 0.6)


if __name__ == "__main__":
    unittest.main()
