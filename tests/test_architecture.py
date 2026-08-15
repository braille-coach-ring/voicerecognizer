import unittest
from unittest.mock import MagicMock, patch

from voicerecognizer.core.factory.recognizer_factory import RecognizerFactory
from voicerecognizer.core.interfaces import RecognitionStrategy
from voicerecognizer.core.services.voice_recognizer import VoiceRecognizer


class FakeRecognitionStrategy(RecognitionStrategy):
    def recognize(self, audio: str) -> str:
        return f"recognized:{audio}"


class ArchitectureTest(unittest.TestCase):
    def test_voice_recognizer_depends_on_strategy(self) -> None:
        recognizer = VoiceRecognizer(FakeRecognitionStrategy())

        self.assertEqual(recognizer.recognize("audio"), "recognized:audio")

    def test_factory_exposes_supported_strategy_names(self) -> None:
        self.assertEqual(
            RecognizerFactory.available_strategies(),
            ("cnn", "wav2vec2", "whisper"),
        )

    @patch("voicerecognizer.core.factory.recognizer_factory.CNNRecognizer")
    @patch("voicerecognizer.core.factory.recognizer_factory.Wav2Vec2Recognizer")
    def test_factory_use_last_and_custom_model_path(
        self, mock_w2v_class: MagicMock, mock_cnn_class: MagicMock
    ) -> None:
        RecognizerFactory.create("cnn", use_last=True)
        _, kwargs_cnn = mock_cnn_class.call_args
        self.assertTrue(str(kwargs_cnn["model_path"]).endswith("last_model.pth"))

        RecognizerFactory.create("wav2vec2", use_last=True)
        _, kwargs_w2v = mock_w2v_class.call_args
        self.assertTrue(str(kwargs_w2v["model_path"]).endswith("wav2vec2_last"))

        RecognizerFactory.create("cnn", model_path="custom/path.pth")
        _, kwargs_custom = mock_cnn_class.call_args
        self.assertEqual(str(kwargs_custom["model_path"]).replace("\\", "/"), "custom/path.pth")


if __name__ == "__main__":
    unittest.main()
