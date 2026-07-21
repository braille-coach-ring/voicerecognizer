import unittest

from core.factory.recognizer_factory import RecognizerFactory
from core.interfaces import RecognitionStrategy
from core.services.voice_recognizer import VoiceRecognizer
from recognizers.wav2vec2_recognizer import Wav2Vec2Recognizer
from recognizers.whisper_recognizer import WhisperRecognizer


class FakeRecognitionStrategy(RecognitionStrategy):
    def recognize(self, audio) -> str:
        return f"recognized:{audio}"


class ArchitectureTest(unittest.TestCase):
    def test_voice_recognizer_depends_on_strategy(self):
        recognizer = VoiceRecognizer(FakeRecognitionStrategy())

        self.assertEqual(recognizer.recognize("audio"), "recognized:audio")

    def test_factory_exposes_supported_strategy_names(self):
        self.assertEqual(
            RecognizerFactory.available_strategies(),
            ("cnn", "wav2vec2", "whisper"),
        )

    def test_future_recognizers_implement_strategy(self):
        self.assertIsInstance(Wav2Vec2Recognizer(), RecognitionStrategy)
        self.assertIsInstance(WhisperRecognizer(), RecognitionStrategy)


if __name__ == "__main__":
    unittest.main()
