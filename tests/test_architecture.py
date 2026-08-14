import unittest

from voicerecognizer.core.factory.recognizer_factory import RecognizerFactory
from voicerecognizer.core.interfaces import RecognitionStrategy
from voicerecognizer.core.services.voice_recognizer import VoiceRecognizer
from voicerecognizer.recognizers.cnn_recognizer import CNNRecognizer
from voicerecognizer.recognizers.wav2vec2_recognizer import Wav2Vec2Recognizer


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
            ("cnn", "wav2vec2"),
        )

    def test_recognizers_implement_strategy(self):
        self.assertIsInstance(Wav2Vec2Recognizer(auto_download=False), RecognitionStrategy)
        self.assertIsInstance(CNNRecognizer(auto_download=False), RecognitionStrategy)


if __name__ == "__main__":
    unittest.main()
