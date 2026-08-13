import asyncio
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from recognizers.cnn_recognizer import CNNRecognizer


class TestPackageAPI(unittest.TestCase):
    def test_cnn_recognizer_default_instantiation(self) -> None:
        """CNNRecognizer should be constructable without arguments (defaults from config)."""
        # This should not raise TypeError about missing required arguments
        recognizer = CNNRecognizer()
        self.assertIsInstance(recognizer, CNNRecognizer)

    @patch.object(CNNRecognizer, "recognize")
    def test_recognize_cnn_via_recognizer(self, mock_recognize: MagicMock) -> None:
        mock_recognize.return_value = "あ"
        dummy_audio = np.zeros(16000, dtype=np.float32)

        recognizer = CNNRecognizer()
        result = recognizer.recognize(dummy_audio)
        self.assertEqual(result, "あ")

    @patch.object(CNNRecognizer, "recognize_async")
    def test_recognize_async_cnn_via_recognizer(self, mock_recognize_async: MagicMock) -> None:
        async def mock_async_fn(audio: np.ndarray) -> str:
            return "い"

        mock_recognize_async.side_effect = mock_async_fn
        dummy_audio = np.zeros(16000, dtype=np.float32)

        recognizer = CNNRecognizer()
        result = asyncio.run(recognizer.recognize_async(dummy_audio))
        self.assertEqual(result, "い")


if __name__ == "__main__":
    unittest.main()
