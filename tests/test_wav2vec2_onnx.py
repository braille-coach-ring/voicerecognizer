import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from voicerecognizer.config import DEFAULT_RECOGNITION_CONFIG
from voicerecognizer.recognizers.wav2vec2_recognizer import Wav2Vec2Recognizer


class TestWav2Vec2ONNXRecognizer(unittest.TestCase):
    def test_missing_onnx_model_raises_file_not_found(self):
        non_existent_dir = Path("weights/non_existent_dir_for_testing")
        recognizer = Wav2Vec2Recognizer(model_path=non_existent_dir)
        with self.assertRaises(FileNotFoundError) as ctx:
            recognizer.recognize(np.zeros(16000, dtype=np.float32))
        self.assertIn("Wav2Vec2 ONNX モデル", str(ctx.exception))

    @patch("onnxruntime.InferenceSession")
    @patch("transformers.AutoFeatureExtractor.from_pretrained")
    def test_recognize_success_with_mock_onnx(self, mock_feature_extractor, mock_onnx_session):
        # Feature extractor mock
        mock_fe = MagicMock()
        mock_fe.return_value = {"input_values": np.zeros((1, 16000), dtype=np.float32)}
        mock_feature_extractor.return_value = mock_fe

        # ONNX Session mock
        mock_sess = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input_values"
        mock_sess.get_inputs.return_value = [mock_input]
        # Simulate 28 output logits where index 0 has highest score
        logits = np.zeros((1, 28), dtype=np.float32)
        logits[0, 0] = 5.0
        mock_sess.run.return_value = [logits]
        mock_onnx_session.return_value = mock_sess

        # Existing model dir with model_int8.onnx mock
        model_dir = DEFAULT_RECOGNITION_CONFIG.wav2vec2_best_model_dir
        with patch.object(Path, "exists", return_value=True):
            recognizer = Wav2Vec2Recognizer(model_path=model_dir, labels=["a", "i", "u"])
            result = recognizer.recognize(np.zeros(16000, dtype=np.float32))
            self.assertEqual(result, "a")
            self.assertIsNotNone(recognizer.last_confidence)
            assert recognizer.last_confidence is not None
            self.assertGreater(recognizer.last_confidence, 0.5)


if __name__ == "__main__":
    unittest.main()
