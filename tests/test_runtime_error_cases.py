"""
Runtime Critical Error Cases & Protection Tests (tests/test_runtime_error_cases.py)

目的:
  実行時 (Runtime) に AttributeError, TypeError, NameError などの致命的なクラッシュを引き起こす
  型不整合や未初期化状態 (None 判定漏れ) のバグを確実に検出・保護するための失敗テスト群。
"""

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from voicerecognizer.core.exceptions import ModelNotFoundError, VoiceRecognizerError
from voicerecognizer.evaluation.evaluator import Evaluator
from voicerecognizer.recognizers.wav2vec2_recognizer import Wav2Vec2Recognizer


class TestWav2Vec2RecognizerRuntimeProtections(unittest.TestCase):
    """Wav2Vec2Recognizer における Session/FeatureExtractor が None の場合の実行時保護テスト"""

    def setUp(self) -> None:
        self.recognizer = Wav2Vec2Recognizer.__new__(Wav2Vec2Recognizer)
        self.recognizer.model_path = Path("fake_path")
        self.recognizer.sample_rate = 16000
        self.recognizer.session = None  # ort_session is Uninitialized / None
        self.recognizer.input_name = "input_values"
        self.recognizer.feature_extractor = None  # feature_extractor is Uninitialized / None
        self.recognizer.audio_preprocessor = MagicMock()
        self.recognizer.audio_preprocessor.preprocess_waveform.side_effect = lambda a, **kwargs: a
        self.recognizer.dynamic_trimming = False

    def test_prepare_input_raises_model_not_found_when_feature_extractor_is_none(self) -> None:
        """feature_extractorがNoneの場合、TypeErrorでクラッシュせずModelNotFoundErrorがスローされるべき"""
        dummy_audio = np.zeros(16000, dtype=np.float32)
        with (
            patch.object(self.recognizer, "_ensure_model_loaded", return_value=None),
            self.assertRaises((ModelNotFoundError, VoiceRecognizerError)),
        ):
            self.recognizer.recognize(dummy_audio)

    def test_recognize_raises_model_not_found_when_session_is_none(self) -> None:
        """sessionがNoneの場合、AttributeErrorでクラッシュせずModelNotFoundErrorがスローされるべき"""
        dummy_audio = np.zeros(16000, dtype=np.float32)
        # Give a fake feature_extractor so input prep succeeds, but session.run fails
        self.recognizer.feature_extractor = MagicMock(return_value={"input_values": dummy_audio})
        with (
            patch.object(self.recognizer, "_ensure_model_loaded", return_value=None),
            self.assertRaises((ModelNotFoundError, VoiceRecognizerError)),
        ):
            self.recognizer.recognize(dummy_audio)

    def test_recognize_with_candidates_raises_model_not_found_when_session_is_none(self) -> None:
        """sessionがNoneの場合、recognize_with_candidatesも保護されるべき"""
        dummy_audio = np.zeros(16000, dtype=np.float32)
        self.recognizer.feature_extractor = MagicMock(return_value={"input_values": dummy_audio})
        with (
            patch.object(self.recognizer, "_ensure_model_loaded", return_value=None),
            self.assertRaises((ModelNotFoundError, VoiceRecognizerError)),
        ):
            self.recognizer.recognize_with_candidates(dummy_audio)

    def test_recognize_async_raises_model_not_found_when_session_is_none(self) -> None:
        """sessionがNoneの場合、非同期認識も保護されるべき"""
        dummy_audio = np.zeros(16000, dtype=np.float32)
        self.recognizer.feature_extractor = MagicMock(return_value={"input_values": dummy_audio})

        async def run_async() -> str:
            with patch.object(self.recognizer, "_ensure_model_loaded", return_value=None):
                return await self.recognizer.recognize_async(dummy_audio)

        with self.assertRaises((ModelNotFoundError, VoiceRecognizerError)):
            asyncio.run(run_async())


class TestEvaluatorLabelSafety(unittest.TestCase):
    """Evaluator における labels 引数の型安全性テスト"""

    def test_evaluator_accepts_sequence_labels_and_converts_to_tuple(self) -> None:
        """labels に list[str] などの Sequence が渡された場合、正常に受容し内部で tuple[str, ...] に安全に変換されるべき"""
        list_labels = ["a", "i", "u", "e", "o"]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            index_path = tmp_path / "index.csv"
            index_path.write_text("filepath,label\nfile1.wav,a\n", encoding="utf-8")

            evaluator = Evaluator(labels=list_labels, dataset_path=tmp_path)
            self.assertIsInstance(evaluator.labels, tuple)
            self.assertEqual(evaluator.labels, ("a", "i", "u", "e", "o"))


class TestONNXExportArgumentTypes(unittest.TestCase):
    """ONNX エクスポート関数における dummy_input の型安全性テスト"""

    @patch("torch.onnx.export")
    def test_export_onnx_passes_dummy_input_as_tuple(self, mock_export: MagicMock) -> None:
        """torch.onnx.export の args 引数には単体 Tensor ではなく (dummy_input,) のタプルが渡されるべき"""
        from voicerecognizer.models.wav2vec2.export_onnx import export_to_onnx

        dummy_model = MagicMock(spec=torch.nn.Module)
        dummy_model.eval = MagicMock()

        with tempfile.TemporaryDirectory() as tmp:
            output_file = Path(tmp) / "model.onnx"

            def mock_export_side_effect(model: torch.nn.Module, args: tuple[torch.Tensor, ...], f: str, **kwargs: object) -> None:
                Path(f).touch()

            mock_export.side_effect = mock_export_side_effect
            export_to_onnx(dummy_model, output_file)

            self.assertTrue(mock_export.called)
            call_args, _ = mock_export.call_args
            # 第二引数 (args) が tuple であることを検証
            passed_input = call_args[1]
            self.assertIsInstance(passed_input, tuple)

            self.assertTrue(mock_export.called)
            args, _ = mock_export.call_args
            # 第二引数 (args) が tuple であることを検証
            passed_input = args[1]
            self.assertIsInstance(passed_input, tuple)


if __name__ == "__main__":
    unittest.main()
