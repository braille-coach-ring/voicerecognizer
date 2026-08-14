import asyncio
import importlib
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

import voicerecognizer as vr
from voicerecognizer.core.exceptions import (
    AudioPreprocessingError,
    DeviceNotFoundError,
    ModelNotFoundError,
    VoiceRecognizerError,
)
from voicerecognizer.core.interfaces import RecognitionStrategy
from voicerecognizer.recognizers.cnn_recognizer import CNNRecognizer
from voicerecognizer.recognizers.wav2vec2_recognizer import Wav2Vec2Recognizer
from voicerecognizer.runtime.stream_listener import AudioStreamListener, RecognitionResult


class TestPackageAPI(unittest.TestCase):
    def test_public_api_exports(self) -> None:
        """Top-level package exports all required classes and exceptions explicitly."""
        self.assertEqual(vr.__version__, "0.1.0")

        # Classes & Types
        self.assertIs(vr.CNNRecognizer, CNNRecognizer)
        self.assertIs(vr.Wav2Vec2Recognizer, Wav2Vec2Recognizer)
        self.assertIs(vr.AudioStreamListener, AudioStreamListener)
        self.assertIs(vr.RecognitionResult, RecognitionResult)
        self.assertIs(vr.RecognitionStrategy, RecognitionStrategy)

        # Exceptions
        self.assertIs(vr.AudioPreprocessingError, AudioPreprocessingError)
        self.assertIs(vr.DeviceNotFoundError, DeviceNotFoundError)
        self.assertIs(vr.ModelNotFoundError, ModelNotFoundError)
        self.assertIs(vr.VoiceRecognizerError, VoiceRecognizerError)

    def test_cnn_recognizer_default_instantiation(self) -> None:
        """CNNRecognizer should be constructable without arguments."""
        recognizer = CNNRecognizer(auto_download=False)
        self.assertIsInstance(recognizer, CNNRecognizer)

    @patch.object(CNNRecognizer, "_load_model")
    @patch.object(CNNRecognizer, "recognize")
    def test_recognize_cnn_via_recognizer(
        self, mock_recognize: MagicMock, mock_load_model: MagicMock
    ) -> None:
        mock_recognize.return_value = "あ"
        dummy_audio = np.zeros(16000, dtype=np.float32)

        recognizer = CNNRecognizer()
        result = recognizer.recognize(dummy_audio)
        self.assertEqual(result, "あ")

    @patch.object(CNNRecognizer, "_load_model")
    @patch.object(CNNRecognizer, "recognize_async")
    def test_recognize_async_cnn_via_recognizer(
        self, mock_recognize_async: MagicMock, mock_load_model: MagicMock
    ) -> None:
        async def mock_async_fn(audio: np.ndarray) -> str:
            return "い"

        mock_recognize_async.side_effect = mock_async_fn
        dummy_audio = np.zeros(16000, dtype=np.float32)

        recognizer = CNNRecognizer()
        result = asyncio.run(recognizer.recognize_async(dummy_audio))
        self.assertEqual(result, "い")

    def test_all_project_packages_and_modules_dynamically(self) -> None:
        """
        ファイル構造から自動的にすべてのパッケージ・サブパッケージを抽出し、
        すべてのルーティングおよびインポートが正常に動作することを検証する。
        """
        project_root = Path(__file__).resolve().parent.parent
        src_dir = project_root / "src"

        discovered_packages: set[str] = set()

        # src 配下のパッケージ
        if src_dir.exists():
            for path in src_dir.rglob("__init__.py"):
                parent_rel = path.parent.relative_to(src_dir)
                mod_name = ".".join(parent_rel.parts)
                discovered_packages.add(mod_name)

        # ルート直下の補助パッケージ
        for extra in ["script", "tests"]:
            if (project_root / extra / "__init__.py").exists():
                discovered_packages.add(extra)

        self.assertGreater(len(discovered_packages), 0, "ファイル構造からパッケージが抽出されませんでした。")

        for pkg in sorted(discovered_packages):
            with self.subTest(package=pkg):
                mod = importlib.import_module(pkg)
                self.assertIsNotNone(mod, f"パッケージ '{pkg}' のインポートに失敗しました。")

        expected_packages = {
            "voicerecognizer",
            "voicerecognizer.core",
            "voicerecognizer.core.factory",
            "voicerecognizer.core.services",
            "voicerecognizer.dataset",
            "voicerecognizer.evaluation",
            "voicerecognizer.models",
            "voicerecognizer.models.cnn",
            "voicerecognizer.models.wav2vec2",
            "voicerecognizer.preprocessing",
            "voicerecognizer.recognizers",
            "voicerecognizer.runtime",
            "voicerecognizer.utils",
            "script",
            "tests",
        }
        missing_expected = expected_packages - discovered_packages
        self.assertEqual(
            missing_expected,
            set(),
            f"ファイル構造抽出結果に期待されるパッケージが含まれていません: {missing_expected}",
        )


if __name__ == "__main__":
    unittest.main()
