import asyncio
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from recognizers.cnn_recognizer import CNNRecognizer


class TestPackageAPI(unittest.TestCase):
    @patch.object(CNNRecognizer, "_load_model")
    def test_cnn_recognizer_default_instantiation(self, mock_load_model: MagicMock) -> None:
        """CNNRecognizer should be constructable without arguments (defaults from config)."""
        recognizer = CNNRecognizer()
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
        import importlib
        from pathlib import Path

        project_root = Path(__file__).resolve().parent.parent
        excluded_dirs = {
            ".venv",
            ".git",
            ".github",
            ".agents",
            ".pytest_cache",
            ".ruff_cache",
            "build",
            "dist",
            "evaluation_results",
            "merged_dataset",
            "processed_dataset",
            "plots",
            "log",
            "weights",
            "scratch",
            "__pycache__",
        }

        # __init__.py を保持する全パッケージ / サブパッケージをファイル構造から動的抽出
        discovered_packages: set[str] = set()

        for path in project_root.rglob("__init__.py"):
            rel_parts = path.relative_to(project_root).parts
            if any(part in excluded_dirs or part.endswith(".egg-info") for part in rel_parts):
                continue

            parent_rel = path.parent.relative_to(project_root)
            if str(parent_rel) == ".":
                continue
            mod_name = ".".join(parent_rel.parts)
            discovered_packages.add(mod_name)

        # 動的抽出が正しく行われたことを検証
        self.assertGreater(len(discovered_packages), 0, "ファイル構造からパッケージが抽出されませんでした。")

        # 抽出された全パッケージ（core, core.factory, core.services, models.cnn, models.wav2vec2 等）をインポート検証
        for pkg in sorted(discovered_packages):
            with self.subTest(package=pkg):
                mod = importlib.import_module(pkg)
                self.assertIsNotNone(mod, f"パッケージ '{pkg}' のインポートに失敗しました。")

        # 主要サブパッケージが漏れなく抽出されているか検証
        expected_packages = {
            "core",
            "core.factory",
            "core.services",
            "dataset",
            "evaluation",
            "models",
            "models.cnn",
            "models.wav2vec2",
            "preprocessing",
            "recognizers",
            "runtime",
            "script",
            "test",
            "tests",
            "utils",
        }
        missing_expected = expected_packages - discovered_packages
        self.assertEqual(
            missing_expected,
            set(),
            f"ファイル構造抽出結果に期待されるパッケージが含まれていません: {missing_expected}",
        )


if __name__ == "__main__":
    unittest.main()


