import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from script.download_from_hf import download_models
from script.setup_environment import setup_dataset, setup_directories, setup_models


class TestSetupScripts(unittest.TestCase):
    def test_setup_directories_creates_all_dirs(self):
        setup_directories()

    @patch("script.setup_environment.download_latest_team_weights_if_needed", return_value=True)
    def test_setup_models_success(self, mock_download: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir)
            ok = setup_models(dest_dir=dest)
            self.assertTrue(ok)
            self.assertEqual(mock_download.call_count, 2)

    @patch("script.download_from_hf.download_latest_team_weights_if_needed", return_value=True)
    def test_download_models_all(self, mock_download: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir)
            ok = download_models(model_type="all", target_dir=dest)
            self.assertTrue(ok)
            self.assertEqual(mock_download.call_count, 2)

    @patch("script.download_from_hf.download_latest_team_weights_if_needed", return_value=True)
    def test_download_models_individual(self, mock_download: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir)
            ok_w2v = download_models(model_type="wav2vec2", target_dir=dest)
            self.assertTrue(ok_w2v)
            ok_cnn = download_models(model_type="cnn", target_dir=dest)
            self.assertTrue(ok_cnn)

    @patch("script.setup_environment.DatasetBuilder")
    def test_setup_dataset_no_raw_files(self, mock_builder_cls: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_dataset(raw_dir=Path(tmpdir) / "empty_dir")
            mock_builder_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
