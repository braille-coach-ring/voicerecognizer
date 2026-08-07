import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

from config import HuggingFaceConfig
from utils.model_uploader import upload_weights_to_hf


class TestModelUploader(unittest.TestCase):
    def test_huggingface_config_default(self):
        cfg = HuggingFaceConfig(token="", repo_id="test/repo", auto_upload=False)
        self.assertEqual(cfg.repo_id, "test/repo")
        self.assertFalse(cfg.auto_upload)

    @patch("utils.model_uploader.login")
    @patch("utils.model_uploader.HfApi")
    @patch("utils.model_uploader.calculate_file_sha256", return_value="dummy_hash_123")
    @patch("utils.model_uploader.get_remote_file_sha256_map", return_value={})
    def test_upload_weights_cnn_only(self, mock_remote_map, mock_sha, mock_hf_api_class, mock_login):
        mock_api_instance = MagicMock()
        mock_hf_api_class.return_value = mock_api_instance

        dummy_cfg = HuggingFaceConfig(
            token="dummy_token_123",
            repo_id="dummy/repo-id",
            auto_upload=True,
        )

        dummy_dir = Path("weights")
        with patch.object(Path, "exists", return_value=True):
            res = upload_weights_to_hf(model_type="cnn", hf_config=dummy_cfg, weights_dir=dummy_dir, force_upload=True)
            self.assertTrue(res)
            mock_login.assert_called_once_with(token="dummy_token_123")
            self.assertEqual(mock_api_instance.upload_file.call_count, 2)

    @patch("utils.model_uploader.login")
    @patch("utils.model_uploader.HfApi")
    @patch("utils.model_uploader.calculate_file_sha256", return_value="same_hash")
    @patch("utils.model_uploader.get_remote_file_sha256_map", return_value={"best_model.pth": "same_hash", "labels.json": "same_hash"})
    def test_upload_weights_cnn_skip_identical(self, mock_remote_map, mock_sha, mock_hf_api_class, mock_login):
        mock_api_instance = MagicMock()
        mock_hf_api_class.return_value = mock_api_instance

        dummy_cfg = HuggingFaceConfig(token="dummy_token_123", repo_id="dummy/repo-id")
        dummy_dir = Path("weights")

        with patch.object(Path, "exists", return_value=True):
            res = upload_weights_to_hf(model_type="cnn", hf_config=dummy_cfg, weights_dir=dummy_dir, force_upload=False)
            self.assertTrue(res)
            # ハッシュがリモートと一致するため upload_file は一度も呼ばれない
            self.assertEqual(mock_api_instance.upload_file.call_count, 0)

    @patch("utils.model_uploader.login")
    @patch("utils.model_uploader.HfApi")
    def test_upload_weights_best_only(self, mock_hf_api_class, mock_login):
        mock_api_instance = MagicMock()
        mock_hf_api_class.return_value = mock_api_instance

        dummy_cfg = HuggingFaceConfig(
            token="dummy_token_123",
            repo_id="dummy/repo-id",
            auto_upload=True,
        )

        dummy_dir = Path("weights")
        with patch.object(Path, "exists", return_value=True):
            res = upload_weights_to_hf(model_type="best_only", hf_config=dummy_cfg, weights_dir=dummy_dir)
            self.assertTrue(res)
            mock_login.assert_called_once_with(token="dummy_token_123")
            mock_api_instance.upload_folder.assert_called_once()

    def test_upload_weights_missing_token(self):
        dummy_cfg = HuggingFaceConfig(token="", repo_id="dummy/repo-id")
        dummy_dir = Path("weights")
        with patch.object(Path, "exists", return_value=True):
            res = upload_weights_to_hf(hf_config=dummy_cfg, weights_dir=dummy_dir)
            self.assertFalse(res)


if __name__ == "__main__":
    unittest.main()
