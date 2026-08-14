import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from voicerecognizer.config import DEFAULT_RECOGNITION_CONFIG, HuggingFaceConfig
from voicerecognizer.utils.model_uploader import download_latest_team_weights_if_needed, upload_weights_to_hf


class TestModelUploader(unittest.TestCase):
    def test_huggingface_config_default(self):
        cfg = HuggingFaceConfig(token="", repo_id="test/repo", auto_upload=False)
        self.assertEqual(cfg.repo_id, "test/repo")
        self.assertFalse(cfg.auto_upload)

    @patch("voicerecognizer.utils.model_uploader.calculate_file_sha256", return_value="same_hash")
    @patch(
        "voicerecognizer.utils.model_uploader.get_remote_file_sha256_map",
        return_value={"best_model.pth": "same_hash", "labels.json": "same_hash"},
    )
    @patch("voicerecognizer.utils.model_uploader.hf_hub_download")
    def test_download_latest_team_weights_skip_when_identical(
        self, mock_download, mock_remote_map, mock_sha
    ):
        dummy_cfg = HuggingFaceConfig(token="dummy_token_123", repo_id="dummy/repo-id")
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_dir = Path(tmpdir)
            (weights_dir / "best_model.pth").touch()
            (weights_dir / "labels.json").touch()

            res = download_latest_team_weights_if_needed(
                model_type="cnn", hf_config=dummy_cfg, weights_dir=weights_dir
            )
            self.assertTrue(res)
            mock_download.assert_not_called()

    @patch("voicerecognizer.utils.model_uploader.calculate_file_sha256", return_value="same_hash")
    @patch("voicerecognizer.utils.model_uploader.get_remote_file_sha256_map")
    @patch("voicerecognizer.utils.model_uploader.hf_hub_download")
    def test_download_latest_team_weights_wav2vec2(self, mock_download, mock_remote_map, mock_sha):
        essential = DEFAULT_RECOGNITION_CONFIG.wav2vec2_essential_filenames
        mock_remote_map.return_value = {f"wav2vec2_best/{fname}": "same_hash" for fname in essential}
        dummy_cfg = HuggingFaceConfig(token="dummy_token_123", repo_id="dummy/repo-id")
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_dir = Path(tmpdir)
            w2v_dir = weights_dir / "wav2vec2_best"
            w2v_dir.mkdir(parents=True, exist_ok=True)
            for fname in essential:
                (w2v_dir / fname).touch()

            res = download_latest_team_weights_if_needed(
                model_type="wav2vec2", hf_config=dummy_cfg, weights_dir=weights_dir
            )
            self.assertTrue(res)
            mock_download.assert_not_called()

    @patch("voicerecognizer.utils.model_uploader.login")
    @patch("voicerecognizer.utils.model_uploader.HfApi")
    @patch("voicerecognizer.utils.model_uploader.calculate_file_sha256", return_value="dummy_hash_123")
    @patch("voicerecognizer.utils.model_uploader.get_remote_file_sha256_map", return_value={})
    def test_upload_weights_cnn_only(
        self, mock_remote_map, mock_sha, mock_hf_api_class, mock_login
    ):
        mock_api_instance = MagicMock()
        mock_hf_api_class.return_value = mock_api_instance
        dummy_cfg = HuggingFaceConfig(
            token="dummy_token_123", repo_id="dummy/repo-id", auto_upload=True
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_dir = Path(tmpdir)
            (weights_dir / "best_model.pth").write_text("dummy pth content")
            (weights_dir / "labels.json").write_text("[]")

            res = upload_weights_to_hf(
                model_type="cnn", hf_config=dummy_cfg, weights_dir=weights_dir, force_upload=True
            )
            self.assertTrue(res)
            mock_login.assert_called_once_with(token="dummy_token_123")
            self.assertEqual(mock_api_instance.upload_file.call_count, 2)

    @patch("voicerecognizer.utils.model_uploader.login")
    @patch("voicerecognizer.utils.model_uploader.HfApi")
    @patch("voicerecognizer.utils.model_uploader.calculate_file_sha256", return_value="same_hash")
    @patch(
        "voicerecognizer.utils.model_uploader.get_remote_file_sha256_map",
        return_value={"best_model.pth": "same_hash", "labels.json": "same_hash"},
    )
    def test_upload_weights_cnn_skip_identical(
        self, mock_remote_map, mock_sha, mock_hf_api_class, mock_login
    ):
        mock_api_instance = MagicMock()
        mock_hf_api_class.return_value = mock_api_instance
        dummy_cfg = HuggingFaceConfig(token="dummy_token_123", repo_id="dummy/repo-id")

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_dir = Path(tmpdir)
            (weights_dir / "best_model.pth").write_text("dummy pth content")
            (weights_dir / "labels.json").write_text("[]")

            res = upload_weights_to_hf(
                model_type="cnn", hf_config=dummy_cfg, weights_dir=weights_dir, force_upload=False
            )
            self.assertTrue(res)
            self.assertEqual(mock_api_instance.upload_file.call_count, 0)

    @patch("voicerecognizer.utils.model_uploader.login")
    @patch("voicerecognizer.utils.model_uploader.HfApi")
    @patch("voicerecognizer.utils.model_uploader.calculate_file_sha256", return_value="dummy_hash_123")
    @patch("voicerecognizer.utils.model_uploader.get_remote_file_sha256_map", return_value={})
    def test_upload_weights_wav2vec2_essential_files_strict_count(
        self, mock_remote_map, mock_sha, mock_hf_api_class, mock_login
    ):
        mock_api_instance = MagicMock()
        mock_hf_api_class.return_value = mock_api_instance
        dummy_cfg = HuggingFaceConfig(token="dummy_token_123", repo_id="dummy/repo-id")

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_dir = Path(tmpdir)
            w2v_dir = weights_dir / "wav2vec2_best"
            w2v_dir.mkdir(parents=True, exist_ok=True)

            # 4 つの必須ファイルのみ作成 (別ファイルは作成しない)
            essential_files = [
                "config.json",
                "labels.json",
                "model.safetensors",
                "preprocessor_config.json",
            ]
            for fname in essential_files:
                (w2v_dir / fname).write_text("dummy content")

            res = upload_weights_to_hf(
                model_type="wav2vec2",
                hf_config=dummy_cfg,
                weights_dir=weights_dir,
                force_upload=True,
            )
            self.assertTrue(res)
            # 存在する必須ファイル数（4件）と一致することを厳密に検証
            self.assertEqual(mock_api_instance.upload_file.call_count, 4)

    def test_upload_weights_missing_token(self):
        dummy_cfg = HuggingFaceConfig(token="", repo_id="dummy/repo-id")
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_dir = Path(tmpdir)
            (weights_dir / "best_model.pth").touch()
            res = upload_weights_to_hf(hf_config=dummy_cfg, weights_dir=weights_dir)
            self.assertFalse(res)


if __name__ == "__main__":
    unittest.main()
