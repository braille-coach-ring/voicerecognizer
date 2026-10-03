import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from voicerecognizer.strategies.registry import (
    create_strategy_recognizer,
    resolve_strategy_model_path,
)


class TestStrategyHFSync(unittest.TestCase):
    def test_resolve_strategy_model_path(self):
        path = resolve_strategy_model_path("wav2vec2_phoneme_multi")
        self.assertEqual(path.name, "wav2vec2_phoneme_multi")

    @patch("voicerecognizer.utils.model_uploader.download_strategy_weights_if_needed")
    @patch(
        "voicerecognizer.recognizers.wav2vec2_recognizer.Wav2Vec2Recognizer.__init__",
        return_value=None,
    )
    def test_create_strategy_recognizer_triggers_download_when_missing(
        self, mock_w2v_init, mock_download
    ):

        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_target_dir = Path(tmpdir) / "wav2vec2_phoneme_multi"
            # Directory does not exist yet

            def fake_download(strategy_name, target_dir=None):
                target_dir.mkdir(parents=True, exist_ok=True)
                (target_dir / "model_mel_int8.onnx").touch()
                (target_dir / "labels.json").touch()
                return True

            mock_download.side_effect = fake_download

            _ = create_strategy_recognizer(
                strategy_name="wav2vec2_phoneme_multi",
                model_path=dummy_target_dir,
                auto_download=True,
            )

            mock_download.assert_called_once_with(
                strategy_name="wav2vec2_phoneme_multi",
                target_dir=dummy_target_dir,
            )
            mock_w2v_init.assert_called_once()

    @patch("voicerecognizer.utils.model_uploader.HfApi")
    def test_list_remote_strategies(self, mock_hf_api_class):
        from voicerecognizer.utils.model_uploader import list_remote_strategies

        mock_api = MagicMock()
        item1 = MagicMock()
        item1.path = "strategies/wav2vec2_phoneme_multi/model.safetensors"
        item2 = MagicMock()
        item2.path = "strategies/wav2vec2_ipa_kd/model.safetensors"
        item3 = MagicMock()
        item3.path = "other_folder/file.txt"
        mock_api.list_repo_tree.return_value = [item1, item2, item3]
        mock_hf_api_class.return_value = mock_api

        strategies = list_remote_strategies()
        self.assertEqual(strategies, ["wav2vec2_ipa_kd", "wav2vec2_phoneme_multi"])
