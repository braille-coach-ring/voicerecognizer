import unittest
from unittest.mock import patch

from preprocessing.dataset_builder import ensure_merged_and_preprocessed
from models.cnn.train import build_parser as cnn_build_parser
from models.wav2vec2.train import build_parser as wav2vec2_build_parser


class TestAutoPrepTraining(unittest.TestCase):
    def test_parsers_have_skip_prep_flag(self):
        cnn_parser = cnn_build_parser()
        args = cnn_parser.parse_args(["--skip-prep"])
        self.assertTrue(args.skip_prep)

        wav2vec2_parser = wav2vec2_build_parser()
        w_args = wav2vec2_parser.parse_args(["--skip-prep"])
        self.assertTrue(w_args.skip_prep)

    @patch("preprocessing.dataset_builder.DatasetBuilder")
    def test_ensure_merged_and_preprocessed_default_runs(self, mock_builder_cls):
        ensure_merged_and_preprocessed(skip_prep=False)
        mock_builder_cls.return_value.build_index.assert_called_once()
        mock_builder_cls.return_value.preprocess_dataset.assert_called_once()

    @patch("preprocessing.dataset_builder.DatasetBuilder")
    def test_ensure_merged_and_preprocessed_skip_prep(self, mock_builder_cls):
        ensure_merged_and_preprocessed(skip_prep=True)
        mock_builder_cls.return_value.build_index.assert_not_called()
        mock_builder_cls.return_value.preprocess_dataset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
