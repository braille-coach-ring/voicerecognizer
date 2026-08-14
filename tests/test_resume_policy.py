import unittest

from voicerecognizer.models.cnn.train import build_parser as cnn_build_parser
from voicerecognizer.models.wav2vec2.train import build_parser as wav2vec2_build_parser


class TestResumePolicy(unittest.TestCase):
    def test_cnn_resume_default_is_true(self):
        parser = cnn_build_parser()
        args_default = parser.parse_args([])
        self.assertTrue(args_default.resume)

        args_from_scratch = parser.parse_args(["--from-scratch"])
        self.assertFalse(args_from_scratch.resume)

        args_no_resume = parser.parse_args(["--no-resume"])
        self.assertFalse(args_no_resume.resume)

    def test_wav2vec2_resume_default_is_true(self):
        parser = wav2vec2_build_parser()
        args_default = parser.parse_args([])
        self.assertTrue(args_default.resume)

        args_from_scratch = parser.parse_args(["--from-scratch"])
        self.assertFalse(args_from_scratch.resume)

        args_no_resume = parser.parse_args(["--no-resume"])
        self.assertFalse(args_no_resume.resume)


if __name__ == "__main__":
    unittest.main()
