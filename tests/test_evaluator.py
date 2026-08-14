import csv
import tempfile
import unittest
from pathlib import Path

from voicerecognizer.core.interfaces import RecognitionStrategy
from voicerecognizer.evaluation.evaluator import EvaluationResult, Evaluator


class MockRecognizer(RecognitionStrategy):
    def recognize(self, audio: str) -> str:
        # Mock recognition logic for testing
        if "a" in audio:
            return "a"
        elif "e" in audio:
            return "e"
        return "other"


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_dir = Path(self.temp_dir.name)

        # Create mock index.csv
        self.index_path = self.dataset_dir / "index.csv"
        with open(self.index_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "label", "machine_id", "predicted_text"])
            writer.writerow(["sample_a1.wav", "a", "mac1", "a"])
            writer.writerow(["sample_a2.wav", "a", "mac1", "e"])  # misclassified
            writer.writerow(["sample_e1.wav", "e", "mac1", "e"])

        # Create mock audio files
        (self.dataset_dir / "sample_a1.wav").touch()
        (self.dataset_dir / "sample_a2.wav").touch()
        (self.dataset_dir / "sample_e1.wav").touch()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_update_from_dataset(self):
        evaluator = Evaluator(
            model=None,
            labels=("a", "e", "i", "o", "u", "other"),
            dataset_path=self.dataset_dir,
        )
        result = evaluator.update_from_dataset()

        self.assertIsInstance(result, EvaluationResult)
        self.assertAlmostEqual(result.overall.accuracy, 2 / 3, places=2)
        self.assertEqual(len(result.misclassified), 1)
        self.assertEqual(result.misclassified[0].true_label, "a")
        self.assertEqual(result.misclassified[0].predicted_label, "e")

    def test_evaluate_with_model(self):
        mock_model = MockRecognizer()
        evaluator = Evaluator(
            model=mock_model,
            labels=("a", "e", "i", "o", "u", "other"),
            dataset_path=self.dataset_dir,
        )
        result = evaluator.evaluate()

        self.assertIsInstance(result, EvaluationResult)
        self.assertGreater(result.overall.total_samples, 0)

    def test_update_index_with_predictions(self):
        mock_model = MockRecognizer()
        evaluator = Evaluator(
            model=mock_model,
            labels=("a", "e", "i", "o", "u", "other"),
            dataset_path=self.dataset_dir,
        )
        evaluator.update_index_with_predictions()

        with open(self.index_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertEqual(len(rows), 3)
            self.assertIn("predicted_text", rows[0])
            self.assertEqual(rows[0]["predicted_text"], "a")

    def test_export_json(self):
        evaluator = Evaluator(
            model=None,
            labels=("a", "e", "i", "o", "u", "other"),
            dataset_path=self.dataset_dir,
        )
        evaluator.update_from_dataset()

        output_json = self.dataset_dir / "output.json"
        success = evaluator.export_json(output_json)
        self.assertTrue(success)
        self.assertTrue(output_json.exists())


if __name__ == "__main__":
    unittest.main()
