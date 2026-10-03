import csv
import json
import tempfile
import unittest
from pathlib import Path

from voicerecognizer.config import DEFAULT_RECOGNITION_CONFIG
from voicerecognizer.core.interfaces import RecognitionStrategy
from voicerecognizer.evaluation.evaluator import (
    EvaluationResult,
    Evaluator,
    compute_evaluation_result,
    resolve_evaluation_labels,
)


class MockRecognizer(RecognitionStrategy):
    def recognize(self, audio: str) -> str:
        # Mock recognition logic for testing
        if "a" in audio:
            return "a"
        elif "e" in audio:
            return "e"
        return "other"


class CandidateMockRecognizer(RecognitionStrategy):
    def __init__(self) -> None:
        self.calls = 0
        self.last_confidence: float | None = None
        self.last_timing_stats: dict[str, float] = {}

    def recognize(self, audio: str) -> str:
        raise AssertionError("recognize_with_candidates should be used when available")

    def recognize_with_candidates(self, audio: str, top_k: int = 3) -> list[tuple[str, float]]:
        self.calls += 1
        if "sample_a2" in audio:
            self.last_timing_stats = {
                "onset_ms": 10.0,
                "offset_ms": 420.0,
                "speech_duration_ms": 410.0,
            }
            candidates = [("e", 0.97), ("a", 0.02), ("o", 0.01)]
        elif "sample_a1" in audio:
            self.last_timing_stats = {
                "onset_ms": 5.0,
                "offset_ms": 80.0,
                "speech_duration_ms": 75.0,
            }
            candidates = [("a", 0.38), ("e", 0.31), ("o", 0.20)]
        else:
            self.last_timing_stats = {
                "onset_ms": 20.0,
                "offset_ms": 360.0,
                "speech_duration_ms": 340.0,
            }
            candidates = [("e", 0.91), ("a", 0.05), ("o", 0.04)]

        self.last_confidence = candidates[0][1]
        return candidates[:top_k]


class TestEvaluator(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_dir = Path(self.temp_dir.name)

        # Create mock index.csv
        self.index_path = self.dataset_dir / "index.csv"
        with open(self.index_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "label", "speaker", "machine_id", "predicted_text"])
            writer.writerow(["sample_a1.wav", "a", "speaker_a", "mac1", "a"])
            writer.writerow(["sample_a2.wav", "a", "speaker_a", "mac1", "e"])  # misclassified
            writer.writerow(["sample_e1.wav", "e", "speaker_b", "mac2", "e"])

        # Create mock audio files
        (self.dataset_dir / "sample_a1.wav").touch()
        (self.dataset_dir / "sample_a2.wav").touch()
        (self.dataset_dir / "sample_e1.wav").touch()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_update_from_dataset(self) -> None:
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
        self.assertEqual(result.misclassified[0].speaker, "speaker_a")
        self.assertEqual(result.speaker_metrics["speaker_a"].total_samples, 2)
        self.assertEqual(result.speaker_metrics["speaker_a"].misclassified_samples, 1)
        self.assertAlmostEqual(result.speaker_metrics["speaker_a"].accuracy, 0.5)
        self.assertEqual(result.speaker_metrics["speaker_b"].total_samples, 1)
        self.assertAlmostEqual(result.speaker_metrics["speaker_b"].accuracy, 1.0)
        self.assertEqual(
            result.speaker_metrics["speaker_a"].top_confusions[0],
            {"true_label": "a", "predicted_label": "e", "count": 1},
        )

    def test_evaluate_with_model(self) -> None:
        mock_model = MockRecognizer()
        evaluator = Evaluator(
            model=mock_model,
            labels=("a", "e", "i", "o", "u", "other"),
            dataset_path=self.dataset_dir,
        )
        result = evaluator.evaluate()

        self.assertIsInstance(result, EvaluationResult)
        self.assertGreater(result.overall.total_samples, 0)

    def test_update_index_with_predictions(self) -> None:
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

    def test_export_json(self) -> None:
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
        with open(output_json, encoding="utf-8") as f:
            payload = json.load(f)
        self.assertIn("speaker_metrics", payload)
        self.assertEqual(payload["speaker_metrics"]["speaker_a"]["total_samples"], 2)

    def test_compute_evaluation_result_accepts_speakers(self) -> None:
        result = compute_evaluation_result(
            ["a", "a", "e"],
            ["a", "e", "e"],
            labels=("a", "e"),
            speakers=["speaker_a", "speaker_a", "speaker_b"],
        )

        self.assertEqual(result.speaker_metrics["speaker_a"].total_samples, 2)
        self.assertAlmostEqual(result.speaker_metrics["speaker_a"].accuracy, 0.5)
        self.assertEqual(result.speaker_metrics["speaker_b"].misclassified_samples, 0)

    def test_export_html_includes_speaker_metrics(self) -> None:
        evaluator = Evaluator(
            model=None,
            labels=("a", "e", "i", "o", "u", "other"),
            dataset_path=self.dataset_dir,
        )
        evaluator.update_from_dataset()

        output_html = self.dataset_dir / "evaluation_report.html"
        self.assertTrue(evaluator.export_html(output_html))

        html = output_html.read_text(encoding="utf-8")
        self.assertIn("話者別精度指標", html)
        self.assertIn("speaker_a", html)
        self.assertIn("speaker_b", html)

    def test_review_candidates_include_mismatch_and_low_confidence(self) -> None:
        mock_model = CandidateMockRecognizer()
        evaluator = Evaluator(
            model=mock_model,
            labels=("a", "e", "i", "o", "u", "other"),
            dataset_path=self.dataset_dir,
        )
        evaluator.evaluate()

        self.assertEqual(mock_model.calls, 3)
        self.assertEqual(len(evaluator.review_candidates), 3)
        candidates = {candidate.filepath: candidate for candidate in evaluator.review_candidates}

        mismatch = candidates["sample_a2.wav"]
        self.assertEqual(mismatch.true_label, "a")
        self.assertEqual(mismatch.predicted_label, "e")
        self.assertAlmostEqual(mismatch.confidence or 0.0, 0.97)

        low_confidence = candidates["sample_a1.wav"]
        self.assertEqual(low_confidence.predicted_label, "a")
        self.assertAlmostEqual(low_confidence.confidence or 0.0, 0.38)
        self.assertIn("short_speech", low_confidence.quality_flags)
        self.assertGreater(mismatch.review_priority, low_confidence.review_priority)

    def test_review_decisions_are_reused_and_reports_are_exported(self) -> None:
        decisions_path = self.dataset_dir / "review_decisions.json"
        with open(decisions_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": 1,
                    "decisions": [
                        {
                            "filepath": "sample_a2.wav",
                            "label": "a",
                            "prediction": "e",
                            "confidence": 0.97,
                            "decision": "delete_candidate",
                        }
                    ],
                },
                f,
            )

        evaluator = Evaluator(
            model=CandidateMockRecognizer(),
            labels=("a", "e", "i", "o", "u", "other"),
            dataset_path=self.dataset_dir,
            review_decisions_path=decisions_path,
        )
        evaluator.evaluate()

        candidates = {candidate.filepath: candidate for candidate in evaluator.review_candidates}
        self.assertEqual(candidates["sample_a2.wav"].decision, "delete_candidate")

        output_json = self.dataset_dir / "review_candidates.json"
        output_html = self.dataset_dir / "review_report.html"
        self.assertTrue(evaluator.export_review_json(output_json))
        self.assertTrue(
            evaluator.export_review_html(
                output_html,
                review_results_path=decisions_path,
            )
        )

        with open(output_json, encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(payload["total"], 3)

        html = output_html.read_text(encoding="utf-8")
        self.assertIn("sample_a2.wav", html)
        self.assertIn("delete_candidate", html)
        self.assertIn("<audio controls", html)
        self.assertIn('data-decision="keep"', html)
        self.assertIn('data-decision="delete_candidate"', html)
        self.assertIn('data-decision="maybe"', html)
        self.assertIn('data-action="quick-relabel"', html)
        self.assertIn('data-action="set-other"', html)
        self.assertIn('<span class="key">A</span> 現行OK', html)
        self.assertIn('<span class="key">D</span> 雑音(other)', html)
        self.assertIn('<span class="key">F</span> 保留', html)


class LabelAwareMockRecognizer(RecognitionStrategy):
    """labels.json を読み込んだ recognizer 相当 (自分が出力しうるラベルを持つ)"""

    def __init__(self, labels: list[str]) -> None:
        self.labels = labels

    def recognize(self, audio: str) -> str:
        return self.labels[0]


class TestEvaluationLabelResolution(unittest.TestCase):
    """評価ラベル集合が、学習側と同じ母数になることを保証する。"""

    def _write_index(self, tmp_path: Path, rows: list[tuple[str, str, str]]) -> None:
        lines = ["filepath,label,predicted_text"]
        lines += [f"{filepath},{label},{predicted}" for filepath, label, predicted in rows]
        (tmp_path / "index.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_resolve_uses_model_labels_when_available(self) -> None:
        model = LabelAwareMockRecognizer(["a", "i", "u"])
        self.assertEqual(list(resolve_evaluation_labels(model)), ["a", "i", "u"])

    def test_resolve_falls_back_to_config_labels_without_model(self) -> None:
        self.assertEqual(tuple(resolve_evaluation_labels(None)), DEFAULT_RECOGNITION_CONFIG.labels)

    def test_evaluator_defaults_to_model_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._write_index(tmp_path, [("f1.wav", "a", "a")])
            model = LabelAwareMockRecognizer(["a", "i", "u"])

            evaluator = Evaluator(model=model, dataset_path=tmp_path)

            # config の 105 ラベルではなく、モデル自身のラベルが使われること
            self.assertEqual(evaluator.labels, ("a", "i", "u"))
            self.assertNotEqual(len(evaluator.labels), len(DEFAULT_RECOGNITION_CONFIG.labels))

    def test_explicit_labels_still_win(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._write_index(tmp_path, [("f1.wav", "a", "a")])
            model = LabelAwareMockRecognizer(["a", "i", "u"])

            evaluator = Evaluator(model=model, labels=["a", "e"], dataset_path=tmp_path)
            self.assertEqual(evaluator.labels, ("a", "e"))

    def test_macro_f1_is_not_diluted_by_untrained_classes(self) -> None:
        """モデルが学習していないクラスを母数に入れると Macro-F1 が不当に下がる。"""
        y_true = ["a", "i", "u"]
        y_pred = ["a", "i", "u"]

        trained_only = compute_evaluation_result(y_true, y_pred, labels=("a", "i", "u"))
        with_untrained = compute_evaluation_result(
            y_true, y_pred, labels=("a", "i", "u", "kya", "pyo")
        )

        self.assertEqual(trained_only.overall.macro_f1, 1.0)
        self.assertLess(with_untrained.overall.macro_f1, 1.0)

    def test_warns_when_true_label_is_outside_evaluation_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._write_index(tmp_path, [("f1.wav", "a", "a"), ("f2.wav", "zzz", "a")])
            evaluator = Evaluator(model=None, labels=["a", "i"], dataset_path=tmp_path)

            with self.assertLogs("voicerecognizer.evaluation.evaluator", level="WARNING") as cm:
                evaluator.update_from_dataset()

            self.assertTrue(any("zzz" in line for line in cm.output))


if __name__ == "__main__":
    unittest.main()
