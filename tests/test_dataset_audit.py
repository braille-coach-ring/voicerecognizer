import csv
import json
import tempfile
import unittest
from pathlib import Path

from voicerecognizer.evaluation.dataset_audit import audit_dataset, format_audit_summary


class TestDatasetAudit(unittest.TestCase):
    def test_audit_reports_missing_duplicates_stale_processed_and_review_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_root = tmp_path / "dataset"
            merged_root = tmp_path / "merged_dataset"
            processed_root = tmp_path / "processed_dataset"
            eval_root = tmp_path / "evaluation_results"
            merged_root.mkdir()
            processed_root.mkdir()
            eval_root.mkdir()

            a1 = raw_root / "speaker1" / "a" / "001.wav"
            a2 = raw_root / "speaker2" / "a" / "002.wav"
            missing = raw_root / "speaker3" / "e" / "999.wav"
            a1.parent.mkdir(parents=True)
            a2.parent.mkdir(parents=True)
            a1.write_bytes(b"same-audio")
            a2.write_bytes(b"same-audio")
            (processed_root / "a").mkdir()
            (processed_root / "a" / "001.wav").write_bytes(b"processed")

            with open(merged_root / "index.csv", "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["filepath", "label", "predicted_text"])
                writer.writerow([str(a1), "a", ""])
                writer.writerow([str(a1), "a", ""])
                writer.writerow([str(a2), "a", ""])
                writer.writerow([str(missing), "e", ""])

            review_candidates_path = eval_root / "review_candidates.json"
            with open(review_candidates_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "candidates": [
                            {
                                "filepath": "sample_a.wav",
                                "true_label": "a",
                                "predicted_label": "e",
                                "decision": "",
                                "review_priority": 320.0,
                                "quality_flags": ["late_onset"],
                            },
                            {
                                "filepath": "sample_e.wav",
                                "true_label": "e",
                                "predicted_label": "e",
                                "decision": "keep",
                                "review_priority": 0.0,
                                "quality_flags": [],
                            },
                        ]
                    },
                    f,
                )

            evaluation_result_path = eval_root / "evaluation_result.json"
            with open(evaluation_result_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "confusion_matrix": {
                            "a": {"a": 2, "e": 1},
                            "e": {"a": 0, "e": 1},
                        }
                    },
                    f,
                )

            report = audit_dataset(
                merged_dataset_dir=merged_root,
                processed_dataset_dir=processed_root,
                raw_dataset_dir=raw_root,
                review_candidates_path=review_candidates_path,
                review_decisions_path=None,
                evaluation_result_path=evaluation_result_path,
                labels=("a", "e", "other"),
                min_label_count=2,
                project_root=tmp_path,
            )

            self.assertEqual(report["merged_index"]["rows"], 4)
            self.assertEqual(report["merged_index"]["existing_files"], 3)
            self.assertEqual(report["merged_index"]["missing_files"], 1)
            self.assertEqual(report["merged_index"]["missing_by_label"][0]["name"], "e")
            self.assertEqual(len(report["merged_index"]["duplicate_filepaths"]), 1)
            self.assertEqual(len(report["merged_index"]["duplicate_sources"]), 1)
            self.assertEqual(len(report["merged_index"]["duplicate_audio_hashes"]), 1)
            self.assertTrue(report["processed_dataset"]["is_stale_or_incomplete"])
            self.assertIn("processed_index_missing", report["processed_dataset"]["reasons"])
            self.assertEqual(report["review_candidates"]["unreviewed"], 1)
            self.assertEqual(report["review_candidates"]["quality_flags"]["late_onset"], 1)
            self.assertEqual(report["confusions"]["top_pairs"][0]["true_label"], "a")

            summary = format_audit_summary(report)
            self.assertIn("Dataset Audit", summary)
            self.assertIn("missing=1", summary)
