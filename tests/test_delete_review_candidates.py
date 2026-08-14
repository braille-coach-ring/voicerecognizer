import json
import tempfile
import unittest
from pathlib import Path

from script.delete_review_candidates import (
    delete_review_candidates,
    summarize_results,
    write_delete_log,
)


class TestDeleteReviewCandidates(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp_dir.name)
        self.dataset_dir = self.project_root / "dataset" / "speaker" / "a"
        self.dataset_dir.mkdir(parents=True)

        self.delete_target = self.dataset_dir / "delete_me.wav"
        self.keep_target = self.dataset_dir / "keep_me.wav"
        self.missing_target = self.dataset_dir / "missing.wav"
        self.delete_target.write_bytes(b"wav")
        self.keep_target.write_bytes(b"wav")

        self.decisions_path = self.project_root / "evaluation_results" / "review_decisions.json"
        self.decisions_path.parent.mkdir(parents=True)
        with open(self.decisions_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": 1,
                    "decisions": [
                        {
                            "filepath": "dataset/speaker/a/delete_me.wav",
                            "label": "a",
                            "prediction": "i",
                            "confidence": 0.97,
                            "decision": "delete_candidate",
                        },
                        {
                            "filepath": "dataset/speaker/a/keep_me.wav",
                            "label": "a",
                            "prediction": "a",
                            "confidence": 0.99,
                            "decision": "keep",
                        },
                        {
                            "filepath": "dataset/speaker/a/missing.wav",
                            "label": "a",
                            "prediction": "u",
                            "confidence": 0.80,
                            "decision": "delete_candidate",
                        },
                    ],
                },
                f,
            )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_dry_run_does_not_delete_files(self) -> None:
        results = delete_review_candidates(
            self.decisions_path,
            project_root=self.project_root,
        )

        self.assertTrue(self.delete_target.exists())
        self.assertTrue(self.keep_target.exists())
        self.assertEqual(summarize_results(results), {"would_delete": 1, "missing": 1})

    def test_execute_deletes_only_delete_candidates(self) -> None:
        results = delete_review_candidates(
            self.decisions_path,
            project_root=self.project_root,
            execute=True,
        )

        self.assertFalse(self.delete_target.exists())
        self.assertTrue(self.keep_target.exists())
        self.assertEqual(summarize_results(results), {"deleted": 1, "missing": 1})

    def test_skips_non_wav_by_default(self) -> None:
        text_target = self.dataset_dir / "not_audio.txt"
        text_target.write_text("not wav", encoding="utf-8")
        decisions_path = self.project_root / "evaluation_results" / "non_wav_decisions.json"
        with open(decisions_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": 1,
                    "decisions": [
                        {
                            "filepath": "dataset/speaker/a/not_audio.txt",
                            "label": "a",
                            "prediction": "i",
                            "confidence": 0.97,
                            "decision": "delete_candidate",
                        }
                    ],
                },
                f,
            )

        results = delete_review_candidates(
            decisions_path,
            project_root=self.project_root,
        )

        self.assertTrue(text_target.exists())
        self.assertEqual(summarize_results(results), {"skipped_not_wav": 1})

    def test_skips_outside_project_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as outside_temp:
            outside_target = Path(outside_temp) / "outside.wav"
            outside_target.write_bytes(b"wav")
            decisions_path = self.project_root / "evaluation_results" / "outside_decisions.json"
            with open(decisions_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "version": 1,
                        "decisions": [
                            {
                                "filepath": str(outside_target),
                                "label": "a",
                                "prediction": "i",
                                "confidence": 0.97,
                                "decision": "delete_candidate",
                            }
                        ],
                    },
                    f,
                )

            results = delete_review_candidates(
                decisions_path,
                project_root=self.project_root,
            )

            self.assertTrue(outside_target.exists())
            self.assertEqual(summarize_results(results), {"skipped_outside_project": 1})

    def test_write_delete_log(self) -> None:
        results = delete_review_candidates(
            self.decisions_path,
            project_root=self.project_root,
        )
        log_path = self.project_root / "evaluation_results" / "delete_log.json"
        write_delete_log(
            log_path,
            results=results,
            execute=False,
            decisions_path=self.decisions_path,
        )

        payload = json.loads(log_path.read_text(encoding="utf-8"))
        self.assertFalse(payload["execute"])
        self.assertEqual(payload["summary"], {"would_delete": 1, "missing": 1})


if __name__ == "__main__":
    unittest.main()
