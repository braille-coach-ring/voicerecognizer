"""評価 HTML レポートの安全性・整合性テスト。

レポートには index.csv 由来のファイルパスや話者名がそのまま入るため、
エスケープを怠るとレポートを開いた時点で任意のマークアップが実行される。
"""

import unittest

from voicerecognizer.evaluation.evaluator import (
    MISCLASSIFIED_DISPLAY_LIMIT,
    compute_evaluation_result,
    generate_html_report,
)


class TestHtmlEscaping(unittest.TestCase):
    def _report_with(self, filepath: str, speaker: str) -> str:
        result = compute_evaluation_result(
            ["a"],
            ["i"],
            labels=("a", "i"),
            filepaths=[filepath],
            confidences=[0.5],
            speakers=[speaker],
        )
        return generate_html_report(result)

    def test_script_tag_in_filepath_is_escaped(self) -> None:
        document = self._report_with("dataset/<script>alert(1)</script>.wav", "pc_1")
        self.assertNotIn("<script>alert(1)</script>", document)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", document)

    def test_quote_in_filepath_cannot_break_out_of_attribute(self) -> None:
        document = self._report_with('dataset/x" onerror="alert(1)".wav', "pc_1")
        self.assertNotIn('onerror="alert(1)"', document)

    def test_script_tag_in_speaker_is_escaped(self) -> None:
        document = self._report_with("dataset/a/001.wav", "<img src=x onerror=alert(1)>")
        self.assertNotIn("<img src=x onerror=alert(1)>", document)


class TestMisclassifiedTableConsistency(unittest.TestCase):
    def test_filter_counts_match_displayed_rows_when_truncated(self) -> None:
        total = MISCLASSIFIED_DISPLAY_LIMIT + 20
        result = compute_evaluation_result(
            ["a"] * total,
            ["i"] * total,
            labels=("a", "i"),
            filepaths=[f"dataset/a/{i:04d}.wav" for i in range(total)],
            speakers=["pc_1"] * total,
        )
        document = generate_html_report(result)

        # 表の行数と、ボタンに出る件数が一致すること
        self.assertEqual(document.count('class="mis-row"'), MISCLASSIFIED_DISPLAY_LIMIT)
        self.assertIn(f"表示中 ({MISCLASSIFIED_DISPLAY_LIMIT})", document)
        self.assertIn(f"正解「a」 ({MISCLASSIFIED_DISPLAY_LIMIT})", document)
        # 打ち切りが起きたことが読み手に伝わること
        self.assertIn(f"全 {total} 件のうち上位", document)

    def test_no_truncation_note_when_everything_fits(self) -> None:
        result = compute_evaluation_result(
            ["a", "a"],
            ["i", "i"],
            labels=("a", "i"),
            filepaths=["dataset/a/1.wav", "dataset/a/2.wav"],
            speakers=["pc_1", "pc_1"],
        )
        document = generate_html_report(result)
        self.assertIn("表示中 (2)", document)
        self.assertNotIn("のうち上位", document)


class TestFilterScript(unittest.TestCase):
    def test_filter_uses_data_attributes_not_inline_onclick(self) -> None:
        result = compute_evaluation_result(
            ["a"], ["i"], labels=("a", "i"), filepaths=["dataset/a/1.wav"], speakers=["pc_1"]
        )
        document = generate_html_report(result)

        self.assertNotIn("onclick='filterCategory", document)
        self.assertNotIn("event.target", document)
        self.assertIn("data-category=", document)
        self.assertIn("addEventListener", document)


if __name__ == "__main__":
    unittest.main()
