"""export_onnx のラベル整合性ガードのテスト。

labels.json とチェックポイントの分類ヘッドのクラス数が食い違ったまま
ONNX を書き出すと、推論は成功するのに結果だけ不正になる。
その経路が必ず例外で止まることを保証する。
"""

import json
import tempfile
import unittest
from pathlib import Path

from voicerecognizer.models.wav2vec2.export_onnx import (
    _read_labels_file,
    _verify_label_count,
)


class TestReadLabelsFile(unittest.TestCase):
    def test_returns_labels_for_valid_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            labels_file = Path(tmp) / "labels.json"
            labels_file.write_text(json.dumps(["a", "i", "u"]), encoding="utf-8")
            self.assertEqual(_read_labels_file(labels_file), ["a", "i", "u"])

    def test_returns_none_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(_read_labels_file(Path(tmp) / "labels.json"))

    def test_returns_none_for_broken_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            labels_file = Path(tmp) / "labels.json"
            labels_file.write_text("{not json", encoding="utf-8")
            self.assertIsNone(_read_labels_file(labels_file))

    def test_returns_none_for_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            labels_file = Path(tmp) / "labels.json"
            labels_file.write_text("[]", encoding="utf-8")
            self.assertIsNone(_read_labels_file(labels_file))

    def test_returns_none_for_non_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            labels_file = Path(tmp) / "labels.json"
            labels_file.write_text(json.dumps({"a": 0}), encoding="utf-8")
            self.assertIsNone(_read_labels_file(labels_file))


class TestVerifyLabelCount(unittest.TestCase):
    def test_passes_when_counts_match(self) -> None:
        _verify_label_count(["a", "i"], 2, Path("weights/wav2vec2_best"), restored=False)

    def test_raises_when_labels_json_disagrees_with_checkpoint(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _verify_label_count(["a", "i", "u"], 105, Path("weights/wav2vec2_best"), restored=False)
        msg = str(ctx.exception)
        self.assertIn("3 クラス", msg)
        self.assertIn("105 クラス", msg)
        self.assertIn("labels.json", msg)

    def test_raises_when_restored_labels_disagree_with_checkpoint(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _verify_label_count(["a"] * 105, 86, Path("weights/wav2vec2_best"), restored=True)
        self.assertIn("復元した", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
