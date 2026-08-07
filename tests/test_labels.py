import unittest

from config_labels import (
    ALL_HIRAGANA_LABELS,
    SEION_LABELS,
    DAKUON_LABELS,
    HANDAKUON_LABELS,
    YOON_LABELS,
    OTHER_LABELS,
    HIRAGANA_TO_ROMAJI,
    ROMAJI_TO_HIRAGANA,
)
from config import RecognitionConfig, DEFAULT_RECOGNITION_CONFIG


class TestLabelsSystem(unittest.TestCase):
    def test_all_hiragana_labels_total_count(self):
        """ALL_HIRAGANA_LABELS が合計 105 ラベルであることを検証"""
        self.assertEqual(len(ALL_HIRAGANA_LABELS), 105)

    def test_category_counts(self):
        """各カテゴリの要素数を検証"""
        self.assertEqual(len(SEION_LABELS), 46)
        self.assertEqual(len(DAKUON_LABELS), 20)
        self.assertEqual(len(HANDAKUON_LABELS), 5)
        self.assertEqual(len(YOON_LABELS), 33)
        self.assertEqual(len(OTHER_LABELS), 1)

    def test_no_duplicate_labels(self):
        """ラベルに重複がないことを検証"""
        self.assertEqual(len(ALL_HIRAGANA_LABELS), len(set(ALL_HIRAGANA_LABELS)))

    def test_hiragana_to_romaji_mapping(self):
        """ひらがな ➔ ローマ字の双方向変換を検証"""
        self.assertEqual(HIRAGANA_TO_ROMAJI["あ"], "a")
        self.assertEqual(HIRAGANA_TO_ROMAJI["が"], "ga")
        self.assertEqual(HIRAGANA_TO_ROMAJI["ぱ"], "pa")
        self.assertEqual(HIRAGANA_TO_ROMAJI["きゃ"], "kya")
        self.assertEqual(HIRAGANA_TO_ROMAJI["other"], "other")

    def test_romaji_to_hiragana_mapping(self):
        """ローマ字 ➔ ひらがなの双方向変換およびエイリアスを検証"""
        self.assertEqual(ROMAJI_TO_HIRAGANA["a"], "あ")
        self.assertEqual(ROMAJI_TO_HIRAGANA["ga"], "が")
        self.assertEqual(ROMAJI_TO_HIRAGANA["pa"], "ぱ")
        self.assertEqual(ROMAJI_TO_HIRAGANA["kya"], "きゃ")
        self.assertEqual(ROMAJI_TO_HIRAGANA["other"], "other")

        # 訓令式/ヘボン式エイリアステスト
        self.assertEqual(ROMAJI_TO_HIRAGANA["si"], "し")
        self.assertEqual(ROMAJI_TO_HIRAGANA["tu"], "つ")
        self.assertEqual(ROMAJI_TO_HIRAGANA["zi"], "じ")
        self.assertEqual(ROMAJI_TO_HIRAGANA["hu"], "ふ")

    def test_config_default_labels_integration(self):
        """DEFAULT_RECOGNITION_CONFIG の labels が ALL_HIRAGANA_LABELS と一致することを検証"""
        cfg = RecognitionConfig()
        self.assertEqual(len(cfg.labels), 105)
        self.assertEqual(cfg.labels, ALL_HIRAGANA_LABELS)
        self.assertEqual(DEFAULT_RECOGNITION_CONFIG.labels, ALL_HIRAGANA_LABELS)


if __name__ == "__main__":
    unittest.main()
