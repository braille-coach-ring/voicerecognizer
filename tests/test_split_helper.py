import unittest

from voicerecognizer.utils.split_helper import safe_stratified_split


class TestSplitHelper(unittest.TestCase):
    def test_all_classes_with_enough_samples(self):
        # 10 samples for 'a', 10 samples for 'i'
        labels = ["a"] * 10 + ["i"] * 10
        train_idx, val_idx = safe_stratified_split(labels, val_rate=0.2, seed=42)

        self.assertEqual(len(train_idx) + len(val_idx), 20)
        self.assertEqual(len(val_idx), 4)

    def test_singletons_placed_in_train_set(self):
        # 'a' has 10, 'i' has 10, but 'kya' and 'ga' have only 1 sample each
        labels = ["a"] * 10 + ["i"] * 10 + ["kya", "ga"]
        train_idx, val_idx = safe_stratified_split(labels, val_rate=0.2, seed=42)

        # Singletons ('kya', 'ga') must be in train_idx
        kya_idx = labels.index("kya")
        ga_idx = labels.index("ga")

        self.assertIn(kya_idx, train_idx)
        self.assertIn(ga_idx, train_idx)
        self.assertNotIn(kya_idx, val_idx)
        self.assertNotIn(ga_idx, val_idx)


if __name__ == "__main__":
    unittest.main()
