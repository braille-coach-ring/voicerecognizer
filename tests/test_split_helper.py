import unittest

from voicerecognizer.utils.split_helper import safe_group_split, safe_stratified_split


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

    def test_group_split_keeps_speakers_disjoint(self):
        labels = ["a", "i"] * 6
        groups = ["speaker1"] * 4 + ["speaker2"] * 4 + ["speaker3"] * 4

        train_idx, val_idx = safe_group_split(labels, groups, val_rate=0.34, seed=42)

        train_groups = {groups[index] for index in train_idx}
        val_groups = {groups[index] for index in val_idx}

        self.assertTrue(train_idx)
        self.assertTrue(val_idx)
        self.assertTrue(train_groups.isdisjoint(val_groups))

    def test_group_split_falls_back_when_speakers_are_unknown(self):
        labels = ["a"] * 10 + ["i"] * 10
        groups = ["unknown"] * len(labels)

        train_idx, val_idx = safe_group_split(labels, groups, val_rate=0.2, seed=42)

        self.assertEqual(len(train_idx) + len(val_idx), len(labels))
        self.assertEqual(len(val_idx), 4)


if __name__ == "__main__":
    unittest.main()
