import unittest
import numpy as np

from config import PreprocessConfig
from preprocessing.threshold_calculator import (
    AdaptiveSilenceThresholdCalculator,
    FixedSilenceThresholdCalculator,
    create_threshold_calculator,
)


class TestThresholdCalculator(unittest.TestCase):
    def test_fixed_threshold_calculator(self):
        calc = FixedSilenceThresholdCalculator(top_db=25.0)
        self.assertEqual(calc.get_silence_threshold(), 25.0)

        # 音声データを渡しても変化しないこと
        dummy_audio = np.random.normal(0, 0.1, 16000)
        calc.update(dummy_audio)
        self.assertEqual(calc.get_silence_threshold(), 25.0)

    def test_adaptive_threshold_calculator(self):
        config = PreprocessConfig(
            top_db=30,
            dynamic_threshold_enabled=True,
            min_top_db=15,
            max_top_db=40,
            noise_update_rate=0.1,
        )
        calc = AdaptiveSilenceThresholdCalculator(config=config)
        initial_threshold = calc.get_silence_threshold()
        self.assertGreaterEqual(initial_threshold, 15.0)

        # 静かな音声（小さなノイズ）を流し込む
        quiet_audio = np.random.normal(0, 0.001, 16000)
        for _ in range(5):
            calc.update(quiet_audio)

        updated_threshold = calc.get_silence_threshold()
        # 閾値がガード（15〜40）の範囲内に収まっていること
        self.assertTrue(15.0 <= updated_threshold <= 40.0)

    def test_factory_function(self):
        fixed_config = PreprocessConfig(dynamic_threshold_enabled=False, top_db=30)
        calc_fixed = create_threshold_calculator(fixed_config)
        self.assertIsInstance(calc_fixed, FixedSilenceThresholdCalculator)

        adaptive_config = PreprocessConfig(dynamic_threshold_enabled=True)
        calc_adaptive = create_threshold_calculator(adaptive_config)
        self.assertIsInstance(calc_adaptive, AdaptiveSilenceThresholdCalculator)


if __name__ == "__main__":
    unittest.main()
