import tempfile
import unittest
from pathlib import Path

import numpy as np

from script.measure_audio_level import (
    CalibrationResult,
    FrameLevels,
    calculate_calibration,
    update_config_file,
)


class TestMeasureAudioLevel(unittest.TestCase):
    def test_calibration_places_vad_thresholds_between_noise_and_speech(self):
        noise = FrameLevels(
            rms=np.full(30, 0.001),
            peak=np.full(30, 0.002),
            db=np.full(30, -60.0),
        )
        speech = FrameLevels(
            rms=np.full(30, 0.05),
            peak=np.full(30, 0.1),
            db=np.full(30, -26.0),
        )

        result = calculate_calibration(noise, speech)

        self.assertGreater(result.vad_rms_threshold, 0.001)
        self.assertLess(result.vad_rms_threshold, 0.05)
        self.assertGreater(result.vad_silence_threshold, 0.002)
        self.assertLess(result.vad_silence_threshold, 0.1)
        self.assertGreater(result.top_db, 10.0)
        self.assertGreater(result.snr_db, 10.0)

    def test_update_config_file_rewrites_preprocess_values(self):
        config_text = """from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessConfig:
    top_db: float = 20
    vad_silence_threshold: float = 0.002
    vad_rms_threshold: float = 0.008
    min_top_db: float = 36.0
    max_top_db: float = 52.0
"""
        result = CalibrationResult(
            top_db=31.2,
            min_top_db=23.2,
            max_top_db=39.2,
            vad_silence_threshold=0.004321,
            vad_rms_threshold=0.001234,
            snr_db=22.0,
            active_speech_ratio=1.0,
            noise_rms_p95=0.001,
            noise_peak_p99=0.002,
            speech_rms_p20=0.05,
            speech_peak_p20=0.1,
            warnings=(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.py"
            config_path.write_text(config_text, encoding="utf-8")

            update_config_file(config_path, result)

            updated = config_path.read_text(encoding="utf-8")

        self.assertIn("top_db: float = 31.2", updated)
        self.assertIn("vad_silence_threshold: float = 0.004321", updated)
        self.assertIn("vad_rms_threshold: float = 0.001234", updated)
        self.assertIn("min_top_db: float = 23.2", updated)
        self.assertIn("max_top_db: float = 39.2", updated)


if __name__ == "__main__":
    unittest.main()
