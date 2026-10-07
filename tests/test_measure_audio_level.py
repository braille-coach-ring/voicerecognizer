import tempfile
import unittest
from pathlib import Path

import numpy as np

from script.measure_audio_level import (
    CalibrationResult,
    Session,
    VadEvaluation,
    VadSettings,
    analyze_recording,
    build_session,
    calculate_calibration,
    calculate_window_levels,
    simulate_vad,
    update_config_file,
    update_env_file,
)
from voicerecognizer.config import PreprocessConfig
from voicerecognizer.runtime.vad import VoiceActivityDetector

SETTINGS = VadSettings(
    sample_rate=16000,
    window_seconds=1.0,
    hop_seconds=0.1,
    min_speech_chunks=2,
    min_active_ratio=0.02,
)


def synthesize_recording(
    noise_std: float,
    speech_amplitude: float,
    utterances: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, Session]:
    """Room noise plus one short voiced syllable shortly after each cue."""
    session = build_session(
        noise_seconds=5.0,
        utterances=utterances,
        cue_interval=2.5,
        window_seconds=SETTINGS.window_seconds,
    )
    sr = SETTINGS.sample_rate
    rng = np.random.default_rng(seed)
    waveform = rng.normal(0.0, noise_std, int(session.total_seconds * sr)).astype(np.float32)

    duration = 0.3
    t = np.arange(int(duration * sr)) / sr
    voiced = sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 6))
    syllable = speech_amplitude * np.hanning(t.size) * voiced / np.max(np.abs(voiced))
    for cue in session.cue_times:
        start = int((cue + 0.4) * sr)
        waveform[start : start + syllable.size] += syllable.astype(np.float32)
    return waveform, session


class TestMeasureAudioLevel(unittest.TestCase):
    def test_calibration_detects_every_utterance_without_noise_triggers(self):
        waveform, session = synthesize_recording(noise_std=0.002, speech_amplitude=0.1)
        recording = analyze_recording(waveform, session, SETTINGS)

        result = calculate_calibration(recording, SETTINGS)

        self.assertTrue(result.reliable, result.warnings)
        self.assertEqual(result.nominal, VadEvaluation(detected=8, voiced=8, false_triggers=0))
        self.assertEqual(result.robust, VadEvaluation(detected=8, voiced=8, false_triggers=0))
        noise_levels = recording.levels
        self.assertGreater(
            result.vad_silence_threshold, np.max(noise_levels.peak[recording.noise_mask])
        )
        self.assertGreater(result.vad_rms_threshold, np.max(noise_levels.rms[recording.noise_mask]))

    def test_thresholds_are_usable_by_voice_activity_detector(self):
        waveform, session = synthesize_recording(noise_std=0.002, speech_amplitude=0.1)
        recording = analyze_recording(waveform, session, SETTINGS)
        result = calculate_calibration(recording, SETTINGS)
        vad = VoiceActivityDetector(
            config=PreprocessConfig(vad_min_speech_chunks=SETTINGS.min_speech_chunks),
            silence_threshold=result.vad_silence_threshold,
            rms_threshold=result.vad_rms_threshold,
        )
        sr = SETTINGS.sample_rate
        window = int(SETTINGS.window_seconds * sr)
        hop = int(SETTINGS.hop_seconds * sr)

        fired_at: list[float] = []
        for start in range(0, waveform.size - window + 1, hop):
            if vad.is_speech(waveform[start : start + window]):
                fired_at.append((start + window) / sr)

        for cue in session.cue_times:
            self.assertTrue(any(cue < t < cue + session.cue_interval for t in fired_at))
        self.assertFalse(any(t <= session.noise_end for t in fired_at))

    def test_simulate_vad_matches_voice_activity_detector(self):
        waveform, _ = synthesize_recording(noise_std=0.003, speech_amplitude=0.05, utterances=3)
        levels = calculate_window_levels(waveform, 16000, 1.0, 0.1)
        sr = SETTINGS.sample_rate

        for peak_threshold, rms_threshold in ((0.02, 0.004), (0.04, 0.006), (0.01, 0.0035)):
            vad = VoiceActivityDetector(
                config=PreprocessConfig(vad_min_speech_chunks=2, vad_min_active_ratio=0.02),
                silence_threshold=peak_threshold,
                rms_threshold=rms_threshold,
                adaptive=False,
            )
            expected = [
                vad.is_speech(waveform[start : start + sr])
                for start in range(0, waveform.size - sr + 1, sr // 10)
            ]
            simulated = simulate_vad(
                levels, np.array([peak_threshold]), np.array([rms_threshold]), 2, 0.02
            )
            self.assertEqual(simulated[0, 0].tolist(), expected)

    def test_calibration_is_unreliable_when_speech_is_buried_in_noise(self):
        waveform, session = synthesize_recording(noise_std=0.02, speech_amplitude=0.01)
        recording = analyze_recording(waveform, session, SETTINGS)

        result = calculate_calibration(recording, SETTINGS)

        self.assertFalse(result.reliable)
        self.assertTrue(result.warnings)

    def test_build_session_rejects_overlapping_cues(self):
        with self.assertRaises(ValueError):
            build_session(noise_seconds=5.0, utterances=4, cue_interval=1.5, window_seconds=1.0)

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
        evaluation = VadEvaluation(detected=8, voiced=8, false_triggers=0)
        result = CalibrationResult(
            top_db=31.2,
            min_top_db=23.2,
            max_top_db=39.2,
            vad_silence_threshold=0.004321,
            vad_rms_threshold=0.001234,
            snr_db=22.0,
            nominal=evaluation,
            robust=evaluation,
            reliable=True,
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

    def test_update_env_file(self):
        evaluation = VadEvaluation(detected=8, voiced=8, false_triggers=0)
        result = CalibrationResult(
            top_db=31.2,
            min_top_db=23.2,
            max_top_db=39.2,
            vad_silence_threshold=0.087654,
            vad_rms_threshold=0.023456,
            snr_db=22.0,
            nominal=evaluation,
            robust=evaluation,
            reliable=True,
            warnings=(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("SOME_VAR=123\nVOICE_VAD_SILENCE_THRESHOLD=0.01\n", encoding="utf-8")

            update_env_file(env_path, result)
            updated = env_path.read_text(encoding="utf-8")

        self.assertIn("SOME_VAR=123", updated)
        self.assertIn("VOICE_VAD_SILENCE_THRESHOLD=0.087654", updated)
        self.assertIn("VOICE_VAD_RMS_THRESHOLD=0.023456", updated)


if __name__ == "__main__":
    unittest.main()
