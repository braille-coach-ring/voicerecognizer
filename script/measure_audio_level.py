"""
Audio input level calibration utility.

The calibration records one continuous stream with two phases:
1. Room noise while you stay silent.
2. Short single-syllable utterances, one per on-screen cue (the way main.py is used).

The VAD in main.py judges a sliding ``window_seconds`` buffer (1 s by default) every
``chunk_seconds``, using the window's peak, RMS and active-sample ratio, and it needs
``vad_min_speech_chunks`` consecutive hits. This script measures exactly those window
features, replays the VAD over a grid of thresholds, and picks the pair that detects
every utterance without firing on room noise, also under the adaptive 0.8x-1.25x
threshold range of the VAD. The result is written back to config.py by default.
"""

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep

import numpy as np

from voicerecognizer.config import (
    DEFAULT_AUDIO_CONFIG,
    DEFAULT_PREPROCESS_CONFIG,
    PROJECT_ROOT,
)

logger = logging.getLogger(__name__)

EPSILON = 1e-10
DEFAULT_NOISE_SECONDS = 5.0
DEFAULT_UTTERANCES = 8
DEFAULT_CUE_INTERVAL_SECONDS = 2.5
WARMUP_DISCARD_SECONDS = 0.5
SPEECH_LEAD_SECONDS = 1.0
# Windows ending this soon after a cue cannot contain the new utterance yet.
CUE_CREDIT_DELAY_SECONDS = 0.15
# Bounds of the adaptive threshold clamp in VoiceActivityDetector.is_speech.
ADAPTIVE_LOW_SCALE = 0.80
ADAPTIVE_HIGH_SCALE = 1.25
GRID_POINTS = 60
VOICED_NOISE_RATIO = 1.2
MIN_UTTERANCES = 3
MIN_DETECTION_RATIO = 0.8
CUE_SYLLABLES = ("あ", "か", "さ", "た", "な", "は", "ま", "ら", "い", "う", "え", "お")
DEFAULT_AUDIO_FILE = PROJECT_ROOT / "measured_audio.wav"
DEFAULT_CHART_FILE = PROJECT_ROOT / "Docs" / "charts" / "audio_level_measurement.png"


@dataclass(frozen=True)
class FrameLevels:
    rms: np.ndarray
    peak: np.ndarray
    db: np.ndarray


@dataclass(frozen=True)
class WindowLevels:
    """Features of the sliding windows that VoiceActivityDetector.is_speech receives."""

    peak: np.ndarray
    rms: np.ndarray
    sorted_abs: np.ndarray
    end_times: np.ndarray
    window_seconds: float

    def active_ratio(self, thresholds: np.ndarray) -> np.ndarray:
        """Ratio of samples with ``abs >= threshold``, shaped (thresholds, windows)."""
        thresholds = np.asarray(thresholds, dtype=np.float64)
        n = self.sorted_abs.shape[1]
        below = np.stack([np.searchsorted(row, thresholds, side="left") for row in self.sorted_abs])
        return ((n - below) / n).T


@dataclass(frozen=True)
class Session:
    """Timeline of one calibration recording, in seconds from the recording start."""

    noise_start: float
    noise_end: float
    cue_times: tuple[float, ...]
    cue_interval: float

    @property
    def total_seconds(self) -> float:
        return self.cue_times[-1] + self.cue_interval


@dataclass(frozen=True)
class VadSettings:
    sample_rate: int
    window_seconds: float
    hop_seconds: float
    min_speech_chunks: int
    min_active_ratio: float


@dataclass(frozen=True)
class Recording:
    """Measured levels of one calibration recording."""

    levels: WindowLevels
    noise_mask: np.ndarray
    cue_masks: tuple[np.ndarray, ...]
    voiced: np.ndarray
    noise_frames: FrameLevels
    speech_frames: FrameLevels
    max_abs: float


@dataclass(frozen=True)
class VadEvaluation:
    detected: int
    voiced: int
    false_triggers: int


@dataclass(frozen=True)
class CalibrationResult:
    top_db: float
    min_top_db: float
    max_top_db: float
    vad_silence_threshold: float
    vad_rms_threshold: float
    snr_db: float
    nominal: VadEvaluation
    robust: VadEvaluation
    reliable: bool
    warnings: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure microphone noise/speech levels and update config.py."
    )
    parser.add_argument(
        "--noise-seconds",
        type=float,
        default=DEFAULT_NOISE_SECONDS,
        help="Seconds to record room noise.",
    )
    parser.add_argument(
        "--utterances",
        type=int,
        default=DEFAULT_UTTERANCES,
        help="Number of single-syllable utterances to record.",
    )
    parser.add_argument(
        "--cue-interval",
        type=float,
        default=DEFAULT_CUE_INTERVAL_SECONDS,
        help="Seconds between utterance cues.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional sounddevice input device id or name.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=PROJECT_ROOT / "src" / "voicerecognizer" / "config.py",
        help="config.py path to update.",
    )
    parser.add_argument(
        "--audio-path",
        type=Path,
        default=DEFAULT_AUDIO_FILE,
        help="Path where the calibration audio is saved.",
    )
    parser.add_argument(
        "--chart-path",
        type=Path,
        default=DEFAULT_CHART_FILE,
        help="Path where the level chart is saved.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print measured values without updating config.py.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Update config.py even when the calibration looks unreliable.",
    )
    parser.add_argument(
        "--skip-chart",
        action="store_true",
        help="Skip chart generation.",
    )
    parser.add_argument(
        "--countdown",
        type=int,
        default=3,
        help="Countdown seconds before recording starts.",
    )
    return parser


def countdown(seconds: int) -> None:
    for remaining in range(max(0, seconds), 0, -1):
        print(remaining)
        sleep(1)


def build_session(
    noise_seconds: float,
    utterances: int,
    cue_interval: float,
    window_seconds: float,
) -> Session:
    if noise_seconds < window_seconds:
        raise ValueError(f"--noise-seconds must be at least the VAD window ({window_seconds}s).")
    if utterances < 1:
        raise ValueError("--utterances must be at least 1.")
    # A window must not hold the previous utterance once the next cue's windows are credited.
    min_interval = window_seconds + 1.5
    if cue_interval < min_interval:
        raise ValueError(f"--cue-interval must be at least {min_interval:.1f}s.")

    noise_start = WARMUP_DISCARD_SECONDS
    noise_end = noise_start + noise_seconds
    first_cue = noise_end + SPEECH_LEAD_SECONDS
    return Session(
        noise_start=noise_start,
        noise_end=noise_end,
        cue_times=tuple(first_cue + index * cue_interval for index in range(utterances)),
        cue_interval=cue_interval,
    )


def record_session(
    session: Session,
    sample_rate: int,
    channels: int,
    device: str | None,
    countdown_seconds: int,
) -> np.ndarray:
    """Record both phases as one stream, printing the cues while recording."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise ImportError(
            "sounddevice is required to record calibration audio. "
            "Install dependencies with: uv sync"
        ) from exc

    print("\nPhase 1: stay silent. Do not type or touch the microphone.")
    countdown(countdown_seconds)
    audio = sd.rec(
        int(session.total_seconds * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        device=device,
    )
    started = monotonic()

    def wait_until(seconds: float) -> None:
        remaining = seconds - (monotonic() - started)
        if remaining > 0:
            sleep(remaining)

    print("Recording room noise...")
    wait_until(session.noise_end)
    print(
        "\nPhase 2: say each syllable ONCE, right after it appears, "
        "at the volume you use with main.py."
    )
    for index, cue_time in enumerate(session.cue_times):
        wait_until(cue_time)
        syllable = CUE_SYLLABLES[index % len(CUE_SYLLABLES)]
        print(f"  [{index + 1}/{len(session.cue_times)}] Say: 「{syllable}」")
    sd.wait()
    print("Done.")
    return np.asarray(audio, dtype=np.float32)[:, 0].copy()


def save_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError(
            "soundfile is required to save measured_audio.wav. Install dependencies with: uv sync"
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate)
    logger.info("Saved calibration audio: %s", path)


def calculate_frame_levels(
    waveform: np.ndarray,
    sample_rate: int,
    chunk_seconds: float,
) -> FrameLevels:
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    waveform = waveform[np.isfinite(waveform)]
    if waveform.size == 0:
        raise ValueError("Recorded audio is empty.")

    frame_size = max(1, int(sample_rate * chunk_seconds))
    rms_values: list[float] = []
    peak_values: list[float] = []
    db_values: list[float] = []

    for start in range(0, len(waveform), frame_size):
        chunk = waveform[start : start + frame_size]
        if chunk.size == 0:
            continue

        rms = float(np.sqrt(np.mean(chunk**2)))
        peak = float(np.max(np.abs(chunk)))
        db = float(20.0 * np.log10(rms + EPSILON))
        rms_values.append(rms)
        peak_values.append(peak)
        db_values.append(db)

    return FrameLevels(
        rms=np.asarray(rms_values, dtype=np.float64),
        peak=np.asarray(peak_values, dtype=np.float64),
        db=np.asarray(db_values, dtype=np.float64),
    )


def calculate_window_levels(
    waveform: np.ndarray,
    sample_rate: int,
    window_seconds: float,
    hop_seconds: float,
) -> WindowLevels:
    """Slide a VAD-sized window over the recording, like AudioCapture.capture_once polling."""
    waveform = np.nan_to_num(np.asarray(waveform, dtype=np.float32).reshape(-1))
    window = max(1, int(sample_rate * window_seconds))
    hop = max(1, int(sample_rate * hop_seconds))
    if waveform.size < window:
        waveform = np.pad(waveform, (window - waveform.size, 0))

    starts = np.arange(0, waveform.size - window + 1, hop)
    peaks = np.empty(starts.size, dtype=np.float64)
    rms = np.empty(starts.size, dtype=np.float64)
    sorted_abs = np.empty((starts.size, window), dtype=np.float32)
    for index, start in enumerate(starts):
        chunk = waveform[start : start + window]
        abs_chunk = np.abs(chunk)
        peaks[index] = float(np.max(abs_chunk))
        rms[index] = float(np.sqrt(np.mean(chunk**2)))
        sorted_abs[index] = np.sort(abs_chunk)

    return WindowLevels(
        peak=peaks,
        rms=rms,
        sorted_abs=sorted_abs,
        end_times=(starts + window) / sample_rate,
        window_seconds=window / sample_rate,
    )


def simulate_vad(
    levels: WindowLevels,
    peak_thresholds: np.ndarray,
    rms_thresholds: np.ndarray,
    min_speech_chunks: int,
    min_active_ratio: float,
) -> np.ndarray:
    """
    Replay VoiceActivityDetector.is_speech (non-adaptive) for every threshold pair.

    Returns a bool array shaped (peak thresholds, rms thresholds, windows).
    """
    peak_thresholds = np.asarray(peak_thresholds, dtype=np.float64)
    rms_thresholds = np.asarray(rms_thresholds, dtype=np.float64)
    peak_ok = (levels.peak[None, :] >= peak_thresholds[:, None]) & (
        levels.active_ratio(peak_thresholds) >= min_active_ratio
    )
    rms_ok = levels.rms[None, :] >= rms_thresholds[:, None]
    passes = peak_ok[:, None, :] & rms_ok[None, :, :]

    fires = np.zeros_like(passes)
    streak = np.zeros(passes.shape[:2], dtype=np.int64)
    for index in range(passes.shape[2]):
        streak = np.where(passes[:, :, index], streak + 1, 0)
        fires[:, :, index] = streak >= max(1, min_speech_chunks)
    return fires


def percentile(values: np.ndarray, q: float, minimum: float = EPSILON) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return minimum
    return max(float(np.percentile(finite, q)), minimum)


def choose_threshold(
    noise_high: float,
    speech_low: float,
    minimum: float,
    noise_multiplier: float,
    speech_fraction: float,
) -> float:
    lower = max(noise_high * noise_multiplier, minimum)
    upper = max(speech_low * speech_fraction, minimum)
    if upper > lower:
        return float(np.clip(np.sqrt(lower * upper), lower, upper))
    return lower


def db_ratio(signal: float, noise: float) -> float:
    return float(20.0 * np.log10((signal + EPSILON) / (noise + EPSILON)))


def clamp(value: float, lower: float, upper: float) -> float:
    return float(min(max(value, lower), upper))


def nth_highest(values: np.ndarray, n: int) -> float:
    if values.size == 0:
        return 0.0
    ordered = np.sort(values)[::-1]
    return float(ordered[min(max(1, n), ordered.size) - 1])


def log_margin(thresholds: np.ndarray, noise_high: float, speech_low: float) -> np.ndarray:
    """Log distance of each threshold from the nearer of the noise and speech levels."""
    log_t = np.log(np.maximum(thresholds, EPSILON))
    return np.minimum(
        log_t - np.log(max(noise_high, EPSILON)),
        np.log(max(speech_low, EPSILON)) - log_t,
    )


def analyze_recording(waveform: np.ndarray, session: Session, settings: VadSettings) -> Recording:
    waveform = np.nan_to_num(np.asarray(waveform, dtype=np.float32).reshape(-1))
    sr = settings.sample_rate
    levels = calculate_window_levels(waveform, sr, settings.window_seconds, settings.hop_seconds)

    ends = levels.end_times
    tolerance = 1e-6
    noise_mask = (ends >= session.noise_start + levels.window_seconds - tolerance) & (
        ends <= session.noise_end + tolerance
    )
    if not noise_mask.any():
        raise ValueError("The noise phase is shorter than one VAD window.")
    cue_masks = tuple(
        (ends >= cue + CUE_CREDIT_DELAY_SECONDS) & (ends < cue + session.cue_interval)
        for cue in session.cue_times
    )

    noise_frames = calculate_frame_levels(
        waveform[int(session.noise_start * sr) : int(session.noise_end * sr)],
        sr,
        settings.hop_seconds,
    )
    slots = [
        waveform[int(cue * sr) : int((cue + session.cue_interval) * sr)]
        for cue in session.cue_times
    ]
    # A cue slot counts as an utterance only if two 100 ms frames rise above the loudest
    # room-noise frame (two, so that a single click is not mistaken for speech).
    frame_gate = max(float(np.max(noise_frames.rms)) * VOICED_NOISE_RATIO, 1e-5)
    voiced = np.array(
        [
            nth_highest(calculate_frame_levels(slot, sr, settings.hop_seconds).rms, 2) >= frame_gate
            for slot in slots
        ]
    )

    return Recording(
        levels=levels,
        noise_mask=noise_mask,
        cue_masks=cue_masks,
        voiced=voiced,
        noise_frames=noise_frames,
        speech_frames=calculate_frame_levels(np.concatenate(slots), sr, settings.hop_seconds),
        max_abs=float(np.max(np.abs(waveform))),
    )


def count_detections(fires: np.ndarray, recording: Recording) -> tuple[np.ndarray, np.ndarray]:
    """Return (detected utterances, noise windows that fired) for each threshold pair."""
    detected = np.zeros(fires.shape[:-1], dtype=np.int64)
    for mask, is_voiced in zip(recording.cue_masks, recording.voiced, strict=True):
        if is_voiced and mask.any():
            detected += fires[..., mask].any(axis=-1)
    false_triggers = fires[..., recording.noise_mask].sum(axis=-1)
    return detected, false_triggers


def evaluate_thresholds(
    recording: Recording,
    settings: VadSettings,
    silence_threshold: float,
    rms_threshold: float,
) -> VadEvaluation:
    """Evaluate one fixed threshold pair (e.g. the current config) on the recording."""
    fires = simulate_vad(
        recording.levels,
        np.array([silence_threshold]),
        np.array([rms_threshold]),
        settings.min_speech_chunks,
        settings.min_active_ratio,
    )
    detected, false_triggers = count_detections(fires, recording)
    return VadEvaluation(
        detected=int(detected[0, 0]),
        voiced=int(np.sum(recording.voiced)),
        false_triggers=int(false_triggers[0, 0]),
    )


def calculate_trim_db(
    noise_frames: FrameLevels,
    speech_frames: FrameLevels,
) -> tuple[float, float, float, float]:
    """Return (top_db, min_top_db, max_top_db, snr_db) from 100 ms frame levels."""
    noise_rms_p95 = percentile(noise_frames.rms, 95)
    noise_peak_p95 = percentile(noise_frames.peak, 95)

    speech_gate_rms = max(noise_rms_p95 * 3.0, 1e-6)
    speech_gate_peak = max(noise_peak_p95 * 3.0, 1e-5)
    active_mask = (speech_frames.rms >= speech_gate_rms) | (speech_frames.peak >= speech_gate_peak)
    if int(np.sum(active_mask)) < 3:
        active_mask = speech_frames.rms >= percentile(speech_frames.rms, 80)

    active_rms = speech_frames.rms[active_mask]
    if active_rms.size == 0:
        active_rms = speech_frames.rms

    speech_rms_p20 = percentile(active_rms, 20)
    speech_rms_p50 = percentile(active_rms, 50)
    speech_rms_p95 = percentile(active_rms, 95)

    trim_floor_rms = choose_threshold(
        noise_high=noise_rms_p95,
        speech_low=speech_rms_p20,
        minimum=1e-6,
        noise_multiplier=1.2,
        speech_fraction=0.45,
    )
    speech_reference_rms = max(speech_rms_p95, speech_rms_p50, trim_floor_rms * 2.0)
    top_db = clamp(db_ratio(speech_reference_rms, trim_floor_rms), 10.0, 80.0)
    min_top_db = clamp(top_db - 8.0, 5.0, top_db)
    max_top_db = clamp(top_db + 8.0, top_db, 80.0)
    snr_db = db_ratio(speech_rms_p50, noise_rms_p95)
    return top_db, min_top_db, max_top_db, snr_db


def calculate_calibration(recording: Recording, settings: VadSettings) -> CalibrationResult:
    levels = recording.levels
    noise_mask = recording.noise_mask
    voiced_masks = [
        mask
        for mask, is_voiced in zip(recording.cue_masks, recording.voiced, strict=True)
        if is_voiced
    ]

    # Level each utterance reaches in at least min_speech_chunks windows (what the VAD needs).
    utterance_peaks = [
        nth_highest(levels.peak[m], settings.min_speech_chunks) for m in voiced_masks
    ]
    utterance_rms = [nth_highest(levels.rms[m], settings.min_speech_chunks) for m in voiced_masks]
    noise_peak_high = float(np.max(levels.peak[noise_mask]))
    noise_rms_high = float(np.max(levels.rms[noise_mask]))
    speech_peak_low = min(utterance_peaks, default=noise_peak_high)
    speech_rms_low = min(utterance_rms, default=noise_rms_high)

    def grid(noise_values: np.ndarray, speech_high: float, minimum: float) -> np.ndarray:
        low = max(float(np.min(noise_values)) * 0.5, minimum)
        return np.geomspace(low, max(speech_high, low * 10.0), GRID_POINTS)

    peak_grid = grid(levels.peak[noise_mask], max(utterance_peaks, default=0.0), 1e-4)
    rms_grid = grid(levels.rms[noise_mask], max(utterance_rms, default=0.0), 1e-6)

    def replay(scale: float) -> tuple[np.ndarray, np.ndarray]:
        fires = simulate_vad(
            levels,
            peak_grid * scale,
            rms_grid * scale,
            settings.min_speech_chunks,
            settings.min_active_ratio,
        )
        return count_detections(fires, recording)

    nominal_detected, nominal_false = replay(1.0)
    # The adaptive VAD moves thresholds within 0.8x-1.25x: lower means more false
    # triggers, higher means more misses. Check both extremes.
    _, robust_false = replay(ADAPTIVE_LOW_SCALE)
    robust_detected, _ = replay(ADAPTIVE_HIGH_SCALE)

    nominal_score = nominal_detected - nominal_false
    robust_score = robust_detected - robust_false
    candidates = nominal_score == nominal_score.max()
    candidates &= robust_score == robust_score[candidates].max()
    # Among the equally good pairs, stay as far as possible from both noise and speech.
    margin = (
        log_margin(peak_grid, noise_peak_high, speech_peak_low)[:, None]
        + log_margin(rms_grid, noise_rms_high, speech_rms_low)[None, :]
    )
    flat_index = int(np.argmax(np.where(candidates, margin, -np.inf)))
    best_peak, best_rms = np.unravel_index(flat_index, margin.shape)

    voiced_count = int(np.sum(recording.voiced))
    nominal = VadEvaluation(
        detected=int(nominal_detected[best_peak, best_rms]),
        voiced=voiced_count,
        false_triggers=int(nominal_false[best_peak, best_rms]),
    )
    robust = VadEvaluation(
        detected=int(robust_detected[best_peak, best_rms]),
        voiced=voiced_count,
        false_triggers=int(robust_false[best_peak, best_rms]),
    )
    top_db, min_top_db, max_top_db, snr_db = calculate_trim_db(
        recording.noise_frames, recording.speech_frames
    )

    warnings: list[str] = []
    missing = len(recording.cue_masks) - voiced_count
    if missing:
        warnings.append(
            f"No speech was found after {missing} cue(s). Say each syllable right after its cue."
        )
    if voiced_count < MIN_UTTERANCES:
        warnings.append(
            f"Fewer than {MIN_UTTERANCES} utterances were recorded, so the thresholds are unreliable."
        )
    if nominal.false_triggers:
        warnings.append(
            "No threshold rejects all room noise while detecting speech. "
            "Calibrate again in a quieter room or speak closer to the mic."
        )
    if nominal.detected < voiced_count:
        warnings.append(
            f"Only {nominal.detected}/{voiced_count} utterances can be detected without "
            "triggering on noise. Speak louder or closer to the mic."
        )
    if robust.false_triggers or robust.detected < voiced_count:
        warnings.append(
            "Speech and noise are close: the adaptive VAD range (0.8x-1.25x) may cause "
            "occasional misses or false triggers."
        )
    if recording.max_abs > 0.98:
        warnings.append("The recording is close to clipping. Lower the microphone gain.")

    reliable = (
        voiced_count >= MIN_UTTERANCES
        and nominal.false_triggers == 0
        and nominal.detected >= int(np.ceil(voiced_count * MIN_DETECTION_RATIO))
    )

    return CalibrationResult(
        top_db=round(top_db, 1),
        min_top_db=round(min_top_db, 1),
        max_top_db=round(max_top_db, 1),
        vad_silence_threshold=round(float(peak_grid[best_peak]), 6),
        vad_rms_threshold=round(float(rms_grid[best_rms]), 6),
        snr_db=round(snr_db, 2),
        nominal=nominal,
        robust=robust,
        reliable=reliable,
        warnings=tuple(warnings),
    )


def format_config_float(value: float) -> str:
    text = f"{value:.6f}" if abs(value) < 1.0 else f"{value:.1f}"
    text = text.rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        return "0.0"
    if text.startswith("."):
        return f"0{text}"
    return text


def replace_class_defaults(
    text: str,
    class_name: str,
    values: dict[str, str],
) -> str:
    lines = text.splitlines(keepends=True)
    class_line_index = None
    for index, line in enumerate(lines):
        if re.match(rf"^class\s+{re.escape(class_name)}\b", line):
            class_line_index = index
            break

    if class_line_index is None:
        raise RuntimeError(f"{class_name} was not found in config.py")

    end_index = len(lines)
    for index in range(class_line_index + 1, len(lines)):
        if lines[index].strip() and not lines[index].startswith((" ", "\t")):
            end_index = index
            break

    found: set[str] = set()
    for index in range(class_line_index + 1, end_index):
        raw_line = lines[index]
        if raw_line.endswith("\r\n"):
            newline = "\r\n"
        elif raw_line.endswith("\n"):
            newline = "\n"
        else:
            newline = ""
        content = raw_line[: -len(newline)] if newline else raw_line

        for field_name, new_value in values.items():
            match = re.match(
                rf"^(\s*{re.escape(field_name)}\s*:[^=]+=\s*)([^#]*?)(\s*(?:#.*)?)$",
                content,
            )
            if match:
                lines[index] = f"{match.group(1)}{new_value}{match.group(3)}{newline}"
                found.add(field_name)
                break

    missing = set(values) - found
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise RuntimeError(f"{class_name} fields were not found: {missing_names}")

    return "".join(lines)


def update_config_file(config_path: Path, result: CalibrationResult) -> None:
    text = config_path.read_text(encoding="utf-8")
    preprocess_values = {
        "top_db": format_config_float(result.top_db),
        "vad_silence_threshold": format_config_float(result.vad_silence_threshold),
        "vad_rms_threshold": format_config_float(result.vad_rms_threshold),
        "min_top_db": format_config_float(result.min_top_db),
        "max_top_db": format_config_float(result.max_top_db),
    }

    text = replace_class_defaults(text, "PreprocessConfig", preprocess_values)
    config_path.write_text(text, encoding="utf-8")


def format_evaluation(evaluation: VadEvaluation) -> str:
    return (
        f"detected {evaluation.detected}/{evaluation.voiced} utterances, "
        f"{evaluation.false_triggers} false trigger(s) on room noise"
    )


def print_levels(recording: Recording) -> None:
    levels = recording.levels
    noise_mask = recording.noise_mask
    print(f"\nVAD window levels ({levels.window_seconds:.1f}s window, as main.py sees them)")
    print(
        f"  Room noise  peak max={np.max(levels.peak[noise_mask]):.6f} "
        f"RMS max={np.max(levels.rms[noise_mask]):.6f}"
    )
    for index, (mask, is_voiced) in enumerate(
        zip(recording.cue_masks, recording.voiced, strict=True), start=1
    ):
        if not mask.any():
            continue
        status = "" if is_voiced else "  (no speech found)"
        print(
            f"  Utterance {index}  peak max={np.max(levels.peak[mask]):.6f} "
            f"RMS max={np.max(levels.rms[mask]):.6f}{status}"
        )


def print_result(result: CalibrationResult, current: VadEvaluation) -> None:
    print("\nCurrent config.py values")
    print(f"  vad_silence_threshold: {DEFAULT_PREPROCESS_CONFIG.vad_silence_threshold}")
    print(f"  vad_rms_threshold: {DEFAULT_PREPROCESS_CONFIG.vad_rms_threshold}")
    print(f"  top_db: {DEFAULT_PREPROCESS_CONFIG.top_db}")
    print(f"  -> on this recording: {format_evaluation(current)}")

    print("\nRecommended config.py values")
    print(f"  vad_silence_threshold: {result.vad_silence_threshold:.6f}")
    print(f"  vad_rms_threshold: {result.vad_rms_threshold:.6f}")
    print(f"  top_db: {result.top_db:.1f}")
    print(f"  min_top_db: {result.min_top_db:.1f}")
    print(f"  max_top_db: {result.max_top_db:.1f}")
    print(f"  -> on this recording: {format_evaluation(result.nominal)}")
    print(f"  -> worst case of the adaptive VAD (0.8x/1.25x): {format_evaluation(result.robust)}")
    print(f"  SNR: {result.snr_db:.2f} dB")

    if result.warnings:
        print("\nWarnings")
        for warning in result.warnings:
            print(f"  - {warning}")


def save_chart(
    recording: Recording,
    session: Session,
    result: CalibrationResult,
    chart_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed. Skipping chart generation.")
        return

    chart_path.parent.mkdir(parents=True, exist_ok=True)
    levels = recording.levels
    time_axis = levels.end_times

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    series = (
        (axes[0], levels.peak, result.vad_silence_threshold, "Window peak", "VAD peak"),
        (axes[1], levels.rms, result.vad_rms_threshold, "Window RMS", "VAD RMS"),
    )
    for ax, values, threshold, label, threshold_label in series:
        ax.plot(time_axis, values, label=label)
        ax.axhline(threshold, color="r", linestyle="--", label=threshold_label)
        ax.axhspan(
            threshold * ADAPTIVE_LOW_SCALE,
            threshold * ADAPTIVE_HIGH_SCALE,
            color="r",
            alpha=0.1,
            label="Adaptive range",
        )
        ax.axvspan(session.noise_start, session.noise_end, color="gray", alpha=0.15, label="Noise")
        for cue in session.cue_times:
            ax.axvline(cue, color="k", linestyle=":", linewidth=0.8)
        ax.set_yscale("log")
        ax.set_ylabel(label)
        ax.legend(loc="upper left")
        ax.grid(True, which="both", alpha=0.3)
    axes[1].set_xlabel("Window end time (sec), dotted lines = cues")

    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close(fig)
    logger.info("Saved level chart: %s", chart_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = build_parser().parse_args()
    settings = VadSettings(
        sample_rate=DEFAULT_AUDIO_CONFIG.sample_rate,
        window_seconds=DEFAULT_AUDIO_CONFIG.window_seconds,
        hop_seconds=DEFAULT_AUDIO_CONFIG.chunk_seconds,
        min_speech_chunks=DEFAULT_PREPROCESS_CONFIG.vad_min_speech_chunks,
        min_active_ratio=DEFAULT_PREPROCESS_CONFIG.vad_min_active_ratio,
    )
    session = build_session(
        args.noise_seconds, args.utterances, args.cue_interval, settings.window_seconds
    )

    print("Microphone calibration")
    print(f"Total recording time: {session.total_seconds:.1f}s")
    waveform = record_session(
        session,
        sample_rate=settings.sample_rate,
        channels=DEFAULT_AUDIO_CONFIG.channels,
        device=args.device,
        countdown_seconds=args.countdown,
    )
    save_audio(args.audio_path, waveform, settings.sample_rate)

    recording = analyze_recording(waveform, session, settings)
    result = calculate_calibration(recording, settings)
    current = evaluate_thresholds(
        recording,
        settings,
        DEFAULT_PREPROCESS_CONFIG.vad_silence_threshold,
        DEFAULT_PREPROCESS_CONFIG.vad_rms_threshold,
    )

    print_levels(recording)
    print_result(result, current)

    if args.dry_run:
        print("\nDry run: config.py was not updated.")
    elif not result.reliable and not args.force:
        print(
            "\nconfig.py was NOT updated because the calibration looks unreliable. "
            "Fix the warnings above and run again, or pass --force to write anyway."
        )
    else:
        update_config_file(args.config_path, result)
        print(f"\nUpdated {args.config_path}")

    if not args.skip_chart:
        save_chart(recording, session, result, args.chart_path)


if __name__ == "__main__":
    main()
