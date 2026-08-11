"""
Audio input level calibration utility.

The calibration records two phases:
1. Room noise while you stay silent.
2. Normal speech.

It then writes the measured thresholds back to config.py by default.
"""

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import re
import sys
from time import sleep

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    DEFAULT_AUDIO_CONFIG,
    DEFAULT_PREPROCESS_CONFIG,
    DEFAULT_RECOGNITION_CONFIG,
)

logger = logging.getLogger(__name__)

EPSILON = 1e-10
DEFAULT_NOISE_SECONDS = 3.0
DEFAULT_SPEECH_SECONDS = 3.0
DEFAULT_AUDIO_FILE = PROJECT_ROOT / "measured_audio.wav"
DEFAULT_CHART_FILE = PROJECT_ROOT / "Docs" / "charts" / "audio_level_measurement.png"


@dataclass(frozen=True)
class FrameLevels:
    rms: np.ndarray
    peak: np.ndarray
    db: np.ndarray


@dataclass(frozen=True)
class CalibrationResult:
    top_db: float
    min_top_db: float
    max_top_db: float
    vad_silence_threshold: float
    vad_rms_threshold: float
    snr_db: float
    active_speech_ratio: float
    noise_rms_p95: float
    noise_peak_p99: float
    speech_rms_p20: float
    speech_peak_p20: float
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
        "--speech-seconds",
        type=float,
        default=DEFAULT_SPEECH_SECONDS,
        help="Seconds to record normal speech.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=DEFAULT_AUDIO_CONFIG.chunk_seconds,
        help="Frame size used for level analysis.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional sounddevice input device id or name.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=PROJECT_ROOT / "config.py",
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
        "--skip-chart",
        action="store_true",
        help="Skip chart generation.",
    )
    parser.add_argument(
        "--countdown",
        type=int,
        default=3,
        help="Countdown seconds before each recording phase.",
    )
    return parser


def countdown(seconds: int) -> None:
    for remaining in range(max(0, seconds), 0, -1):
        print(remaining)
        sleep(1)


def record_audio(
    label: str,
    seconds: float,
    sample_rate: int,
    channels: int,
    device: str | None,
    countdown_seconds: int,
) -> np.ndarray:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise ImportError(
            "sounddevice is required to record calibration audio. "
            "Install dependencies with: uv sync"
        ) from exc

    print(f"\n[{label}] starts after countdown. Duration: {seconds:.1f}s")
    countdown(countdown_seconds)
    print("Recording...")
    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        device=device,
    )
    sd.wait()
    print("Done.")
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def save_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError(
            "soundfile is required to save measured_audio.wav. "
            "Install dependencies with: uv sync"
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


def calculate_calibration(
    noise_levels: FrameLevels,
    speech_levels: FrameLevels,
) -> CalibrationResult:
    noise_rms_p95 = percentile(noise_levels.rms, 95)
    noise_peak_p99 = percentile(noise_levels.peak, 99)
    noise_peak_p95 = percentile(noise_levels.peak, 95)

    speech_gate_rms = max(noise_rms_p95 * 3.0, 1e-6)
    speech_gate_peak = max(noise_peak_p95 * 3.0, 1e-5)
    active_mask = (speech_levels.rms >= speech_gate_rms) | (
        speech_levels.peak >= speech_gate_peak
    )

    min_active_frames = max(3, int(np.ceil(speech_levels.rms.size * 0.2)))
    if int(np.sum(active_mask)) < min_active_frames:
        fallback_gate = percentile(speech_levels.rms, 60)
        active_mask = speech_levels.rms >= fallback_gate

    active_rms = speech_levels.rms[active_mask]
    active_peak = speech_levels.peak[active_mask]
    if active_rms.size == 0:
        active_rms = speech_levels.rms
        active_peak = speech_levels.peak

    speech_rms_p20 = percentile(active_rms, 20)
    speech_rms_p50 = percentile(active_rms, 50)
    speech_rms_p95 = percentile(active_rms, 95)
    speech_peak_p20 = percentile(active_peak, 20)

    vad_rms_threshold = choose_threshold(
        noise_high=noise_rms_p95,
        speech_low=speech_rms_p20,
        minimum=1e-6,
        noise_multiplier=1.5,
        speech_fraction=0.70,
    )
    vad_silence_threshold = choose_threshold(
        noise_high=noise_peak_p99,
        speech_low=speech_peak_p20,
        minimum=1e-4,
        noise_multiplier=1.2,
        speech_fraction=0.70,
    )

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
    active_speech_ratio = float(np.mean(active_mask)) if active_mask.size else 0.0
    warnings: list[str] = []

    if snr_db < 10.0:
        warnings.append(
            "Speech is less than 10 dB above the noise floor. "
            "Calibrate again in a quieter room or speak closer to the mic."
        )
    if active_speech_ratio < 0.2:
        warnings.append(
            "Very little speech was detected during the speech phase. "
            "Calibrate again and keep speaking through the whole phase."
        )
    max_peak = max(
        percentile(noise_levels.peak, 100),
        percentile(speech_levels.peak, 100),
    )
    if max_peak > 0.98:
        warnings.append(
            "The recording is close to clipping. Lower the microphone gain."
        )

    return CalibrationResult(
        top_db=round(top_db, 1),
        min_top_db=round(min_top_db, 1),
        max_top_db=round(max_top_db, 1),
        vad_silence_threshold=round(vad_silence_threshold, 6),
        vad_rms_threshold=round(vad_rms_threshold, 6),
        snr_db=round(snr_db, 2),
        active_speech_ratio=round(active_speech_ratio, 3),
        noise_rms_p95=round(noise_rms_p95, 6),
        noise_peak_p99=round(noise_peak_p99, 6),
        speech_rms_p20=round(speech_rms_p20, 6),
        speech_peak_p20=round(speech_peak_p20, 6),
        warnings=tuple(warnings),
    )


def format_config_float(value: float) -> str:
    if abs(value) < 1.0:
        text = f"{value:.6f}"
    else:
        text = f"{value:.1f}"
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
    recognition_values = {
        "top_db": format_config_float(result.top_db),
    }

    text = replace_class_defaults(text, "PreprocessConfig", preprocess_values)
    text = replace_class_defaults(text, "RecognitionConfig", recognition_values)
    config_path.write_text(text, encoding="utf-8")


def print_phase_summary(label: str, levels: FrameLevels) -> None:
    print(f"\n{label}")
    print(f"  RMS  mean={np.mean(levels.rms):.6f} p95={percentile(levels.rms, 95):.6f}")
    print(
        f"  Peak mean={np.mean(levels.peak):.6f} "
        f"p99={percentile(levels.peak, 99):.6f}"
    )
    print(f"  dB   mean={np.mean(levels.db):.2f} max={np.max(levels.db):.2f}")


def print_result(result: CalibrationResult) -> None:
    print("\nRecommended config.py values")
    print(f"  top_db: {result.top_db:.1f}")
    print(f"  min_top_db: {result.min_top_db:.1f}")
    print(f"  max_top_db: {result.max_top_db:.1f}")
    print(f"  vad_silence_threshold: {result.vad_silence_threshold:.6f}")
    print(f"  vad_rms_threshold: {result.vad_rms_threshold:.6f}")
    print(f"  SNR: {result.snr_db:.2f} dB")
    print(f"  Active speech frames: {result.active_speech_ratio * 100:.1f}%")

    if result.warnings:
        print("\nWarnings")
        for warning in result.warnings:
            print(f"  - {warning}")


def save_chart(
    noise_levels: FrameLevels,
    speech_levels: FrameLevels,
    result: CalibrationResult,
    chunk_seconds: float,
    chart_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed. Skipping chart generation.")
        return

    chart_path.parent.mkdir(parents=True, exist_ok=True)
    rms = np.concatenate([noise_levels.rms, speech_levels.rms])
    peak = np.concatenate([noise_levels.peak, speech_levels.peak])
    db = np.concatenate([noise_levels.db, speech_levels.db])
    time_axis = np.arange(len(rms)) * chunk_seconds
    phase_boundary = len(noise_levels.rms) * chunk_seconds

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(time_axis, rms, marker="o", label="RMS")
    axes[0].axhline(
        result.vad_rms_threshold,
        color="r",
        linestyle="--",
        label="VAD RMS",
    )
    axes[0].axvline(phase_boundary, color="k", linestyle=":", label="Speech start")
    axes[0].set_ylabel("RMS")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(time_axis, peak, marker="s", color="orange", label="Peak")
    axes[1].axhline(
        result.vad_silence_threshold,
        color="r",
        linestyle="--",
        label="VAD Peak",
    )
    axes[1].axvline(phase_boundary, color="k", linestyle=":")
    axes[1].set_ylabel("Peak")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(time_axis, db, marker="^", color="green", label="dB")
    axes[2].axvline(phase_boundary, color="k", linestyle=":")
    axes[2].set_xlabel("Time (sec)")
    axes[2].set_ylabel("dB")
    axes[2].legend()
    axes[2].grid(True)

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
    sample_rate = DEFAULT_AUDIO_CONFIG.sample_rate

    print("Microphone calibration")
    print("Phase 1: stay silent and let the script measure room noise.")
    noise = record_audio(
        label="noise",
        seconds=args.noise_seconds,
        sample_rate=sample_rate,
        channels=DEFAULT_AUDIO_CONFIG.channels,
        device=args.device,
        countdown_seconds=args.countdown,
    )

    print("\nPhase 2: speak normally through the whole recording.")
    speech = record_audio(
        label="speech",
        seconds=args.speech_seconds,
        sample_rate=sample_rate,
        channels=DEFAULT_AUDIO_CONFIG.channels,
        device=args.device,
        countdown_seconds=args.countdown,
    )

    gap = np.zeros(int(sample_rate * 0.25), dtype=np.float32)
    save_audio(args.audio_path, np.concatenate([noise, gap, speech]), sample_rate)

    noise_levels = calculate_frame_levels(noise, sample_rate, args.chunk_seconds)
    speech_levels = calculate_frame_levels(speech, sample_rate, args.chunk_seconds)
    result = calculate_calibration(noise_levels, speech_levels)

    print_phase_summary("Noise phase", noise_levels)
    print_phase_summary("Speech phase", speech_levels)
    print_result(result)

    print("\nCurrent config.py values")
    print(f"  PreprocessConfig.top_db: {DEFAULT_PREPROCESS_CONFIG.top_db}")
    print(
        "  PreprocessConfig.vad_silence_threshold: "
        f"{DEFAULT_PREPROCESS_CONFIG.vad_silence_threshold}"
    )
    print(
        "  PreprocessConfig.vad_rms_threshold: "
        f"{DEFAULT_PREPROCESS_CONFIG.vad_rms_threshold}"
    )
    print(f"  RecognitionConfig.top_db: {DEFAULT_RECOGNITION_CONFIG.top_db}")

    if args.dry_run:
        print("\nDry run: config.py was not updated.")
    else:
        update_config_file(args.config_path, result)
        print(f"\nUpdated {args.config_path}")

    if not args.skip_chart:
        save_chart(
            noise_levels=noise_levels,
            speech_levels=speech_levels,
            result=result,
            chunk_seconds=args.chunk_seconds,
            chart_path=args.chart_path,
        )


if __name__ == "__main__":
    main()
