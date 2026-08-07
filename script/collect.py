"""
Interactive voice dataset collector.

Usage:
  uv run python script/collect.py
  uv run python script/collect.py rinry --repeat 10
  uv run python script/collect.py rinry --mode fixed --start-delay 0.3
"""

from __future__ import annotations

import argparse
import logging
import math
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    DEFAULT_AUDIO_CONFIG,
    DEFAULT_PREPROCESS_CONFIG,
    DEFAULT_RECOGNITION_CONFIG,
)
from config_labels import (  # noqa: E402
    DAKUON_LABELS,
    HANDAKUON_LABELS,
    ROMAJI_TO_HIRAGANA,
    SEION_LABELS,
    YOON_LABELS,
)

logger = logging.getLogger(__name__)

SAMPLE_RATE = DEFAULT_AUDIO_CONFIG.sample_rate
CHANNELS = DEFAULT_AUDIO_CONFIG.channels
RECORD_SECONDS = DEFAULT_AUDIO_CONFIG.window_seconds
DEFAULT_REPEAT = 10
ROOT = DEFAULT_RECOGNITION_CONFIG.raw_dataset_dir

LABEL_GROUPS: dict[str, tuple[str, ...]] = {
    "seion": SEION_LABELS,
    "dakuon": DAKUON_LABELS,
    "handakuon": HANDAKUON_LABELS,
    "yoon": YOON_LABELS,
}
DEFAULT_GROUP_NAMES = ("seion", "dakuon", "handakuon", "yoon")
DEFAULT_COLLECT_LABELS: tuple[str, ...] = tuple(
    label for group_name in DEFAULT_GROUP_NAMES for label in LABEL_GROUPS[group_name]
)
KNOWN_LABELS = set(DEFAULT_COLLECT_LABELS) | {"other"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect short hiragana voice samples into "
            "dataset/<speaker_id>/<label>/."
        )
    )
    parser.add_argument(
        "speaker_id",
        nargs="?",
        help="Speaker ID. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=DEFAULT_REPEAT,
        help="Number of new samples to collect per label.",
    )
    parser.add_argument(
        "--target-per-label",
        type=int,
        default=None,
        help=(
            "Collect only the missing amount until each label has this many wav "
            "files. Useful when resuming a large collection."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("vad", "fixed"),
        default="vad",
        help=(
            "vad saves each spoken utterance automatically. fixed records fixed "
            "length takes with a short interval."
        ),
    )
    parser.add_argument(
        "--groups",
        default="all",
        help=(
            "Comma-separated groups to collect: all, seion, dakuon, handakuon, "
            "yoon. Ignored when --labels is set."
        ),
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Comma-separated romaji labels to collect, e.g. a,i,u,kya,gyu.",
    )
    parser.add_argument(
        "--include-other",
        action="store_true",
        help='Also collect the "other" label.',
    )
    parser.add_argument(
        "--start-at",
        default=None,
        help="Start from this label in the resolved label order.",
    )
    parser.add_argument(
        "--confirm-each-label",
        action="store_true",
        help="Ask before starting every label. You can skip or quit at prompts.",
    )
    parser.add_argument(
        "--label-delay",
        type=float,
        default=1.0,
        help="Seconds to wait before each label when not confirming manually.",
    )
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=RECORD_SECONDS,
        help="Fixed-mode recording length for one take.",
    )
    parser.add_argument(
        "--start-delay",
        type=float,
        default=0.25,
        help="Fixed-mode delay before each take starts.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.25,
        help="Fixed-mode pause after a saved take.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=0.05,
        help="VAD-mode audio chunk size.",
    )
    parser.add_argument(
        "--pre-roll-seconds",
        type=float,
        default=0.15,
        help="VAD-mode audio kept before speech is detected.",
    )
    parser.add_argument(
        "--speech-end-seconds",
        type=float,
        default=0.25,
        help="VAD-mode silence duration that ends one utterance.",
    )
    parser.add_argument(
        "--min-utterance-seconds",
        type=float,
        default=0.12,
        help="VAD-mode minimum utterance length to save.",
    )
    parser.add_argument(
        "--max-utterance-seconds",
        type=float,
        default=1.2,
        help="VAD-mode maximum utterance length before forced save.",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=DEFAULT_PREPROCESS_CONFIG.vad_silence_threshold,
        help="Peak threshold for detecting speech.",
    )
    parser.add_argument(
        "--rms-threshold",
        type=float,
        default=DEFAULT_PREPROCESS_CONFIG.vad_rms_threshold,
        help="RMS threshold for detecting speech.",
    )
    upload_group = parser.add_mutually_exclusive_group()
    upload_group.add_argument(
        "--upload",
        action="store_true",
        help="Run git add/commit/pull/push after collection without asking.",
    )
    upload_group.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip the GitHub upload prompt after collection.",
    )
    return parser


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def resolve_labels(args: argparse.Namespace) -> tuple[str, ...]:
    if args.labels:
        labels = parse_csv(args.labels)
    else:
        requested_groups = [group.lower() for group in parse_csv(args.groups)]
        if not requested_groups or "all" in requested_groups:
            requested_groups = list(DEFAULT_GROUP_NAMES)

        invalid_groups = [
            group for group in requested_groups if group not in LABEL_GROUPS
        ]
        if invalid_groups:
            raise ValueError(f"Unknown label group(s): {', '.join(invalid_groups)}")

        labels = [
            label
            for group_name in requested_groups
            for label in LABEL_GROUPS[group_name]
        ]

    if args.include_other and "other" not in labels:
        labels.append("other")

    deduped_labels = list(dict.fromkeys(labels))
    invalid_labels = [label for label in deduped_labels if label not in KNOWN_LABELS]
    if invalid_labels:
        raise ValueError(f"Unknown label(s): {', '.join(invalid_labels)}")

    if args.start_at:
        if args.start_at not in deduped_labels:
            raise ValueError(f"--start-at label is not in target labels: {args.start_at}")
        start_index = deduped_labels.index(args.start_at)
        deduped_labels = deduped_labels[start_index:]

    return tuple(deduped_labels)


def validate_args(args: argparse.Namespace) -> None:
    positive_values = {
        "--repeat": args.repeat,
        "--label-delay": args.label_delay,
        "--record-seconds": args.record_seconds,
        "--start-delay": args.start_delay,
        "--interval": args.interval,
        "--chunk-seconds": args.chunk_seconds,
        "--pre-roll-seconds": args.pre_roll_seconds,
        "--speech-end-seconds": args.speech_end_seconds,
        "--min-utterance-seconds": args.min_utterance_seconds,
        "--max-utterance-seconds": args.max_utterance_seconds,
    }
    for name, value in positive_values.items():
        if value < 0:
            raise ValueError(f"{name} must be 0 or greater")

    if args.repeat <= 0:
        raise ValueError("--repeat must be greater than 0")
    if args.target_per_label is not None and args.target_per_label < 0:
        raise ValueError("--target-per-label must be 0 or greater")
    if args.record_seconds <= 0:
        raise ValueError("--record-seconds must be greater than 0")
    if args.chunk_seconds <= 0:
        raise ValueError("--chunk-seconds must be greater than 0")
    if args.max_utterance_seconds <= 0:
        raise ValueError("--max-utterance-seconds must be greater than 0")
    if args.max_utterance_seconds < args.min_utterance_seconds:
        raise ValueError(
            "--max-utterance-seconds must be greater than or equal to "
            "--min-utterance-seconds"
        )


def format_label(label: str) -> str:
    hiragana = ROMAJI_TO_HIRAGANA.get(label, label)
    if hiragana == label:
        return label
    return f"{label} ({hiragana})"


def next_file_number(folder: Path) -> int:
    numbers: list[int] = []
    for wav_path in folder.glob("*.wav"):
        try:
            numbers.append(int(wav_path.stem))
        except ValueError:
            continue
    return max(numbers, default=0) + 1


def wav_count(folder: Path) -> int:
    return sum(1 for _ in folder.glob("*.wav"))


def samples_needed(folder: Path, args: argparse.Namespace) -> int:
    if args.target_per_label is None:
        return args.repeat
    return max(0, args.target_per_label - wav_count(folder))


def to_mono(audio: np.ndarray) -> np.ndarray:
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    return waveform.reshape(-1)


def audio_stats(audio: np.ndarray) -> tuple[float, float]:
    if audio.size == 0:
        return 0.0, 0.0
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float32))))
    return peak, rms


def is_speech(audio: np.ndarray, args: argparse.Namespace) -> bool:
    peak, rms = audio_stats(audio)
    return peak >= args.silence_threshold and rms >= args.rms_threshold


def save_audio(folder: Path, number: int, audio: np.ndarray) -> Path:
    filename = folder / f"{number:03d}.wav"
    sf.write(filename, audio.astype(np.float32), SAMPLE_RATE)
    return filename


def print_saved(
    filename: Path,
    saved: int,
    needed: int,
    audio: np.ndarray,
) -> None:
    peak, rms = audio_stats(audio)
    duration = audio.size / SAMPLE_RATE
    print(
        f"  saved {saved}/{needed}: {filename.name} "
        f"({duration:.2f}s, peak={peak:.4f}, rms={rms:.4f})"
    )


def collect_fixed(
    label: str,
    folder: Path,
    needed: int,
    start_number: int,
    args: argparse.Namespace,
) -> int:
    number = start_number
    saved = 0
    samples = int(args.record_seconds * SAMPLE_RATE)

    while saved < needed:
        print(f"  {format_label(label)} {saved + 1}/{needed}: speak now")
        time.sleep(args.start_delay)

        audio = sd.rec(
            samples,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.float32,
        )
        sd.wait()
        waveform = to_mono(audio)

        if not is_speech(waveform, args):
            peak, rms = audio_stats(waveform)
            print(
                "  ignored silence/noise "
                f"(peak={peak:.4f}, rms={rms:.4f}); retrying"
            )
            continue

        filename = save_audio(folder, number, waveform)
        saved += 1
        print_saved(filename, saved, needed, waveform)
        number += 1
        time.sleep(args.interval)

    return number


def collect_vad(
    label: str,
    folder: Path,
    needed: int,
    start_number: int,
    args: argparse.Namespace,
) -> int:
    number = start_number
    saved = 0
    chunk_samples = max(1, int(args.chunk_seconds * SAMPLE_RATE))
    pre_roll_chunks = max(1, math.ceil(args.pre_roll_seconds / args.chunk_seconds))
    speech_end_chunks = max(1, math.ceil(args.speech_end_seconds / args.chunk_seconds))
    max_chunks = max(1, math.ceil(args.max_utterance_seconds / args.chunk_seconds))
    min_samples = int(args.min_utterance_seconds * SAMPLE_RATE)
    min_speech_chunks = max(1, DEFAULT_PREPROCESS_CONFIG.vad_min_speech_chunks)

    pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_chunks)
    utterance_chunks: list[np.ndarray] = []
    speech_run = 0
    silence_run = 0

    print(
        f"  Listening for {needed} utterance(s). "
        "Say the label once each time; short pauses split samples."
    )

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=np.float32,
        blocksize=chunk_samples,
    ) as stream:
        while saved < needed:
            chunk, overflowed = stream.read(chunk_samples)
            if overflowed:
                logger.warning("Input overflow was detected while recording.")

            waveform = to_mono(chunk.copy())
            speech = is_speech(waveform, args)

            if not utterance_chunks:
                pre_roll.append(waveform)
                if speech:
                    speech_run += 1
                    if speech_run >= min_speech_chunks:
                        utterance_chunks = list(pre_roll)
                        pre_roll.clear()
                        silence_run = 0
                else:
                    speech_run = 0
                continue

            utterance_chunks.append(waveform)
            if speech:
                silence_run = 0
            else:
                silence_run += 1

            reached_end = silence_run >= speech_end_chunks
            reached_max = len(utterance_chunks) >= max_chunks
            if not (reached_end or reached_max):
                continue

            audio = np.concatenate(utterance_chunks)
            peak, rms = audio_stats(audio)
            valid_length = audio.size >= min_samples
            valid_volume = peak >= args.silence_threshold and rms >= args.rms_threshold

            if valid_length and valid_volume:
                filename = save_audio(folder, number, audio)
                saved += 1
                print_saved(filename, saved, needed, audio)
                number += 1
            else:
                print(
                    "  ignored short/noisy input "
                    f"({audio.size / SAMPLE_RATE:.2f}s, "
                    f"peak={peak:.4f}, rms={rms:.4f})"
                )

            pre_roll.clear()
            utterance_chunks = []
            speech_run = 0
            silence_run = 0

    return number


def maybe_wait_for_label(args: argparse.Namespace) -> str:
    if args.confirm_each_label:
        answer = input("[Enter]=start, s=skip, q=quit > ").strip().lower()
        if answer in {"s", "skip"}:
            return "skip"
        if answer in {"q", "quit"}:
            return "quit"
        return "collect"

    time.sleep(args.label_delay)
    return "collect"


def upload_to_github(speaker_id: str) -> None:
    logger.info("Git add...")
    subprocess.run(["git", "add", "."], check=True)

    logger.info("Git commit...")
    subprocess.run(["git", "commit", "-m", f"Add voice data ({speaker_id})"], check=True)

    logger.info("Git pull --rebase...")
    subprocess.run(["git", "pull", "--rebase"], check=True)

    logger.info("Git push...")
    subprocess.run(["git", "push"], check=True)


def maybe_upload(speaker_id: str, args: argparse.Namespace) -> None:
    if args.no_upload:
        logger.info("Skipped GitHub upload.")
        return

    should_upload = args.upload
    if not should_upload:
        answer = input("\nGitHubへアップロードしますか？ [Y/n] ").strip().lower()
        should_upload = answer in {"", "y", "yes"}

    if not should_upload:
        logger.info("Skipped GitHub upload.")
        return

    try:
        upload_to_github(speaker_id)
        logger.info("GitHub upload completed.")
    except subprocess.CalledProcessError as exc:
        logger.error("GitHub upload failed: %s", exc, exc_info=True)


def run_collection(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        validate_args(args)
        labels = resolve_labels(args)
    except ValueError as exc:
        parser.error(str(exc))

    ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Voice dataset collector")
    print("=" * 60)

    speaker_id = args.speaker_id
    while not speaker_id:
        speaker_id = input("Speaker ID: ").strip()

    person_dir = ROOT / speaker_id
    person_dir.mkdir(parents=True, exist_ok=True)
    for label in labels:
        (person_dir / label).mkdir(exist_ok=True)

    print()
    print(f"Save directory: {person_dir}")
    print(f"Labels: {len(labels)}")
    print(f"Mode: {args.mode}")
    if args.target_per_label is None:
        print(f"New samples per label: {args.repeat}")
    else:
        print(f"Target wav files per label: {args.target_per_label}")
    print(
        "Thresholds: "
        f"peak={args.silence_threshold:.4f}, rms={args.rms_threshold:.4f}"
    )
    input("\nPress Enter to start.")

    try:
        for index, label in enumerate(labels, start=1):
            folder = person_dir / label
            needed = samples_needed(folder, args)
            display = format_label(label)

            print("\n" + "=" * 60)
            print(f"Label {index}/{len(labels)}: {display}")
            print(f"Existing wav files: {wav_count(folder)}")

            if needed <= 0:
                print("Already complete; skipping.")
                continue

            print(f"Collecting: {needed}")
            action = maybe_wait_for_label(args)
            if action == "skip":
                print("Skipped.")
                continue
            if action == "quit":
                print("Stopped before this label.")
                break

            start_number = next_file_number(folder)
            if args.mode == "vad":
                collect_vad(label, folder, needed, start_number, args)
            else:
                collect_fixed(label, folder, needed, start_number, args)

    except KeyboardInterrupt:
        print("\nStopped by user. Saved files remain in the dataset directory.")

    print("\n" + "=" * 60)
    print("Collection finished")
    print("=" * 60)

    maybe_upload(speaker_id, args)


if __name__ == "__main__":
    run_collection()
