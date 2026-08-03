import argparse
from pathlib import Path
import time

import numpy as np
import sounddevice as sd
import soundfile as sf


def collect_voice_samples(
    speaker_id: str,
    root: str | Path = "dataset",
    labels: tuple[str, ...] = ("a", "i", "u", "e", "o"),
    sample_rate: int = 16000,
    record_seconds: float = 1.0,
    repeat: int = 10,
    silence_threshold: float = 0.02,
) -> None:
    speaker_dir = Path(root) / speaker_id
    speaker_dir.mkdir(parents=True, exist_ok=True)

    for label in labels:
        folder = speaker_dir / label
        folder.mkdir(exist_ok=True)
        next_number = _next_file_number(folder)
        saved = 0

        while saved < repeat:
            print(f"{label}: {saved + 1}/{repeat}")
            _countdown(3)
            audio = sd.rec(
                int(record_seconds * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32,
            )
            sd.wait()

            if np.max(np.abs(audio)) < silence_threshold:
                print("Silence detected. Please try again.")
                continue

            sf.write(folder / f"{next_number:03d}.wav", audio, sample_rate)
            saved += 1
            next_number += 1


def _next_file_number(folder: Path) -> int:
    numbers = []
    for wav_path in folder.glob("*.wav"):
        try:
            numbers.append(int(wav_path.stem))
        except ValueError:
            continue
    return max(numbers, default=0) + 1


def _countdown(seconds: int) -> None:
    for count in range(seconds, 0, -1):
        print(count)
        time.sleep(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect short voice samples.")
    parser.add_argument("speaker_id")
    parser.add_argument("--root", type=Path, default=Path("dataset"))
    parser.add_argument("--repeat", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    collect_voice_samples(args.speaker_id, root=args.root, repeat=args.repeat)


if __name__ == "__main__":
    main()
