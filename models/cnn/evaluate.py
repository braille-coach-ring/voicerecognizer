import argparse
from pathlib import Path

from config import (
    DEFAULT_AUDIO_CONFIG,
    DEFAULT_PREPROCESS_CONFIG,
    DEFAULT_RECOGNITION_CONFIG,
)
from dataset.hiragana_dataset import HiraganaDataset
from recognizers.cnn_recognizer import CNNRecognizer


def evaluate(args: argparse.Namespace) -> None:
    dataset = HiraganaDataset(
        root_dir=args.root_dir,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
    )
    recognizer = CNNRecognizer(
        model_path=args.model_path,
        labels=dataset.labels,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
    )

    total = 0
    correct = 0
    class_total = {label: 0 for label in dataset.labels}
    class_correct = {label: 0 for label in dataset.labels}

    for label in dataset.labels:
        folder = Path(args.root_dir) / label
        for wav_path in sorted(folder.glob("*.wav")):
            predicted_label = recognizer.recognize(wav_path)
            total += 1
            class_total[label] += 1
            if predicted_label == label:
                correct += 1
                class_correct[label] += 1
            print(f"{label}/{wav_path.name}: predicted={predicted_label}")

    print()
    for label in dataset.labels:
        acc = class_correct[label] / class_total[label] * 100
        print(f"{label}: {class_correct[label]}/{class_total[label]} ({acc:.2f}%)")
    print(f"Total: {correct}/{total} ({correct / total * 100:.2f}%)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the Hiragana CNN recognizer."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=DEFAULT_RECOGNITION_CONFIG.processed_dataset_dir,
    )
    parser.add_argument(
        "--model-path", type=Path, default=DEFAULT_RECOGNITION_CONFIG.cnn_weight_path
    )
    parser.add_argument(
        "--sample-rate", type=int, default=DEFAULT_AUDIO_CONFIG.sample_rate
    )
    parser.add_argument("--n-mels", type=int, default=DEFAULT_PREPROCESS_CONFIG.n_mels)
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
