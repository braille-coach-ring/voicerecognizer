import argparse
import logging
from pathlib import Path

from voicerecognizer.config import DEFAULT_RECOGNITION_CONFIG, PROJECT_ROOT
from voicerecognizer.evaluation.dynamic_trimming_compare import (
    compare_dynamic_trimming_accuracy,
    format_dynamic_trimming_summary,
    write_dynamic_trimming_comparison_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Wav2Vec2 accuracy with dynamic_trimming True and False."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir,
        help="Dataset directory containing index.csv",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_RECOGNITION_CONFIG.wav2vec2_best_model_dir,
        help="Wav2Vec2 model directory to evaluate",
    )
    parser.add_argument(
        "--review-decisions",
        type=Path,
        default=PROJECT_ROOT / "evaluation_results" / "review_decisions.json",
        help="Review decisions JSON path",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "evaluation_results" / "dynamic_trimming_accuracy.json",
        help="Output comparison JSON path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    comparison = compare_dynamic_trimming_accuracy(
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        review_decisions_path=args.review_decisions,
    )
    print(format_dynamic_trimming_summary(comparison))
    if args.output_json:
        write_dynamic_trimming_comparison_json(comparison, args.output_json)
        print(f"Saved JSON: {args.output_json}")


if __name__ == "__main__":
    main()
