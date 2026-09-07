import argparse
import logging
from pathlib import Path

from voicerecognizer.config import DEFAULT_RECOGNITION_CONFIG, PROJECT_ROOT
from voicerecognizer.evaluation.dataset_audit import (
    audit_dataset,
    format_audit_summary,
    write_audit_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit voice dataset health in one pass.")
    parser.add_argument(
        "--merged-dataset-dir",
        type=Path,
        default=DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir,
        help="Directory containing merged_dataset/index.csv",
    )
    parser.add_argument(
        "--processed-dataset-dir",
        type=Path,
        default=DEFAULT_RECOGNITION_CONFIG.processed_dataset_dir,
        help="Directory containing processed wav files",
    )
    parser.add_argument(
        "--raw-dataset-dir",
        type=Path,
        default=DEFAULT_RECOGNITION_CONFIG.raw_dataset_dir,
        help="Raw dataset root used for speaker inference",
    )
    parser.add_argument(
        "--review-candidates",
        type=Path,
        default=PROJECT_ROOT / "evaluation_results" / "review_candidates.json",
        help="Review candidates JSON path",
    )
    parser.add_argument(
        "--review-decisions",
        type=Path,
        default=PROJECT_ROOT / "evaluation_results" / "review_decisions.json",
        help="Review decisions JSON path",
    )
    parser.add_argument(
        "--evaluation-result",
        type=Path,
        default=PROJECT_ROOT / "evaluation_results" / "evaluation_result.json",
        help="Evaluation result JSON path used to summarize confusion pairs",
    )
    parser.add_argument(
        "--min-label-count",
        type=int,
        default=50,
        help="Minimum target sample count per label",
    )
    parser.add_argument(
        "--skip-hash-duplicates",
        action="store_true",
        help="Skip exact audio duplicate detection by file hash",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "evaluation_results" / "dataset_audit.json",
        help="Output JSON report path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit_dataset(
        merged_dataset_dir=args.merged_dataset_dir,
        processed_dataset_dir=args.processed_dataset_dir,
        raw_dataset_dir=args.raw_dataset_dir,
        review_candidates_path=args.review_candidates,
        review_decisions_path=args.review_decisions,
        evaluation_result_path=args.evaluation_result,
        min_label_count=args.min_label_count,
        include_hash_duplicates=not args.skip_hash_duplicates,
    )
    print(format_audit_summary(report))
    if args.output_json:
        write_audit_json(report, args.output_json)
        print(f"Saved JSON: {args.output_json}")


if __name__ == "__main__":
    main()
