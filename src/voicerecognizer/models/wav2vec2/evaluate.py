import argparse
import logging
from pathlib import Path

from voicerecognizer.config import DEFAULT_AUDIO_CONFIG, DEFAULT_RECOGNITION_CONFIG, PROJECT_ROOT
from voicerecognizer.evaluation.evaluator import Evaluator
from voicerecognizer.recognizers.wav2vec2_recognizer import Wav2Vec2Recognizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def evaluate(args: argparse.Namespace) -> None:
    recognizer = Wav2Vec2Recognizer(model_path=args.model_path)
    evaluator = Evaluator(
        model=recognizer,
        dataset_path=args.dataset_dir,
        review_decisions_path=args.review_decisions,
    )
    result = evaluator.evaluate()

    logger.info("--- Wav2Vec2 evaluation results ---")
    logger.info("Accuracy    : %.4f", result.overall.accuracy)
    logger.info("Macro F1    : %.4f", result.overall.macro_f1)
    logger.info("Weighted F1 : %.4f", result.overall.weighted_f1)
    logger.info("Total       : %d samples", result.overall.total_samples)

    if args.output_json:
        evaluator.export_json(args.output_json)

    if args.output_review_json:
        evaluator.export_review_json(args.output_review_json)

    if args.output_review_html:
        evaluator.export_review_html(
            args.output_review_html,
            title="Voice Data Quality Review (WAV2VEC2)",
            review_results_path=args.review_decisions,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned Wav2Vec2 recognizer.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_RECOGNITION_CONFIG.wav2vec2_best_model_dir,
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_AUDIO_CONFIG.sample_rate,
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "evaluation_results" / "wav2vec2_result.json",
    )
    parser.add_argument(
        "--output-review-json",
        type=Path,
        default=PROJECT_ROOT / "evaluation_results" / "wav2vec2_review_candidates.json",
    )
    parser.add_argument(
        "--output-review-html",
        type=Path,
        default=PROJECT_ROOT / "evaluation_results" / "wav2vec2_review_report.html",
    )
    parser.add_argument(
        "--review-decisions",
        type=Path,
        default=PROJECT_ROOT / "evaluation_results" / "review_decisions.json",
    )
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
