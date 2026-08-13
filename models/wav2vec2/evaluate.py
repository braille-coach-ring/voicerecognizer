import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_RECOGNITION_CONFIG  # noqa: E402
from evaluation.evaluator import Evaluator  # noqa: E402
from recognizers.wav2vec2_recognizer import Wav2Vec2Recognizer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def evaluate(args: argparse.Namespace) -> None:
    recognizer = Wav2Vec2Recognizer(
        model_path=args.model_path,
        sample_rate=args.sample_rate,
    )
    evaluator = Evaluator(model=recognizer, dataset_path=args.dataset_dir)
    result = evaluator.evaluate()

    logger.info("--- Wav2Vec2 evaluation results ---")
    logger.info("Accuracy    : %.4f", result.overall.accuracy)
    logger.info("Macro F1    : %.4f", result.overall.macro_f1)
    logger.info("Weighted F1 : %.4f", result.overall.weighted_f1)
    logger.info("Total       : %d samples", result.overall.total_samples)

    if args.output_json:
        evaluator.export_json(args.output_json)


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
        default=DEFAULT_RECOGNITION_CONFIG.sample_rate,
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "evaluation_results" / "wav2vec2_result.json",
    )
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
