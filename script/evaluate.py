"""
Model Evaluation CLI & HTML Report Generator Script

役割:
  指定モデル (cnn / wav2vec2 / whisper) の精度 (Accuracy, Macro-F1) の定量評価を行い、
  JSON 評価結果および対話型 HTML ダッシュボード (誤判定音声のインライン再生プレーヤー付き) を生成します。

使い方:
  uv run python script/evaluate.py                 # デフォルト CNN モデルの評価 ＆ HTML出力
  uv run python script/evaluate.py --model-type wav2vec2
  uv run python script/evaluate.py --from-dataset-only # 過去録音の predicted_text のみで即時評価
"""

import argparse
import logging
from pathlib import Path

from voicerecognizer.config import DEFAULT_RECOGNITION_CONFIG, PROJECT_ROOT
from voicerecognizer.core.factory.recognizer_factory import RecognizerFactory
from voicerecognizer.evaluation.evaluator import Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Voice Recognizer Model")
    parser.add_argument(
        "--model-type",
        type=str,
        default="cnn",
        choices=RecognizerFactory.available_strategies(),
        help="Model strategy to evaluate",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir,
        help="Path to dataset directory containing index.csv",
    )
    parser.add_argument(
        "--from-dataset-only",
        action="store_true",
        help="Evaluate using recorded predicted_text in index.csv without model inference",
    )
    parser.add_argument(
        "--update-index",
        action="store_true",
        help="Update predicted_text column in index.csv with latest model predictions",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "evaluation_results" / "evaluation_result.json",
        help="Output path for evaluation JSON report",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=PROJECT_ROOT / "evaluation_results" / "evaluation_report.html",
        help="Output path for human-friendly HTML dashboard report",
    )
    parser.add_argument(
        "--output-review-json",
        type=Path,
        default=PROJECT_ROOT / "evaluation_results" / "review_candidates.json",
        help="Output path for review candidate JSON",
    )
    parser.add_argument(
        "--output-review-html",
        type=Path,
        default=PROJECT_ROOT / "evaluation_results" / "review_report.html",
        help="Output path for browser-based audio quality review report",
    )
    parser.add_argument(
        "--review-decisions",
        type=Path,
        default=PROJECT_ROOT / "evaluation_results" / "review_decisions.json",
        help="Path used to load and save per-audio review decisions",
    )

    args = parser.parse_args()

    if args.update_index:
        logger.info("Updating index.csv with latest model predictions (%s)...", args.model_type)
        model = RecognizerFactory.create(args.model_type)
        evaluator = Evaluator(
            model=model,
            dataset_path=args.dataset_dir,
            review_decisions_path=args.review_decisions,
        )
        evaluator.update_index_with_predictions()

    if args.from_dataset_only:
        logger.info("Evaluating using recorded predicted_text in index.csv...")
        evaluator = Evaluator(
            model=None,
            dataset_path=args.dataset_dir,
            review_decisions_path=args.review_decisions,
        )
        result = evaluator.update_from_dataset()
    else:
        logger.info("Loading model (%s) and evaluating on raw audio files...", args.model_type)
        model = RecognizerFactory.create(args.model_type)
        evaluator = Evaluator(
            model=model,
            dataset_path=args.dataset_dir,
            review_decisions_path=args.review_decisions,
        )
        result = evaluator.evaluate()

    logger.info("--- Evaluation Overall Results ---")
    logger.info("Accuracy    : %.4f", result.overall.accuracy)
    logger.info("Macro F1    : %.4f", result.overall.macro_f1)
    logger.info("Weighted F1 : %.4f", result.overall.weighted_f1)
    logger.info("Total       : %d samples", result.overall.total_samples)

    if args.output_json:
        evaluator.export_json(args.output_json)

    if args.output_html:
        evaluator.export_html(
            args.output_html, title=f"モデル評価レポート ({args.model_type.upper()})"
        )

    if args.output_review_json:
        evaluator.export_review_json(args.output_review_json)

    if args.output_review_html:
        evaluator.export_review_html(
            args.output_review_html,
            title=f"Voice Data Quality Review ({args.model_type.upper()})",
            review_results_path=args.review_decisions,
        )


if __name__ == "__main__":
    main()
