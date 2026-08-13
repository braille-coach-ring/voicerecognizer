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
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_RECOGNITION_CONFIG  # noqa: E402
from core.factory.recognizer_factory import RecognizerFactory  # noqa: E402
from evaluation.evaluator import Evaluator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
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

    args = parser.parse_args()

    if args.update_index:
        logger.info("Updating index.csv with latest model predictions (%s)...", args.model_type)
        model = RecognizerFactory.create(args.model_type)
        evaluator = Evaluator(model=model, dataset_path=args.dataset_dir)
        evaluator.update_index_with_predictions()

    if args.from_dataset_only:
        logger.info("Evaluating using recorded predicted_text in index.csv...")
        evaluator = Evaluator(model=None, dataset_path=args.dataset_dir)
        result = evaluator.update_from_dataset()
    else:
        logger.info("Loading model (%s) and evaluating on raw audio files...", args.model_type)
        model = RecognizerFactory.create(args.model_type)
        evaluator = Evaluator(model=model, dataset_path=args.dataset_dir)
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


if __name__ == "__main__":
    main()
