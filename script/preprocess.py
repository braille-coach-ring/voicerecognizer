"""
Audio Preprocessing Script

【設計理由: 前処理の目的】
・学習前に音声から無音部分を削除 (top_db指定) し、指定のサンプルレート (16kHz) とターゲット長 (1.0秒)
  に揃えることで、モデルの学習効率と認識精度を向上させます。
・`merged_dataset/index.csv` を入力とし、前処理後の波形を `processed_dataset/` へ出力します。
"""

import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_RECOGNITION_CONFIG
from preprocessing.dataset_builder import DatasetBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    input_dir = DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir
    output_dir = DEFAULT_RECOGNITION_CONFIG.processed_dataset_dir

    logger.info("=== 音声前処理を開始します ===")
    logger.info("  入力パス: %s", input_dir)
    logger.info("  出力パス: %s", output_dir)

    if not input_dir.exists():
        logger.warning("入力ディレクトリ/インデックスが存在しません: %s (スキップします)", input_dir)
        return

    start_time = time.time()
    builder = DatasetBuilder()
    builder.preprocess_dataset(input_root=input_dir, output_root=output_dir)
    elapsed = time.time() - start_time

    # 出力されたラベル別ファイルの件数を集計してログ表示
    total_processed = 0
    if output_dir.exists():
        for label_dir in sorted(output_dir.iterdir()):
            if label_dir.is_dir():
                count = len(list(label_dir.glob("*.wav")))
                total_processed += count
                logger.info("  処理完了ラベル [%s] : %d 件", label_dir.name, count)

    logger.info("=== 音声前処理が完了しました (合計: %d 件, 処理時間: %.2f 秒) ===", total_processed, elapsed)


if __name__ == "__main__":
    main()
