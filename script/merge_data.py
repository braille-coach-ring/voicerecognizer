"""
Dataset Merging Script (Index Generation)

【設計理由: なぜインデックス (index.csv) 方式なのか】
・音声ファイル (.wav) を物理コピーしてまとめる従来方式では、ディスク容量を二重に圧迫し、
  データセット増大に伴いコピー待ち時間が数秒〜数分と無駄に伸びていました。
・本スクリプトでは Rawデータ (dataset/) および実環境 Collectedデータ (dataset/collected/) の
  ファイルパスと正解ラベル (ground_truth) のみをインデックス (merged_dataset/index.csv) に出力します。
・これにより、音声コピー時間ゼロ・ストレージ消費ゼロで一瞬で統合を完了できます。
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.dataset_builder import DatasetBuilder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== データセット統合 (インデックスマニフェスト生成) を開始します ===")

    builder = DatasetBuilder()
    index_file = builder.build_index()

    label_counts: dict[str, int] = {}
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            f.readline()  # ヘッダー読み飛ばし
            for line in f:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) >= 2 and parts[1]:
                    lbl = parts[1]
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1

    total_files = sum(label_counts.values())
    logger.info("=== 統合インデックス作成完了: %s (合計: %d 件) ===", index_file, total_files)

    for label in builder.labels:
        n = label_counts.get(label, 0)
        logger.info("  ラベル [%s] : %d 件", label, n)


if __name__ == "__main__":
    main()
