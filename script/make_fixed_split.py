"""
話者単位で学習用 / 検証用 / テスト用を固定分割し、学習用と検証用だけを前処理するスクリプト。

train.py の --train-csv / --val-csv は「前処理済みデータセットが train ∪ val と完全に一致すること」を
要求するため、テスト話者を含めない専用の前処理ディレクトリを作る。

使い方:
  uv run python script/merge_data.py
  uv run python script/make_fixed_split.py
  uv run python script/make_fixed_split.py --val-speakers yumike r9 r10 --test-speakers take r8

出力 (既定):
  data_splits/fixed/train.csv, val.csv          学習用 / 検証用マニフェスト (speaker 列付き)
  data_splits/fixed/test_eval/index.csv         テスト話者 (学習に一切使わない)
  processed_dataset_fixed/                      train ∪ val の前処理済みデータ
"""

import argparse
import csv
import logging
from collections import Counter
from pathlib import Path

from voicerecognizer.config import PROJECT_ROOT
from voicerecognizer.preprocessing.dataset_builder import DatasetBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FIELDNAMES = ["filepath", "label", "speaker", "predicted_text"]


def extract_speaker(filepath: str) -> str:
    parts = Path(filepath).parts
    if len(parts) >= 2 and parts[0] == "dataset":
        return parts[1]
    return "unknown"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s (%d rows)", path, len(rows))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--merged-index", type=Path, default=PROJECT_ROOT / "merged_dataset" / "index.csv"
    )
    parser.add_argument("--val-speakers", nargs="+", default=["yumike", "mikeryu", "r9", "r10"])
    parser.add_argument("--test-speakers", nargs="*", default=["take", "yu-ota", "reon", "r8"])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data_splits" / "fixed")
    parser.add_argument(
        "--processed-dir", type=Path, default=PROJECT_ROOT / "processed_dataset_fixed"
    )
    parser.add_argument("--skip-preprocess", action="store_true", help="CSV の書き出しだけ行う")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    val_speakers, test_speakers = set(args.val_speakers), set(args.test_speakers)
    if val_speakers & test_speakers:
        raise SystemExit(f"検証用とテスト用で話者が重複しています: {val_speakers & test_speakers}")

    with open(args.merged_index, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    splits: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        row["filepath"] = row["filepath"].replace("\\", "/")
        row["speaker"] = extract_speaker(row["filepath"])
        if row["speaker"] in test_speakers:
            splits["test"].append(row)
        elif row["speaker"] in val_speakers:
            splits["val"].append(row)
        else:
            splits["train"].append(row)

    known = {row["speaker"] for row in rows}
    missing = (val_speakers | test_speakers) - known
    if missing:
        raise SystemExit(f"merged index に存在しない話者が指定されています: {sorted(missing)}")
    if not splits["val"]:
        raise SystemExit("検証用データが 0 件です")

    for name, split_rows in splits.items():
        speakers = Counter(r["speaker"] for r in split_rows)
        labels = {r["label"] for r in split_rows}
        logger.info(
            "%-5s: %5d 件 / %3d ラベル / 話者 %s",
            name,
            len(split_rows),
            len(labels),
            dict(speakers.most_common()),
        )
    missing_val_labels = {r["label"] for r in splits["train"]} - {r["label"] for r in splits["val"]}
    if missing_val_labels:
        logger.warning("検証用に存在しないラベル: %s", sorted(missing_val_labels))

    out = args.output_dir
    write_csv(out / "train.csv", splits["train"])
    write_csv(out / "val.csv", splits["val"])
    write_csv(out / "test_eval" / "index.csv", splits["test"])
    trainval_index = out / "trainval" / "index.csv"
    write_csv(trainval_index, splits["train"] + splits["val"])

    if args.skip_preprocess:
        return
    logger.info("train ∪ val を前処理します -> %s", args.processed_dir)
    DatasetBuilder().preprocess_dataset(input_root=trainval_index, output_root=args.processed_dir)


if __name__ == "__main__":
    main()
