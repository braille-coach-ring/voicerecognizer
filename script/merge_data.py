import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_RECOGNITION_CONFIG
from preprocessing.dataset_builder import DatasetBuilder


def main() -> None:
    print("=" * 40)
    print("データセットを統合（インデックス作成）します")
    print("=" * 40)

    builder = DatasetBuilder()
    index_file = builder.build_index()

    print()
    print("=" * 40)
    print("統合インデックス作成完了:", index_file)
    print("=" * 40)

    label_counts: dict[str, int] = {}
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) >= 2 and parts[1]:
                    lbl = parts[1]
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1

    for label in builder.labels:
        n = label_counts.get(label, 0)
        print(f"{label} : {n} files")


if __name__ == "__main__":
    main()
