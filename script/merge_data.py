import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_RECOGNITION_CONFIG
from preprocessing.dataset_builder import DatasetBuilder


def main() -> None:
    print("=" * 40)
    print("データセットを統合します")
    print("=" * 40)

    source_root = DEFAULT_RECOGNITION_CONFIG.raw_dataset_dir
    output_root = DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir

    builder = DatasetBuilder()
    builder.merge_by_label(source_root=source_root, output_root=output_root)

    print()
    print("=" * 40)
    print("統合完了")
    print("=" * 40)

    for label in builder.labels:
        folder = output_root / label
        n = len(list(folder.glob("*.wav"))) if folder.exists() else 0
        print(f"{label} : {n} files")


if __name__ == "__main__":
    main()
