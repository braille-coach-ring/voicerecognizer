import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_RECOGNITION_CONFIG
from preprocessing.dataset_builder import DatasetBuilder


def main() -> None:
    print("=" * 40)
    print("音声データ前処理")
    print("=" * 40)

    input_dir = DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir
    output_dir = DEFAULT_RECOGNITION_CONFIG.processed_dataset_dir

    if not input_dir.exists():
        print(f"\n{input_dir} は存在しないのでスキップ")
        return

    builder = DatasetBuilder()
    builder.preprocess_dataset(input_root=input_dir, output_root=output_dir)

    print()
    print("=" * 40)
    print("前処理完了")
    print("=" * 40)


if __name__ == "__main__":
    main()

