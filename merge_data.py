from pathlib import Path
import shutil

# ==========================
# 設定
# ==========================

SOURCE_ROOT = Path("dataset")
OUTPUT_ROOT = Path("merged_dataset")

LABELS = [
    "a",
    "i",
    "u",
    "e",
    "o"
]

# ==========================

# 出力フォルダ作成
OUTPUT_ROOT.mkdir(exist_ok=True)

for label in LABELS:
    folder = OUTPUT_ROOT / label

    if folder.exists():
        shutil.rmtree(folder)

    folder.mkdir(parents=True)

# ==========================
# コピー開始
# ==========================

print("=" * 40)
print("データセットを統合します")
print("=" * 40)

for label in LABELS:

    output_folder = OUTPUT_ROOT / label

    count = 1

    for person in sorted(SOURCE_ROOT.iterdir()):

        if not person.is_dir():
            continue

        input_folder = person / label

        if not input_folder.exists():
            continue

        wav_files = sorted(input_folder.glob("*.wav"))

        for wav in wav_files:

            dst = output_folder / f"{count:03d}.wav"

            shutil.copy2(
                wav,
                dst
            )

            print(
                f"{person.name}/{label}/{wav.name} -> {dst.name}"
            )

            count += 1

print()
print("=" * 40)
print("統合完了")
print("=" * 40)

# 件数表示
print()

for label in LABELS:

    n = len(list((OUTPUT_ROOT / label).glob("*.wav")))

    print(f"{label} : {n} files")