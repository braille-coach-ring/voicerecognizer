from pathlib import Path
import librosa
import soundfile as sf
import numpy as np

# ==========================
# 設定
# ==========================

INPUT_DIR = Path("merged_dataset")
OUTPUT_DIR = Path("processed_dataset")

SR = 16000
TARGET_LENGTH = 1.0      # 秒
TOP_DB = 20

# ==========================

OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 40)
print("音声データ前処理")
print("=" * 40)

for label_dir in sorted(INPUT_DIR.iterdir()):

    if not label_dir.is_dir():
        continue

    out_dir = OUTPUT_DIR / label_dir.name
    out_dir.mkdir(exist_ok=True)

    wav_files = sorted(label_dir.glob("*.wav"))

    print(f"\n{label_dir.name}")

    for wav in wav_files:

        print(f"  {wav.name}")

        # ----------------------
        # 読み込み
        # ----------------------

        y, sr = librosa.load(
            wav,
            sr=SR,
            mono=True
        )

        # ----------------------
        # 無音除去
        # ----------------------

        y, _ = librosa.effects.trim(
            y,
            top_db=TOP_DB
        )

        # ----------------------
        # 音量正規化
        # ----------------------

        if np.max(np.abs(y)) > 0:

            y = y / np.max(np.abs(y))

        # ----------------------
        # 長さ統一
        # ----------------------

        target_samples = int(TARGET_LENGTH * SR)

        if len(y) > target_samples:

            y = y[:target_samples]

        elif len(y) < target_samples:

            y = np.pad(
                y,
                (0, target_samples - len(y))
            )

        # ----------------------
        # 保存
        # ----------------------

        out_file = out_dir / wav.name

        sf.write(
            out_file,
            y,
            SR
        )

print()
print("=" * 40)
print("前処理完了")
print("=" * 40)