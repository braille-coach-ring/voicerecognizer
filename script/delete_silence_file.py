from pathlib import Path
import librosa
import numpy as np

# ==========================
# 設定
# ==========================

ROOT = Path("dataset")

# 最大音量がこれ未満なら無音と判定
THRESHOLD = 0.02

deleted = 0

print("無音ファイルを検索中...\n")

# ==========================
# 全フォルダ探索
# ==========================

for folder in ROOT.iterdir():
    if not folder.is_dir():
        continue

    for wav in folder.glob("*.wav"):
        y, sr = librosa.load(wav, sr=16000)

        volume = np.max(np.abs(y))

        if volume < THRESHOLD:
            print(f"削除: {wav}  (音量={volume:.5f})")

            wav.unlink()

            deleted += 1

print("\n=========================")
print(f"削除したファイル数: {deleted}")
print("=========================")
