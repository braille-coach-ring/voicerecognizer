from pathlib import Path
import librosa
import numpy as np

from config import (
    DEFAULT_AUDIO_CONFIG,
    DEFAULT_PREPROCESS_CONFIG,
    DEFAULT_RECOGNITION_CONFIG,
)

# ==========================
# 設定（config.py から参照）
# ==========================

ROOT = DEFAULT_RECOGNITION_CONFIG.raw_dataset_dir

# 最大音量がこれ未満なら無音と判定
THRESHOLD = DEFAULT_PREPROCESS_CONFIG.vad_silence_threshold

deleted = 0

print("無音ファイルを検索中...\n")

# ==========================
# 全フォルダ探索
# ==========================

for folder in ROOT.iterdir():
    if not folder.is_dir():
        continue

    for wav in folder.glob("*.wav"):
        y, sr = librosa.load(wav, sr=DEFAULT_AUDIO_CONFIG.sample_rate)

        volume = np.max(np.abs(y))

        if volume < THRESHOLD:
            print(f"削除: {wav}  (音量={volume:.5f})")

            wav.unlink()

            deleted += 1

print("\n=========================")
print(f"削除したファイル数: {deleted}")
print("=========================")
