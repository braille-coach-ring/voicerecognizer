from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from config import DEFAULT_AUDIO_CONFIG

# ==========================
# 設定
# ==========================

INPUT_DIR = Path("merged_dataset")

OUTPUT_DIR = Path("processed_dataset")

SR = DEFAULT_AUDIO_CONFIG.sample_rate
CHUNK_SECONDS = DEFAULT_AUDIO_CONFIG.chunk_seconds
TOP_DB = DEFAULT_AUDIO_CONFIG.top_db

# ==========================

# 出力フォルダ作り直し
if OUTPUT_DIR.exists():
    import shutil

    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir()

print("=" * 40)
print("音声データ前処理")
print("=" * 40)


if not INPUT_DIR.exists():
    print(f"\n{INPUT_DIR} は存在しないのでスキップ")
    exit()

print(f"\n===== {INPUT_DIR} =====")

for label_dir in sorted(INPUT_DIR.iterdir()):
    if not label_dir.is_dir():
        continue

    out_dir = OUTPUT_DIR / label_dir.name
    out_dir.mkdir(exist_ok=True)

    file_number = len(list(out_dir.glob("*.wav"))) + 1

    print(f"\n{label_dir.name}")

    for wav in sorted(label_dir.glob("*.wav")):
        print(f"  {wav.name}")

        try:
            y, _ = librosa.load(wav, sr=SR, mono=True)
        except Exception as e:
            print("読み込み失敗:", wav)
            print(e)
            continue

        # 無音除去
        y, _ = librosa.effects.trim(y, top_db=TOP_DB)

        # 音量正規化
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        # 長さ統一
        target_samples = int(CHUNK_SECONDS * SR)

        if len(y) > target_samples:
            y = y[:target_samples]
        else:
            y = np.pad(y, (0, target_samples - len(y)))

        # -------------------------
        # 同名ファイルがあれば番号を振る
        # -------------------------

        out_file = out_dir / f"{file_number:03d}.wav"

        sf.write(out_file, y, SR)

        file_number += 1

print()
print("=" * 40)
print("前処理完了")
print("=" * 40)
