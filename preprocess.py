from pathlib import Path
import librosa
import soundfile as sf
import numpy as np

# ===== 設定 =====
INPUT_DIR = Path("dataset")
OUTPUT_DIR = Path("processed_dataset")

SR = 16000          # サンプリング周波数
TARGET_LENGTH = 0.5 # 秒

# =================

OUTPUT_DIR.mkdir(exist_ok=True)

for label_dir in INPUT_DIR.iterdir():

    if not label_dir.is_dir():
        continue

    out_label = OUTPUT_DIR / label_dir.name
    out_label.mkdir(exist_ok=True)

    for wav_file in label_dir.glob("*.wav"):

        print("処理中:", wav_file)

        # 読み込み
        y, sr = librosa.load(wav_file, sr=SR, mono=True)

        # 無音除去
        y, _ = librosa.effects.trim(
            y,
            top_db=20
        )

        # 音量正規化
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        # 長さを統一
        target_samples = int(TARGET_LENGTH * SR)

        if len(y) > target_samples:
            y = y[:target_samples]

        elif len(y) < target_samples:
            y = np.pad(
                y,
                (0, target_samples - len(y))
            )

        # 保存
        sf.write(
            out_label / wav_file.name,
            y,
            SR
        )

print("全部終わりました！")