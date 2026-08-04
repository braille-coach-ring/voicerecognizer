from pathlib import Path

from config import DEFAULT_RECOGNITION_CONFIG

# datasetフォルダの場所
root = DEFAULT_RECOGNITION_CONFIG.raw_dataset_dir

# 各文字フォルダを処理
for folder in root.iterdir():
    if folder.is_dir():
        files = sorted(folder.glob("*.wav"))

        # 一時的な名前に変更（重複防止）
        temp_names = []
        for i, file in enumerate(files):
            temp = folder / f"temp_{i:03d}.wav"
            file.rename(temp)
            temp_names.append(temp)

        # 001.wav, 002.wav... に変更
        for i, temp in enumerate(temp_names, start=1):
            new_name = folder / f"{i:03d}.wav"
            temp.rename(new_name)

        print(f"{folder.name}: {len(temp_names)}個変更")

print("完了！")
