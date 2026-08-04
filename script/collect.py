from pathlib import Path
import sounddevice as sd
import soundfile as sf
import numpy as np
import subprocess
import time

from config import (
    DEFAULT_AUDIO_CONFIG,
    DEFAULT_PREPROCESS_CONFIG,
    DEFAULT_RECOGNITION_CONFIG,
)

# ==========================
# 設定（config.py から集約）
# ==========================

SAMPLE_RATE = DEFAULT_AUDIO_CONFIG.sample_rate
RECORD_SECONDS = DEFAULT_AUDIO_CONFIG.window_seconds
REPEAT = 10

LABELS = list(DEFAULT_RECOGNITION_CONFIG.labels)

ROOT = Path("dataset")

# 無音判定
SILENCE_THRESHOLD = DEFAULT_PREPROCESS_CONFIG.vad_silence_threshold

# ==========================

ROOT.mkdir(exist_ok=True)

print("=" * 40)
print("一文字音声データ収集ツール")
print("=" * 40)

name = input("名前(ID)：").strip()

while name == "":
    name = input("名前(ID)：").strip()

person_dir = ROOT / name
person_dir.mkdir(exist_ok=True)

for label in LABELS:
    (person_dir / label).mkdir(exist_ok=True)

print()
print(f"録音時間：{RECORD_SECONDS}秒")
print(f"各文字：{REPEAT}回")
input("\nEnterで開始")

# ==========================

for label in LABELS:
    print("\n" + "=" * 40)
    print(f"現在の文字：{label}")
    print("=" * 40)

    folder = person_dir / label

    files = sorted(folder.glob("*.wav"))

    numbers = []

    for f in files:
        try:
            numbers.append(int(f.stem))
        except:
            pass

    if len(numbers) == 0:
        next_number = 1
    else:
        next_number = max(numbers) + 1

    saved = 0

    while saved < REPEAT:
        print(f"\n残り {saved + 1}/{REPEAT}")

        for c in [3, 2, 1]:
            print(c)
            time.sleep(1)

        print("録音中...")

        audio = sd.rec(
            int(RECORD_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
        )

        sd.wait()

        volume = np.max(np.abs(audio))

        if volume < SILENCE_THRESHOLD:
            print("無音でした。録り直します。")
            continue

        filename = folder / f"{next_number:03d}.wav"

        sf.write(filename, audio, SAMPLE_RATE)

        print("保存:", filename)

        saved += 1
        next_number += 1

# ==========================

print("\n" + "=" * 40)
print("録音終了")
print("=" * 40)

answer = input("\nGitHubへアップロードしますか？ [Y/n] ")

if answer.lower() in ["", "y", "yes"]:
    try:
        print("\nGit Add...")
        subprocess.run(["git", "add", "."], check=True)

        print("Git Commit...")
        subprocess.run(["git", "commit", "-m", f"Add voice data ({name})"], check=True)

        print("Git Pull...")
        subprocess.run(["git", "pull", "--rebase"], check=True)

        print("Git Push...")
        subprocess.run(["git", "push"], check=True)

        print("\nアップロード完了！")

    except subprocess.CalledProcessError:
        print("\nGitHubへのアップロードに失敗しました。")

else:
    print("アップロードをスキップしました。")
