"""
Interactive Voice Dataset Collector Script

Usage:
  uv run python script/collect.py
"""

import argparse
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from config import (
    DEFAULT_AUDIO_CONFIG,
    DEFAULT_PREPROCESS_CONFIG,
    DEFAULT_RECOGNITION_CONFIG,
)
from config_labels import ROMAJI_TO_HIRAGANA

logger = logging.getLogger(__name__)

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

def run_collection() -> None:
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
        hiragana = ROMAJI_TO_HIRAGANA.get(label, label)
        disp_text = f"{label} ({hiragana})" if hiragana != label else label
        print("\n" + "=" * 40)
        print(f"現在の文字：{disp_text}")
        print("=" * 40)

        folder = person_dir / label

        files = sorted(folder.glob("*.wav"))

        numbers = []

        for f in files:
            try:
                numbers.append(int(f.stem))
            except Exception:
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
                logger.warning("無音を検知しました (最大音量: %.4f < 閾値: %.4f)。録り直します。", volume, SILENCE_THRESHOLD)
                continue

            filename = folder / f"{next_number:03d}.wav"

            sf.write(filename, audio, SAMPLE_RATE)

            logger.info("保存完了: %s", filename)

            saved += 1
            next_number += 1

    # ==========================

    print("\n" + "=" * 40)
    print("録音終了")
    print("=" * 40)

    answer = input("\nGitHubへアップロードしますか？ [Y/n] ")

    if answer.lower() in ["", "y", "yes"]:
        try:
            logger.info("Git Add...")
            subprocess.run(["git", "add", "."], check=True)

            logger.info("Git Commit...")
            subprocess.run(["git", "commit", "-m", f"Add voice data ({name})"], check=True)

            logger.info("Git Pull...")
            subprocess.run(["git", "pull", "--rebase"], check=True)

            logger.info("Git Push...")
            subprocess.run(["git", "push"], check=True)

            logger.info("アップロード完了！")

        except subprocess.CalledProcessError as e:
            logger.error("GitHubへのアップロードに失敗しました: %s", e, exc_info=True)

    else:
        logger.info("アップロードをスキップしました。")


if __name__ == "__main__":
    run_collection()

