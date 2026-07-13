from pathlib import Path
import sounddevice as sd
import soundfile as sf
import numpy as np
import time

# ==========================
# 設定
# ==========================

SAMPLE_RATE = 16000
RECORD_SECONDS = 1.0
REPEAT = 10
# 無音判定（RMS）
SILENCE_THRESHOLD = 0.005

LABELS = [
    "a",
    "i",
    "u",
    "e",
    "o"
]

ROOT = Path("dataset")

# ==========================
# datasetフォルダ作成
# ==========================

ROOT.mkdir(exist_ok=True)

for label in LABELS:
    (ROOT / label).mkdir(exist_ok=True)

print("=" * 40)
print("一文字音声データ収集ツール")
print("=" * 40)

print(f"\n録音時間 : {RECORD_SECONDS}秒")
print(f"録音回数 : {REPEAT}回")

input("\nEnterキーで開始")

# ==========================
# 録音開始
# ==========================

for label in LABELS:

    print("\n" + "=" * 40)
    print(f"「{label}」を発音してください")
    print("=" * 40)

    folder = ROOT / label

    # 現在の最大番号を取得
    files = sorted(folder.glob("*.wav"))

    if len(files) == 0:
        start = 1
    else:
        numbers = []

        for f in files:

            try:
                numbers.append(int(f.stem))
            except:
                pass

        if len(numbers) == 0:
            start = 1
        else:
            start = max(numbers) + 1

    # 録音
    for i in range(REPEAT):

        print(f"\n残り {i+1}/{REPEAT}")

        print("3...")
        time.sleep(1)

        print("2...")
        time.sleep(1)

        print("1...")
        time.sleep(1)

        print("録音中...")

        audio = sd.rec(
            int(RECORD_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32
        )

        sd.wait()

        # 1次元配列に変換
        audio = audio.flatten()

        # RMS（実効音量）を計算
        rms = np.sqrt(np.mean(audio ** 2))

        # 無音判定
        if rms < SILENCE_THRESHOLD:
            print(f"無音のため保存しませんでした (RMS={rms:.5f})")
            continue

        filename = folder / f"{start+i:03d}.wav"

        sf.write(
            filename,
            audio,
            SAMPLE_RATE
        )

        print(f"保存: {filename} (RMS={rms:.5f})")


print("\n==============================")
print("すべて終了しました！")
print("==============================")