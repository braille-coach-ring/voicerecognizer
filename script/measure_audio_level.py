import logging
from time import sleep

import numpy as np
import sounddevice as sd
import soundfile as sf

from config import DEFAULT_AUDIO_CONFIG

logger = logging.getLogger(__name__)

# ==========================
# 設定（config.py から参照）
# ==========================

SR = DEFAULT_AUDIO_CONFIG.sample_rate  # サンプリングレート
RECORD_SECONDS = 3.0  # 録音時間（3秒）
CHUNK_SIZE = int(
    DEFAULT_AUDIO_CONFIG.sample_rate * DEFAULT_AUDIO_CONFIG.chunk_seconds
)  # フレームサイズ
AUDIO_FILE = "measured_audio.wav"  # 保存先ファイル（毎回上書き）


def main() -> None:
    # ==========================
    # カウントダウン
    # ==========================

    print("マイク音量レベルを測定します")
    print("3秒間の音声を記録します\n")

    print("3秒後に計測開始...")
    sleep(1)
    print("3")
    sleep(1)
    print("2")
    sleep(1)
    print("1")
    sleep(1)

    print("発声してください（自然な会話レベルで）\n")

    # ==========================
    # 録音
    # ==========================

    audio = sd.rec(int(RECORD_SECONDS * SR), samplerate=SR, channels=1, dtype="float32")

    sd.wait()

    logger.info("マイク計測完了")

    # (16000*3, 1) → (16000*3,)
    y = audio.flatten()

    # ==========================
    # 録音ファイルを保存
    # ==========================

    sf.write(AUDIO_FILE, y, SR)
    logger.info("📁 録音ファイルを保存: %s", AUDIO_FILE)

# ==========================
# フレームごとの音量計算
# ==========================

print("=" * 60)
print("フレーム別の音量レベル")
print("=" * 60)

frame_volumes = []
frame_peaks = []
frame_dbs = []

for i in range(0, len(y), CHUNK_SIZE):
    chunk = y[i : i + CHUNK_SIZE]

    if len(chunk) == 0:
        continue

    # ピーク値（最大振幅）
    peak = np.max(np.abs(chunk))

    # RMS（二乗平均平方根）
    rms = np.sqrt(np.mean(chunk**2))

    # デシベル値（最大値を基準）
    db = 20 * np.log10(rms + 1e-10)

    frame_volumes.append(rms)
    frame_peaks.append(peak)
    frame_dbs.append(db)

    time_sec = i / SR

    print(f"[{time_sec:5.2f}s] RMS: {rms:.4f}  Peak: {peak:.4f}  dB: {db:7.2f}")

# ==========================
# 統計情報
# ==========================

print("\n" + "=" * 60)
print("統計情報（全体）")
print("=" * 60)

print(f"\n【RMS値（平均音量）】")
print(f"  平均: {np.mean(frame_volumes):.4f}")
print(f"  最大: {np.max(frame_volumes):.4f}")
print(f"  最小: {np.min(frame_volumes):.4f}")
print(f"  標準偏差: {np.std(frame_volumes):.4f}")

print(f"\n【ピーク値（最大振幅）】")
print(f"  平均: {np.mean(frame_peaks):.4f}")
print(f"  最大: {np.max(frame_peaks):.4f}")
print(f"  最小: {np.min(frame_peaks):.4f}")

print(f"\n【デシベル値】")
print(f"  平均: {np.mean(frame_dbs):.2f} dB")
print(f"  最大: {np.max(frame_dbs):.2f} dB")
print(f"  最小: {np.min(frame_dbs):.2f} dB")

# ==========================
# 無音判定の推奨値
# ==========================

print("\n" + "=" * 60)
print("推奨されるしきい値（TOP_DB設定値）")
print("=" * 60)

# 最小RMSの1.5倍を目安
silence_threshold_rms = np.min(frame_volumes) * 1.5
# デシベル値に変換
silence_threshold_db = 20 * np.log10(silence_threshold_rms + 1e-10)

print(f"\n無音と判定する基準:")
print(f"  現在の最小RMS値: {np.min(frame_volumes):.4f}")
print(f"  推奨RMS値: {silence_threshold_rms:.4f}")
print(f"  推奨TOP_DB値: {silence_threshold_db:.2f} dB")

print(f"\n（現在のpreprocess.pyでは TOP_DB = 20 に設定されています）")

# ==========================
# グラフ描画
# ==========================

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    time_axis = np.arange(len(frame_volumes)) * (CHUNK_SIZE / SR)

    # RMS値
    axes[0].plot(time_axis, frame_volumes, marker="o", label="RMS")
    axes[0].axhline(y=np.mean(frame_volumes), color="r", linestyle="--", label="平均")
    axes[0].set_ylabel("RMS値")
    axes[0].set_title("フレーム別RMS値（平均音量）")
    axes[0].legend()
    axes[0].grid(True)

    # ピーク値
    axes[1].plot(time_axis, frame_peaks, marker="s", color="orange", label="Peak")
    axes[1].axhline(y=np.mean(frame_peaks), color="r", linestyle="--", label="平均")
    axes[1].set_ylabel("ピーク値")
    axes[1].set_title("フレーム別ピーク値（最大振幅）")
    axes[1].legend()
    axes[1].grid(True)

    # デシベル値
    axes[2].plot(time_axis, frame_dbs, marker="^", color="green", label="dB")
    axes[2].axhline(y=np.mean(frame_dbs), color="r", linestyle="--", label="平均")
    axes[2].set_xlabel("時間（秒）")
    axes[2].set_ylabel("dB値")
    axes[2].set_title("フレーム別デシベル値")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("Docs/charts/audio_level_measurement.png")
    logger.info("グラフを 'Docs/charts/audio_level_measurement.png' に保存しました")
    plt.close()

except ImportError:
    logger.warning("matplotlib がインストールされていないため、グラフ生成をスキップしました")


if __name__ == "__main__":
    main()


# ==========================
# 使用例
# ==========================

print("\n" + "=" * 60)
print("結果の使い方")
print("=" * 60)
print(
    """
【保存されたファイル】
  • measured_audio.wav
    └─ 実際に録音した音声ファイル
    └─ 毎回実行時に上書きされるため、容量は約1MB で固定
    └─ 「音量が十分か」「ノイズレベルはどのくらいか」を確認できます

  • audio_level_measurement.png
    └─ 音量分析のグラフ

【preprocess.py の設定を更新する場合】
preprocess.py の以下の行を修正してください：

  TOP_DB = 20  # 現在の値

以下のようにします（推奨値に基づいて）：
  TOP_DB = {:.1f}  # 推奨値

TOP_DB が小さいほど、より多くの無音部分を除去します
TOP_DB が大きいほど、より少ない部分だけ除去します

あなたのマイク・発声レベルに合わせて調整してください
""".format(silence_threshold_db)
)
