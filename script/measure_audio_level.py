"""
Audio Input Level & Noise Floor Measurement Utility

Usage:
  uv run python script/measure_audio_level.py
"""

import logging
from pathlib import Path
import sys
from time import sleep

import numpy as np
import sounddevice as sd
import soundfile as sf

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import DEFAULT_AUDIO_CONFIG, DEFAULT_PREPROCESS_CONFIG


logger = logging.getLogger(__name__)

# ==========================
# 設定（config.py から参照）
# ==========================

SR = DEFAULT_AUDIO_CONFIG.sample_rate
RECORD_SECONDS = 3.0
CHUNK_SIZE = int(DEFAULT_AUDIO_CONFIG.sample_rate * DEFAULT_AUDIO_CONFIG.chunk_seconds)
AUDIO_FILE = Path("measured_audio.wav")
CHART_FILE = Path("Docs") / "charts" / "audio_level_measurement.png"


def main() -> None:
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

    audio = sd.rec(int(RECORD_SECONDS * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    logger.info("マイク計測完了")

    y = audio.flatten()
    sf.write(str(AUDIO_FILE), y, SR)
    logger.info("録音ファイルを保存: %s", AUDIO_FILE)

    print("=" * 60)
    print("フレーム別の音量レベル")
    print("=" * 60)

    frame_volumes: list[float] = []
    frame_peaks: list[float] = []
    frame_dbs: list[float] = []

    for i in range(0, len(y), CHUNK_SIZE):
        chunk = y[i : i + CHUNK_SIZE]
        if len(chunk) == 0:
            continue

        peak = float(np.max(np.abs(chunk)))
        rms = float(np.sqrt(np.mean(chunk**2)))
        db = float(20 * np.log10(rms + 1e-10))

        frame_volumes.append(rms)
        frame_peaks.append(peak)
        frame_dbs.append(db)

        time_sec = i / SR
        print(f"[{time_sec:5.2f}s] RMS: {rms:.4f}  Peak: {peak:.4f}  dB: {db:7.2f}")

    if not frame_volumes:
        raise RuntimeError("録音データが空のため、音量分析を実行できませんでした。")

    print("\n" + "=" * 60)
    print("統計情報（全体）")
    print("=" * 60)
    print("\n【RMS値（平均音量）】")
    print(f"  平均: {np.mean(frame_volumes):.4f}")
    print(f"  最大: {np.max(frame_volumes):.4f}")
    print(f"  最小: {np.min(frame_volumes):.4f}")
    print(f"  標準偏差: {np.std(frame_volumes):.4f}")
    print("\n【ピーク値（最大振幅）】")
    print(f"  平均: {np.mean(frame_peaks):.4f}")
    print(f"  最大: {np.max(frame_peaks):.4f}")
    print(f"  最小: {np.min(frame_peaks):.4f}")
    print("\n【デシベル値】")
    print(f"  平均: {np.mean(frame_dbs):.2f} dB")
    print(f"  最大: {np.max(frame_dbs):.2f} dB")
    print(f"  最小: {np.min(frame_dbs):.2f} dB")

    print("\n" + "=" * 60)
    print("推奨されるしきい値（TOP_DB設定値）")
    print("=" * 60)

    # ほぼ無音のフレームに引っ張られすぎないよう下位20%点を使う
    noise_floor_rms = float(np.percentile(frame_volumes, 20))
    silence_threshold_rms = max(noise_floor_rms * 1.5, 1e-6)
    silence_threshold_db = float(20 * np.log10(silence_threshold_rms))
    peak_rms = max(float(np.max(frame_volumes)), 1e-6)
    recommended_top_db = float(20 * np.log10(peak_rms / silence_threshold_rms))
    noise_floor_peak = float(np.percentile(frame_peaks, 20))
    recommended_vad_silence_threshold = max(noise_floor_peak * 1.5, 1e-4)
    recommended_min_top_db = max(5.0, recommended_top_db - 8.0)
    recommended_max_top_db = min(80.0, recommended_top_db + 8.0)

    print("\n無音と判定する基準:")
    print(f"  ノイズ床RMS値（下位20%点）: {noise_floor_rms:.4f}")
    print(f"  推奨RMS値: {silence_threshold_rms:.4f}")
    print(f"  参考dB値（絶対レベル）: {silence_threshold_db:.2f} dB")
    print(f"  ノイズ床Peak値（下位20%点）: {noise_floor_peak:.4f}")
    print(f"  推奨vad_silence_threshold（VAD用）: {recommended_vad_silence_threshold:.4f}")
    print(f"  推奨TOP_DB値（trim用）: {recommended_top_db:.2f}")
    print(f"  推奨min_top_db（動的モード下限）: {recommended_min_top_db:.2f}")
    print(f"  推奨max_top_db（動的モード上限）: {recommended_max_top_db:.2f}")
    print(
        f"\n（現在のconfig.pyでは min_top_db = {DEFAULT_PREPROCESS_CONFIG.min_top_db}, "
        f"max_top_db = {DEFAULT_PREPROCESS_CONFIG.max_top_db}, "
        f"vad_silence_threshold = {DEFAULT_PREPROCESS_CONFIG.vad_silence_threshold} "
        f"に設定されています）"
    )

    try:
        import matplotlib.pyplot as plt

        CHART_FILE.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        time_axis = np.arange(len(frame_volumes)) * (CHUNK_SIZE / SR)

        axes[0].plot(time_axis, frame_volumes, marker="o", label="RMS")
        axes[0].axhline(y=np.mean(frame_volumes), color="r", linestyle="--", label="Mean")
        axes[0].set_ylabel("RMS")
        axes[0].set_title("RMS by Frame")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(time_axis, frame_peaks, marker="s", color="orange", label="Peak")
        axes[1].axhline(y=np.mean(frame_peaks), color="r", linestyle="--", label="Mean")
        axes[1].set_ylabel("Peak")
        axes[1].set_title("Peak by Frame")
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(time_axis, frame_dbs, marker="^", color="green", label="dB")
        axes[2].axhline(y=np.mean(frame_dbs), color="r", linestyle="--", label="Mean")
        axes[2].set_xlabel("Time (sec)")
        axes[2].set_ylabel("dB")
        axes[2].set_title("dB by Frame")
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(CHART_FILE)
        logger.info("グラフを保存: %s", CHART_FILE)
        plt.close()

    except ImportError:
        logger.warning("matplotlib がインストールされていないため、グラフ生成をスキップしました")


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

  • Docs/charts/audio_level_measurement.png
    └─ 音量分析のグラフ

【preprocess.py の設定を更新する場合】
preprocess.py の以下の行を修正してください：

  TOP_DB = 20  # 現在の値

以下のようにします（推奨値に基づいて）：
  TOP_DB = {:.1f}  # 推奨値（正の値）
  MIN_TOP_DB = {:.1f}  # 推奨下限（動的モード）
  MAX_TOP_DB = {:.1f}  # 推奨上限（動的モード）
  VAD_SILENCE_THRESHOLD = {:.4f}  # 推奨値（マイク誤検知対策）

TOP_DB が小さいほど、より多くの無音部分を除去します
TOP_DB が大きいほど、より少ない部分だけ除去します

あなたのマイク・発声レベルに合わせて調整してください
""".format(
            recommended_top_db,
            recommended_min_top_db,
            recommended_max_top_db,
            recommended_vad_silence_threshold,
        )
    )


if __name__ == "__main__":
    main()
