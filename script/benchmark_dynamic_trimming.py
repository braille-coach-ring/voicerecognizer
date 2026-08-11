import sys
import time
from pathlib import Path

# プロジェクトルートの追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.interfaces
import numpy as np
from recognizers.wav2vec2_recognizer import Wav2Vec2Recognizer


def benchmark():
    print("=== Wav2Vec2 ONNX 推論レイテンシ比較ベンチマーク ===")

    # Recognizer 初期化 (動的トリミング有効 vs 無効)
    rec_fixed = Wav2Vec2Recognizer(dynamic_trimming=False)
    rec_dynamic = Wav2Vec2Recognizer(dynamic_trimming=True)

    sr = 16000
    # 0.4秒の短音サンプルのシミュレーション（発声部 0.4秒）
    t = np.linspace(0, 0.4, int(sr * 0.4), endpoint=False)
    short_audio = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    # 5回実行ウォームアップ
    for _ in range(5):
        rec_fixed.recognize(short_audio)
        rec_dynamic.recognize(short_audio)

    # 1.0s 固定長パディングの推論時間測定 (20回平均)
    times_fixed = []
    for _ in range(20):
        t0 = time.perf_counter()
        rec_fixed.recognize(short_audio)
        t1 = time.perf_counter()
        times_fixed.append((t1 - t0) * 1000)

    # 動的トリミング入力の推論時間測定 (20回平均)
    times_dynamic = []
    for _ in range(20):
        t0 = time.perf_counter()
        rec_dynamic.recognize(short_audio)
        t1 = time.perf_counter()
        times_dynamic.append((t1 - t0) * 1000)

    avg_fixed = float(np.mean(times_fixed))
    avg_dynamic = float(np.mean(times_dynamic))
    speedup = avg_fixed / avg_dynamic if avg_dynamic > 0 else 0.0

    print(f"1.0秒固定長パディング (Fixed 1.0s) : {avg_fixed:.2f} ms")
    print(f"動的トリミング入力   (Dynamic ~0.4s): {avg_dynamic:.2f} ms")
    print(f"高速化率: {speedup:.2f} 倍 (レイテンシ {avg_fixed - avg_dynamic:.2f} ms 削減)")


if __name__ == "__main__":
    benchmark()
