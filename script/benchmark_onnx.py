"""
ONNX Model Benchmark & Evaluation Comparison Script (script/benchmark_onnx.py)

役割:
  1. Standard ONNX FP32 (model_fp32.onnx)
  2. Standard ONNX INT8 (model_int8.onnx)
  3. Mel-Prepended ONNX INT8 (model_mel_int8.onnx)
  の3種類のモデル構造について、全テストデータセットでの評価精度 (Macro-F1, Accuracy)、
  前処理・推論レイテンシ (ms)、モデルファイル容量 (MB) を一括計測・比較し、
  evaluation_results/onnx_benchmark_report.json に結果を出力します。

使い方:
  uv run python script/benchmark_onnx.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

from config import DEFAULT_RECOGNITION_CONFIG  # noqa: E402
from evaluation.evaluator import Evaluator  # noqa: E402
from recognizers.wav2vec2_recognizer import Wav2Vec2Recognizer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def benchmark_model_variant(
    model_dir: Path,
    target_onnx_filename: str,
    dataset_dir: Path,
    num_samples: int = 50,
) -> dict[str, Any]:
    """特定の ONNX モデルファイルバリアントの精度と推論スピードを計測します。"""
    onnx_file = model_dir / target_onnx_filename
    if not onnx_file.exists():
        logger.warning("ONNX ファイルが存在しません: %s", onnx_file)
        return {}

    logger.info("=== バリアント計測中: %s ===", target_onnx_filename)

    # 1. 認識器のロード
    rec = Wav2Vec2Recognizer(model_path=model_dir)
    rec.onnx_model_path = onnx_file
    rec._ensure_model_loaded()

    # 2. 全データセット評価 (Accuracy / Macro-F1)
    evaluator = Evaluator(model=rec, dataset_path=dataset_dir)
    eval_result = evaluator.evaluate()

    # 3. 推論スピード・前処理時間の詳細プロファイリング (num_samples 回)
    prep_latencies: list[float] = []
    inf_latencies: list[float] = []
    total_latencies: list[float] = []

    # ウォームアップ
    dummy_audio = np.random.randn(9600).astype(np.float32)
    for _ in range(5):
        rec.recognize(dummy_audio)

    # プロファイリング計測
    for _ in range(num_samples):
        rec.recognize(dummy_audio)
        stats = rec.last_timing_stats
        prep_latencies.append(stats.get("prep_latency_ms", 0.0))
        inf_latencies.append(stats.get("inference_latency_ms", 0.0))
        total_latencies.append(stats.get("total_latency_ms", 0.0))

    avg_prep_ms = float(np.mean(prep_latencies))
    avg_inf_ms = float(np.mean(inf_latencies))
    avg_total_ms = float(np.mean(total_latencies))
    file_size_mb = float(onnx_file.stat().st_size / (1024 * 1024))

    return {
        "filename": target_onnx_filename,
        "input_type": "waveform" if rec.input_name == "waveform" else "input_values",
        "file_size_mb": round(file_size_mb, 2),
        "accuracy": round(eval_result.overall.accuracy, 4),
        "macro_f1": round(eval_result.overall.macro_f1, 4),
        "avg_prep_latency_ms": round(avg_prep_ms, 3),
        "avg_inference_latency_ms": round(avg_inf_ms, 3),
        "avg_total_latency_ms": round(avg_total_ms, 3),
    }


def main() -> None:
    model_dir = DEFAULT_RECOGNITION_CONFIG.wav2vec2_best_model_dir
    dataset_dir = DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir
    output_report_path = PROJECT_ROOT / "evaluation_results" / "onnx_benchmark_report.json"

    variants = [
        "model_mel_int8.onnx",
        "model_int8.onnx",
        "model_fp32.onnx",
    ]

    results = {}
    for var in variants:
        res = benchmark_model_variant(model_dir, var, dataset_dir)
        if res:
            results[var] = res

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("ベンチマーク完了。結果を保存しました: %s", output_report_path)

    print("\n" + "=" * 70)
    print(" 🚀 Wav2Vec2 ONNX モデル比較ベンチマーク評価レポート")
    print("=" * 70)
    print(
        f"{'モデルファイル':<22} | {'入力形式':<10} | {'容量(MB)':<8} | {'Accuracy':<8} | {'Macro-F1':<8} | {'推論時間(ms)':<10} | {'前処理(ms)':<8}"
    )
    print("-" * 70)
    for data in results.values():
        print(
            f"{data['filename']:<22} | "
            f"{data['input_type']:<10} | "
            f"{data['file_size_mb']:<8} | "
            f"{data['accuracy']:<8} | "
            f"{data['macro_f1']:<8} | "
            f"{data['avg_inference_latency_ms']:<10} | "
            f"{data['avg_prep_latency_ms']:<8}"
        )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
