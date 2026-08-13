"""
Wav2Vec2 ONNX Export & Benchmark Script

役割:
  Fine-tuned Wav2Vec2 PyTorch モデルを ONNX フォーマットにエクスポートし、
  ONNX Runtime の Graph Optimization により CPU 推論を高速化します (model_fp32.onnx)。
  また、`labels.json` が欠損している場合は自動的に生成・修復します。

使い方:
  uv run python models/wav2vec2/export_onnx.py
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from config import DEFAULT_RECOGNITION_CONFIG  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def export_to_onnx(
    model: torch.nn.Module,
    output_onnx_path: Path,
    sample_rate: int = DEFAULT_RECOGNITION_CONFIG.sample_rate,
    target_length_seconds: float = DEFAULT_RECOGNITION_CONFIG.target_length_seconds,
) -> None:
    """PyTorch Wav2Vec2 モデルを ONNX 形式にエクスポートします。"""
    model.eval()
    num_samples = int(sample_rate * target_length_seconds)
    dummy_input = torch.randn(1, num_samples, dtype=torch.float32)

    logger.info("PyTorch モデルを ONNX フォーマットへエクスポート中: %s", output_onnx_path)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={
            "input_values": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        dynamo=False,
    )
    logger.info(
        "ONNX エクスポート完了: %s (%.2f MB)",
        output_onnx_path,
        output_onnx_path.stat().st_size / (1024 * 1024),
    )


def quantize_onnx_int8(fp32_onnx_path: Path, int8_onnx_path: Path) -> None:
    """ONNX モデルを INT8 動的量子化します (MatMul / Attention レイヤーをターゲットとし低レイテンシ化)。"""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise ImportError("ONNX quantization requires 'onnxruntime'.") from exc

    logger.info("ONNX INT8 動的量子化を適用中 (MatMul/Attention ターゲット)...")
    quantize_dynamic(
        model_input=str(fp32_onnx_path),
        model_output=str(int8_onnx_path),
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=["MatMul", "Gather", "Attention"],
    )
    logger.info(
        "INT8 量子化 ONNX モデル生成完了: %.2f MB", int8_onnx_path.stat().st_size / (1024 * 1024)
    )


def run_benchmark(
    pytorch_model: torch.nn.Module,
    fp32_onnx_path: Path,
    int8_onnx_path: Path | None = None,
    iterations: int = 30,
) -> tuple[float, float, float | None]:
    """PyTorch FP32, ONNX FP32, (オプション) ONNX INT8 の CPU 推論レイテンシを比較計測します。"""
    import onnxruntime as ort

    dummy_audio = np.random.randn(16000).astype(np.float32)
    dummy_tensor = torch.from_numpy(dummy_audio).unsqueeze(0)

    # 1. PyTorch
    pytorch_model.eval()
    with torch.no_grad():
        for _ in range(5):
            _ = pytorch_model(dummy_tensor)
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = pytorch_model(dummy_tensor)
        pt_time = (time.perf_counter() - t0) / iterations * 1000.0

    # ONNX SessionOptions
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    cpu_threads = max(1, min(4, os.cpu_count() or 4))
    so.intra_op_num_threads = cpu_threads
    so.inter_op_num_threads = 1

    # 2. ONNX FP32
    sess_fp32 = ort.InferenceSession(str(fp32_onnx_path), so, providers=["CPUExecutionProvider"])
    inp_name = sess_fp32.get_inputs()[0].name
    for _ in range(5):
        _ = sess_fp32.run(None, {inp_name: dummy_audio[np.newaxis, :]})
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = sess_fp32.run(None, {inp_name: dummy_audio[np.newaxis, :]})
    onnx_fp32_time = (time.perf_counter() - t0) / iterations * 1000.0

    # 3. ONNX INT8 (指定された場合のみ)
    onnx_int8_time: float | None = None
    if int8_onnx_path and int8_onnx_path.exists():
        sess_int8 = ort.InferenceSession(
            str(int8_onnx_path), so, providers=["CPUExecutionProvider"]
        )
        for _ in range(5):
            _ = sess_int8.run(None, {inp_name: dummy_audio[np.newaxis, :]})
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = sess_int8.run(None, {inp_name: dummy_audio[np.newaxis, :]})
        onnx_int8_time = (time.perf_counter() - t0) / iterations * 1000.0

    return pt_time, onnx_fp32_time, onnx_int8_time


def export_and_benchmark(
    model_dir: Path | str = DEFAULT_RECOGNITION_CONFIG.wav2vec2_best_model_dir,
    export_int8: bool = True,
) -> Path:
    """Wav2Vec2 チェックポイントを ONNX 化 (FP32 & INT8) し、labels.json を自動生成・修復します。"""
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Wav2Vec2 モデルディレクトリが存在しません: {model_path}")

    try:
        from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
    except ImportError as exc:
        raise ImportError("Wav2Vec2 エクスポートには 'transformers' が必要です。") from exc

    labels_file = model_path / "labels.json"
    labels = None

    if labels_file.exists():
        try:
            with open(labels_file, "r", encoding="utf-8") as f:
                labels = json.load(f)
        except Exception:
            pass

    if not labels:
        # hiragana_dataset から学習時のラベルを自動復元・保存
        try:
            from dataset.hiragana_dataset import HiraganaDataset

            ds = HiraganaDataset(
                root_dir=DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir, sample_rate=16000
            )
            labels = list(ds.labels)
            logger.info("HiraganaDataset からラベルリスト (%d 件) を復元しました。", len(labels))
        except Exception:
            labels = list(DEFAULT_RECOGNITION_CONFIG.labels)

        # labels.json を保存
        with open(labels_file, "w", encoding="utf-8") as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)
        logger.info("labels.json を修復・保存しました: %s", labels_file)

    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    except Exception:
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            DEFAULT_RECOGNITION_CONFIG.wav2vec2_pretrained_model_name
        )
        try:
            feature_extractor.save_pretrained(model_path)
        except Exception:
            pass

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(labels),
        ignore_mismatched_sizes=True,
    )

    fp32_onnx_path = model_path / "model_fp32.onnx"
    export_to_onnx(model, fp32_onnx_path)

    int8_onnx_path: Path | None = None
    if export_int8:
        int8_onnx_path = model_path / "model_int8.onnx"
        quantize_onnx_int8(fp32_onnx_path, int8_onnx_path)

    pt_time, onnx_fp32_time, onnx_int8_time = run_benchmark(model, fp32_onnx_path, int8_onnx_path)

    times = [("ONNX FP32", onnx_fp32_time)]
    if onnx_int8_time is not None:
        times.append(("ONNX INT8", onnx_int8_time))
    fastest_name, fastest_time = min(times, key=lambda x: x[1])
    speedup = pt_time / max(fastest_time, 1e-6)

    print("\n=======================================================")
    print("  Wav2Vec2 ONNX ベンチマーク結果")
    print("=======================================================")
    print(f" モデルディレクトリ: {model_path}")
    print("-------------------------------------------------------")
    print(f" PyTorch FP32 CPU レイテンシ: {pt_time:.2f} ms")
    print(f" ONNX FP32 CPU レイテンシ   : {onnx_fp32_time:.2f} ms")
    if onnx_int8_time is not None:
        print(f" ONNX INT8 CPU レイテンシ   : {onnx_int8_time:.2f} ms")
    print("-------------------------------------------------------")
    print(f" 最速構成                   : {fastest_name} ({fastest_time:.2f} ms)")
    print(f" PyTorch 比高速化倍率       : {speedup:.2f}x")
    print("=======================================================\n")

    return int8_onnx_path if int8_onnx_path and int8_onnx_path.exists() else fp32_onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Wav2Vec2 model to ONNX format.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_RECOGNITION_CONFIG.wav2vec2_best_model_dir,
        help="Model directory containing fine-tuned Wav2Vec2 model",
    )
    parser.add_argument(
        "--no-int8",
        action="store_false",
        dest="export_int8",
        help="Skip dynamic INT8 quantization export",
    )
    args = parser.parse_args()
    export_and_benchmark(model_dir=args.model_dir, export_int8=args.export_int8)


if __name__ == "__main__":
    main()
