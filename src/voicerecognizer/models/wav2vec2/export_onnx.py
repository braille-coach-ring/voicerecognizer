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
import contextlib
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import override

if sys.platform == "win32":
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

from voicerecognizer.config import DEFAULT_RECOGNITION_CONFIG

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
        (dummy_input,),
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


class WaveformPrependedWav2Vec2(torch.nn.Module):
    """生音声波形 (1D tensor) への正規化前処理を内包した Wav2Vec2 統合ラッパーモジュール"""

    def __init__(self, base_model: torch.nn.Module) -> None:
        super().__init__()
        self.base_model = base_model

    @override
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (batch_size, num_samples) の raw audio float32 波形
        Returns:
            logits: (batch_size, num_classes) の推論ログ確率
        """
        mean = waveform.mean(dim=-1, keepdim=True)
        var = waveform.var(dim=-1, keepdim=True, unbiased=False)
        normalized_input = (waveform - mean) / torch.sqrt(var + 1e-7)
        outputs = self.base_model(normalized_input)
        return outputs.logits


def export_mel_prepended_onnx(
    model: torch.nn.Module,
    output_fp32_path: Path,
    sample_rate: int = DEFAULT_RECOGNITION_CONFIG.sample_rate,
    target_length_seconds: float = DEFAULT_RECOGNITION_CONFIG.target_length_seconds,
) -> None:
    """前処理内包型 Wav2Vec2 モデルを ONNX フォーマットへエクスポートします。"""
    model.eval()
    prepended_model = WaveformPrependedWav2Vec2(model)
    prepended_model.eval()

    num_samples = int(sample_rate * target_length_seconds)
    dummy_waveform = torch.randn(1, num_samples, dtype=torch.float32)

    logger.info("前処理内包型 ONNX モデルを出力中: %s", output_fp32_path)
    torch.onnx.export(
        prepended_model,
        (dummy_waveform,),
        str(output_fp32_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["waveform"],
        output_names=["logits"],
        dynamic_axes={
            "waveform": {0: "batch_size", 1: "num_samples"},
            "logits": {0: "batch_size"},
        },
        dynamo=False,
    )
    logger.info(
        "前処理内包型 FP32 ONNX エクスポート完了: %s (%.2f MB)",
        output_fp32_path,
        output_fp32_path.stat().st_size / (1024 * 1024),
    )


def export_and_benchmark(
    model_dir: Path | str = DEFAULT_RECOGNITION_CONFIG.wav2vec2_best_model_dir,
    export_int8: bool = True,
    skip_benchmark: bool = False,
) -> Path:
    from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Wav2Vec2 モデルディレクトリが存在しません: {model_path}")

    labels_file = model_path / "labels.json"
    labels: list[str] | None = None

    if labels_file.exists():
        try:
            with open(labels_file, encoding="utf-8") as f:
                labels = json.load(f)
        except Exception:
            pass

    if not labels:
        try:
            from voicerecognizer.dataset.hiragana_dataset import HiraganaDataset

            ds = HiraganaDataset(
                root_dir=DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir, sample_rate=16000
            )
            labels = list(ds.labels)
            logger.info("HiraganaDataset からラベルリスト (%d 件) を復元しました。", len(labels))
        except Exception:
            labels = list(DEFAULT_RECOGNITION_CONFIG.labels)

        with open(labels_file, "w", encoding="utf-8") as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)
        logger.info("labels.json を修復・保存しました: %s", labels_file)

    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    except Exception:
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            DEFAULT_RECOGNITION_CONFIG.wav2vec2_pretrained_model_name
        )
        with contextlib.suppress(Exception):
            feature_extractor.save_pretrained(model_path)

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(labels),
        ignore_mismatched_sizes=True,
    )

    # 1. 前処理内包型 ONNX (model_mel_*.onnx - 最速・最優先モデル) の生成
    mel_fp32_onnx_path = model_path / DEFAULT_RECOGNITION_CONFIG.wav2vec2_mel_fp32_onnx_filename
    mel_int8_onnx_path = model_path / DEFAULT_RECOGNITION_CONFIG.wav2vec2_mel_int8_onnx_filename
    export_mel_prepended_onnx(model, mel_fp32_onnx_path)
    if export_int8:
        quantize_onnx_int8(mel_fp32_onnx_path, mel_int8_onnx_path)

    # 2. 通常 ONNX (model_*.onnx - 汎用フォールバック) の生成
    fp32_onnx_path = model_path / "model_fp32.onnx"
    export_to_onnx(model, fp32_onnx_path)
    int8_onnx_path: Path | None = None
    if export_int8:
        int8_onnx_path = model_path / "model_int8.onnx"
        quantize_onnx_int8(fp32_onnx_path, int8_onnx_path)

    primary_onnx_path = (
        mel_int8_onnx_path if (export_int8 and mel_int8_onnx_path.exists()) else mel_fp32_onnx_path
    )

    if not skip_benchmark:
        pt_time, onnx_fp32_time, onnx_int8_time = run_benchmark(
            model, mel_fp32_onnx_path, mel_int8_onnx_path
        )

        times: list[tuple[str, float | None]] = [("ONNX FP32 (mel)", onnx_fp32_time)]
        if onnx_int8_time is not None:
            times.append(("ONNX INT8 (mel)", onnx_int8_time))

        def _get_item_time(item: tuple[str, float | None]) -> float:
            return item[1] if item[1] is not None else float("inf")

        fastest_name, fastest_time = min(times, key=_get_item_time)
        speedup = pt_time / max(fastest_time or 1e-6, 1e-6)

        print("\n=======================================================")
        print("  Wav2Vec2 ONNX ベンチマーク結果 (前処理内包型)")
        print("=======================================================")
        print(f" モデルディレクトリ: {model_path}")
        print("-------------------------------------------------------")
        print(f" PyTorch FP32 CPU レイテンシ: {pt_time:.2f} ms")
        print(f" ONNX FP32 CPU レイテンシ   : {onnx_fp32_time:.2f} ms")
        if onnx_int8_time is not None:
            print(f" ONNX INT8 CPU レイテンシ   : {onnx_int8_time:.2f} ms")
        print("-------------------------------------------------------")
        print(f" 最速構成                   : {fastest_name} ({(fastest_time or 0.0):.2f} ms)")
        print(f" PyTorch 比高速化倍率       : {speedup:.2f}x")
        print("=======================================================\n")

    return primary_onnx_path


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
