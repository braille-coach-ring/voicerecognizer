"""
Mel / Waveform Prepended Wav2Vec2 ONNX Export & Quantization Script

役割:
  生音声波形 (1D float32 waveform tensor) を直接入力として受ける
  前処理内包型 Wav2Vec2 ONNX モデル (model_mel_fp32.onnx / model_mel_int8.onnx) を生成します。
  これにより Python / C++ 側の特徴量抽出前処理オーバーヘッドを完全に排除し、
  ラズパイ(ARM64)や組込みデバイス上で単一 ONNX ファイルとして超高速に動作させることができます。

使い方:
  uv run python models/wav2vec2/export_mel_prepended_onnx.py
"""

import io
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, Wav2Vec2ForSequenceClassification

from voicerecognizer.config import DEFAULT_RECOGNITION_CONFIG

if sys.platform == "win32":
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class WaveformPrependedWav2Vec2(nn.Module):
    """
    生音声波形 (1D tensor) への正規化前処理を内包した Wav2Vec2 統合ラッパーモジュール
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (batch_size, num_samples) の raw audio float32 波形
        Returns:
            logits: (batch_size, num_classes) の推論ログ確率
        """
        # 波形の平均・標準偏差による Z-score 正規化 (Wav2Vec2 FeatureExtractor 互換)
        mean = waveform.mean(dim=-1, keepdim=True)
        var = waveform.var(dim=-1, keepdim=True, unbiased=False)
        normalized_input = (waveform - mean) / torch.sqrt(var + 1e-7)

        outputs = self.base_model(normalized_input)
        return outputs.logits


def export_mel_prepended_onnx(
    model_dir: Path,
    output_fp32_path: Path,
    sample_rate: int = DEFAULT_RECOGNITION_CONFIG.sample_rate,
    target_length_seconds: float = DEFAULT_RECOGNITION_CONFIG.target_length_seconds,
) -> None:
    """前処理内包型 Wav2Vec2 モデルを ONNX フォーマットへエクスポートします。"""
    config_path = model_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Model config.json not found: {config_path}")

    config = AutoConfig.from_pretrained(model_dir)
    base_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir, config=config)
    base_model.eval()

    prepended_model = WaveformPrependedWav2Vec2(base_model)
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


def quantize_onnx_int8(fp32_onnx_path: Path, int8_onnx_path: Path) -> None:
    """ONNX モデルを INT8 動的量子化します。"""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise ImportError("ONNX quantization requires 'onnxruntime'.") from exc

    logger.info("前処理内包型 ONNX INT8 動的量子化を適用中...")
    quantize_dynamic(
        model_input=str(fp32_onnx_path),
        model_output=str(int8_onnx_path),
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=["MatMul", "Gather", "Attention"],
    )
    logger.info(
        "前処理内包型 INT8 ONNX モデル生成完了: %s (%.2f MB)",
        int8_onnx_path,
        int8_onnx_path.stat().st_size / (1024 * 1024),
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Export Wav2Vec2 to Waveform-prepended ONNX format"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Path to wav2vec2 model directory containing model.safetensors and config.json",
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    if model_dir is None:
        local_weights_dir = Path("weights/wav2vec2_best")
        if (local_weights_dir / "model.safetensors").exists() or (
            local_weights_dir / "config.json"
        ).exists():
            model_dir = local_weights_dir
        else:
            model_dir = DEFAULT_RECOGNITION_CONFIG.wav2vec2_best_model_dir

    fp32_path = model_dir / DEFAULT_RECOGNITION_CONFIG.wav2vec2_mel_fp32_onnx_filename
    int8_path = model_dir / DEFAULT_RECOGNITION_CONFIG.wav2vec2_mel_int8_onnx_filename

    export_mel_prepended_onnx(model_dir, fp32_path)
    quantize_onnx_int8(fp32_path, int8_path)


if __name__ == "__main__":
    main()
