import argparse
import json
import logging
from pathlib import Path

import torch

from config import (
    DEFAULT_PREPROCESS_CONFIG,
    DEFAULT_RECOGNITION_CONFIG,
)
from models.cnn.hiragana_cnn import HiraganaCNN

logger = logging.getLogger(__name__)


def export_torchscript(
    model_path: Path,
    output_path: Path,
    num_classes: int | None = None,
    n_mels: int = DEFAULT_PREPROCESS_CONFIG.n_mels,
    time_steps: int = 101,
) -> None:
    logger.info("TorchScript モデルへの書き出しを開始します: %s", model_path)

    labels_json_path = model_path.parent / "labels.json"
    if num_classes is None:
        if labels_json_path.exists():
            try:
                labels = json.loads(labels_json_path.read_text(encoding="utf-8"))
                num_classes = len(labels)
                logger.info("labels.json からクラス数を自動検出しました: %d クラス", num_classes)
            except Exception as e:
                logger.warning("labels.json の読み込みに失敗しました: %s", e)
        if num_classes is None:
            num_classes = len(DEFAULT_RECOGNITION_CONFIG.labels)

    model = HiraganaCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    example = torch.randn(1, 1, n_mels, time_steps)
    traced = torch.jit.trace(model, example)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(output_path)
    logger.info("TorchScript モデルを保存しました: %s", output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the CNN model to TorchScript.")
    parser.add_argument(
        "--model-path", type=Path, default=DEFAULT_RECOGNITION_CONFIG.cnn_weight_path
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_RECOGNITION_CONFIG.torchscript_model_path,
    )
    parser.add_argument("--num-classes", type=int, default=len(DEFAULT_RECOGNITION_CONFIG.labels))
    parser.add_argument("--n-mels", type=int, default=DEFAULT_PREPROCESS_CONFIG.n_mels)
    parser.add_argument("--time-steps", type=int, default=101)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    export_torchscript(
        model_path=args.model_path,
        output_path=args.output_path,
        num_classes=args.num_classes,
        n_mels=args.n_mels,
        time_steps=args.time_steps,
    )


if __name__ == "__main__":
    main()
