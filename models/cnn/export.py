import argparse
from pathlib import Path

import torch

from models.cnn.hiragana_cnn import HiraganaCNN


def export_torchscript(
    model_path: Path,
    output_path: Path,
    num_classes: int,
    n_mels: int,
    time_steps: int,
) -> None:
    model = HiraganaCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    example = torch.randn(1, 1, n_mels, time_steps)
    traced = torch.jit.trace(model, example)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the CNN model to TorchScript.")
    parser.add_argument("--model-path", type=Path, default=Path("weights/best_model.pth"))
    parser.add_argument("--output-path", type=Path, default=Path("weights/hiragana_cnn.pt"))
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--n-mels", type=int, default=64)
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
