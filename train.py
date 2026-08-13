"""
Model Training Dispatcher Script

役割:
  学習前に自動でデータ統合 (merge_data: index.csv) および音声前処理 (preprocess: processed_dataset/)
  を実行した上で、指定モデル (cnn / wav2vec2) の学習を行います。
  既存重みはデフォルトで自動再利用 (reuse) されます。

使い方:
  uv run python train.py                           # CNN モデルを継続学習 (自動前処理ON)
  uv run python train.py --model wav2vec2          # Wav2Vec2 モデルを継続ファインチューニング
  uv run python train.py --from-scratch            # 既存重みを破棄して 0 から新規学習
  uv run python train.py --skip-prep               # 前処理・データ統合をスキップして学習
"""

import argparse
from collections.abc import Sequence

TRAINABLE_MODELS = ("cnn", "wav2vec2")


def build_dispatch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a voice recognition model.",
        add_help=False,
    )
    parser.add_argument(
        "--model",
        choices=TRAINABLE_MODELS,
        default="cnn",
        help="Model type to train.",
    )
    parser.add_argument("-h", "--help", action="store_true")
    return parser


def _load_model_trainer(model_type: str):
    if model_type == "cnn":
        from models.cnn.train import build_parser, train

        return build_parser, train

    if model_type == "wav2vec2":
        from models.wav2vec2.train import build_parser, train

        return build_parser, train

    available = ", ".join(TRAINABLE_MODELS)
    raise ValueError(f"Unknown trainable model: {model_type}. Available: {available}")


def main(argv: Sequence[str] | None = None) -> None:
    dispatch_args, remaining = build_dispatch_parser().parse_known_args(argv)
    build_parser, train = _load_model_trainer(dispatch_args.model)
    model_parser = build_parser()
    model_parser.prog = f"train.py --model {dispatch_args.model}"

    if dispatch_args.help:
        model_parser.print_help()
        return

    train(model_parser.parse_args(remaining))


if __name__ == "__main__":
    main()
