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
