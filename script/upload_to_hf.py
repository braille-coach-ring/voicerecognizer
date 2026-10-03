import argparse
import logging
import sys

from voicerecognizer.config import PROJECT_ROOT, load_env
from voicerecognizer.strategies.registry import DEFINED_STRATEGIES, StrategyStatus
from voicerecognizer.utils.model_uploader import (
    upload_strategy_weights_to_hf,
    upload_weights_to_hf,
)

load_env()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload model weights to Hugging Face Hub smartly (skip if identical)"
    )
    parser.add_argument(
        "--type",
        choices=["cnn", "wav2vec2", "strategy"],
        default=None,
        help="Specify which base model weights to upload ('cnn', 'wav2vec2', or 'strategy')",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Specify strategy name (e.g., 'wav2vec2_phoneme_multi', 'all-active', 'all').",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force upload even if local file matches remote SHA-256 hash",
    )
    return parser


def upload_strategies(strategy_arg: str, force: bool = False) -> bool:
    strategies_dir = PROJECT_ROOT / "weights" / "strategies"
    targets: list[str] = []

    if strategy_arg == "all-active":
        for name, meta in DEFINED_STRATEGIES.items():
            if meta.status == StrategyStatus.ACTIVE and (strategies_dir / name).exists():
                targets.append(name)
    elif strategy_arg == "all":
        if strategies_dir.exists():
            for item in sorted(strategies_dir.iterdir()):
                if item.is_dir() and (item / "model.safetensors").exists():
                    targets.append(item.name)
    else:
        targets.append(strategy_arg)

    if not targets:
        logger.warning("アップロード対象の戦略が見つかりませんでした: %s", strategy_arg)
        return False

    all_success = True
    for s_name in targets:
        logger.info("==========================================")
        logger.info("戦略 '%s' の Hugging Face アップロード開始", s_name)
        logger.info("==========================================")
        ok = upload_strategy_weights_to_hf(strategy_name=s_name, force_upload=force)
        if not ok:
            all_success = False

    return all_success


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.strategy:
        success = upload_strategies(strategy_arg=args.strategy, force=args.force)
    elif args.type == "strategy":
        logger.error(
            "--type strategy が指定されましたが、--strategy <名前またはall-active> が指定されていません。"
        )
        sys.exit(1)
    else:
        model_type = args.type or "cnn"
        success = upload_weights_to_hf(model_type=model_type, force_upload=args.force)

    if not success:
        sys.exit(1)
