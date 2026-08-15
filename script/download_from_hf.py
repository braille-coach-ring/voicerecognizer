"""Hugging Face Hub からモデル重み・設定ファイルをダウンロード・同期するスクリプト

使い方:
  uv run python script/download_from_hf.py --type all
  uv run python script/download_from_hf.py --type wav2vec2
  uv run python script/download_from_hf.py --type cnn
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Literal

from voicerecognizer.config import (
    DEFAULT_RECOGNITION_CONFIG,
    HuggingFaceConfig,
    load_env,
)
from voicerecognizer.utils.model_uploader import download_latest_team_weights_if_needed

load_env()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def download_models(
    model_type: Literal["all", "cnn", "wav2vec2"] = "all",
    target_dir: Path | None = None,
    repo_id: str | None = None,
    token: str | None = None,
) -> bool:
    """指定されたモデルを Hugging Face Hub よりダウンロードします。"""
    if repo_id is not None and token is not None:
        hf_config = HuggingFaceConfig(repo_id=repo_id, token=token)
    elif repo_id is not None:
        hf_config = HuggingFaceConfig(repo_id=repo_id)
    elif token is not None:
        hf_config = HuggingFaceConfig(token=token)
    else:
        hf_config = HuggingFaceConfig()

    dest_dir = target_dir or DEFAULT_RECOGNITION_CONFIG.weights_dir

    success = True
    if model_type in ("all", "wav2vec2"):
        logger.info("=== Wav2Vec2 モデル重みのダウンロード・同期を開始します ===")
        ok = download_latest_team_weights_if_needed(
            model_type="wav2vec2",
            hf_config=hf_config,
            weights_dir=dest_dir,
        )
        success = success and ok

    if model_type in ("all", "cnn"):
        logger.info("=== CNN モデル重みのダウンロード・同期を開始します ===")
        ok = download_latest_team_weights_if_needed(
            model_type="cnn",
            hf_config=hf_config,
            weights_dir=dest_dir,
        )
        success = success and ok

    return success


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and sync model weights from Hugging Face Hub"
    )
    parser.add_argument(
        "--type",
        choices=["all", "cnn", "wav2vec2"],
        default="all",
        help="Specify which model weights to download (default: all)",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=None,
        help="Target directory to save model weights (default: ~/.cache/voicerecognizer/weights)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Hugging Face model repository ID",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    ok = download_models(
        model_type=args.type,
        target_dir=args.target_dir,
        repo_id=args.repo_id,
    )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
