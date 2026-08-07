import sys
import argparse
import logging
from pathlib import Path

# 親ディレクトリを sys.path に追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.model_uploader import upload_weights_to_hf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload model weights to Hugging Face Hub smartly (skip if identical)")
    parser.add_argument(
        "--type",
        choices=["cnn", "wav2vec2"],
        default="cnn",
        help="Specify which model weights to upload (default: cnn)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force upload even if local file matches remote SHA-256 hash",
    )
    return parser

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    success = upload_weights_to_hf(model_type=args.type, force_upload=args.force)
    if not success:
        sys.exit(1)
