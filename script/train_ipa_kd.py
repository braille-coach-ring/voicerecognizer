#!/usr/bin/env python3
"""
CLI entry point to train Strategy 01: Wav2Vec2 + XLS-R IPA Knowledge Distillation.

Usage:
  uv run python script/train_ipa_kd.py --epochs 3 --batch-size 8
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from voicerecognizer.strategies.ipa_kd.train import main  # noqa: E402

if __name__ == "__main__":
    main()
