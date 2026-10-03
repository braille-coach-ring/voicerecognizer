#!/usr/bin/env python3
"""
CLI entry point to train Strategy: Wav2Vec2 Consonant/Vowel Multi-Task + XLS-R IPA KD Hybrid.

Usage:
  uv run python script/train_phoneme_ipa_hybrid.py --epochs 12 --batch-size 8 --patience 2
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from voicerecognizer.strategies.phoneme_ipa_hybrid.train import main  # noqa: E402

if __name__ == "__main__":
    main()
