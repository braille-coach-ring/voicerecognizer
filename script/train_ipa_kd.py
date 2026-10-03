#!/usr/bin/env python3
"""
CLI entry point to train Strategy 01: Wav2Vec2 + XLS-R IPA Knowledge Distillation.

Usage:
  uv run python script/train_ipa_kd.py --epochs 3 --batch-size 8
"""

from voicerecognizer.strategies.ipa_kd.train import main

if __name__ == "__main__":
    main()
