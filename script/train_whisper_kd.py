#!/usr/bin/env python3
"""
CLI entry point to train Strategy 02: Wav2Vec2 + Whisper Hidden Feature Distillation.

Usage:
  uv run python script/train_whisper_kd.py --epochs 12 --batch-size 8 --patience 2
"""

from voicerecognizer.strategies.whisper_kd.train import main

if __name__ == "__main__":
    main()
