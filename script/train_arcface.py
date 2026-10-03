#!/usr/bin/env python3
"""
CLI entry point to train Strategy 04: Wav2Vec2 + ArcFace Angular Margin Loss.

Usage:
  uv run python script/train_arcface.py --epochs 12 --batch-size 8 --patience 2
"""

from voicerecognizer.strategies.arcface.train import main

if __name__ == "__main__":
    main()
