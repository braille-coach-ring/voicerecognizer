#!/usr/bin/env python3
"""
CLI entry point to train Strategy 06: Wav2Vec2 Consonant/Vowel Multi-Task Learning.

Usage:
  uv run python script/train_phoneme_multi.py --epochs 12 --batch-size 8 --patience 2
"""

from voicerecognizer.strategies.phoneme_multi.train import main

if __name__ == "__main__":
    main()
