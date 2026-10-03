#!/usr/bin/env python3
"""
CLI entry point to train Strategy: Wav2Vec2 Consonant/Vowel Multi-Task + XLS-R IPA KD Hybrid.

Usage:
  uv run python script/train_phoneme_ipa_hybrid.py --epochs 12 --batch-size 8 --patience 2
"""

from voicerecognizer.strategies.phoneme_ipa_hybrid.train import main

if __name__ == "__main__":
    main()
