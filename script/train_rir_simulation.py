#!/usr/bin/env python3
"""
CLI entry point to train Strategy 07: Wav2Vec2 + RIR Room Acoustics & Microphone Distortion Simulation.

Usage:
  uv run python script/train_rir_simulation.py --epochs 14 --batch-size 8 --patience 3
"""

from voicerecognizer.strategies.rir_simulation.train import main

if __name__ == "__main__":
    main()
