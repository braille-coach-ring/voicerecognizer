#!/usr/bin/env python3
"""
CLI entry point to train Strategy 07: Wav2Vec2 + RIR Room Acoustics & Microphone Distortion Simulation.

Usage:
  uv run python script/train_rir_simulation.py --epochs 14 --batch-size 8 --patience 3
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from voicerecognizer.strategies.rir_simulation.train import main  # noqa: E402

if __name__ == "__main__":
    main()
