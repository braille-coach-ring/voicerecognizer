"""
Silence Audio File Cleaner Script

Usage:
  uv run python script/delete_silence_file.py
"""

import logging

import librosa
import numpy as np

from voicerecognizer.config import (
    DEFAULT_AUDIO_CONFIG,
    DEFAULT_PREPROCESS_CONFIG,
    DEFAULT_RECOGNITION_CONFIG,
)

logger = logging.getLogger(__name__)

# ==========================
# 設定（config.py から参照）
# ==========================

ROOT = DEFAULT_RECOGNITION_CONFIG.raw_dataset_dir

# 最大音量がこれ未満なら無音と判定
THRESHOLD = DEFAULT_PREPROCESS_CONFIG.vad_silence_threshold


def main() -> None:
    deleted = 0
    logger.info("無音ファイルの検索を開始します... (対象: %s, 閾値: %.4f)", ROOT, THRESHOLD)

    if not ROOT.exists():
        logger.warning("対象ディレクトリが存在しません: %s", ROOT)
        return

    for folder in ROOT.iterdir():
        if not folder.is_dir():
            continue

        for wav in folder.glob("*.wav"):
            try:
                y, _ = librosa.load(wav, sr=DEFAULT_AUDIO_CONFIG.sample_rate)
                volume = float(np.max(np.abs(y))) if y.size else 0.0

                if volume < THRESHOLD:
                    logger.info(
                        "無音ファイルを削除: %s (音量: %.5f < 閾値: %.4f)", wav, volume, THRESHOLD
                    )
                    wav.unlink()
                    deleted += 1
            except Exception as e:
                logger.error("ファイル読み込みエラー: %s (%s)", wav, e, exc_info=True)

    logger.info("無音ファイル探索完了 - 削除数: %d 件", deleted)


if __name__ == "__main__":
    main()
