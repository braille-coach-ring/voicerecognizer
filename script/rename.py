"""
Raw Dataset File Sequential Renaming Utility

Usage:
  uv run python script/rename.py
"""

import logging

from voicerecognizer.config import DEFAULT_RECOGNITION_CONFIG

logger = logging.getLogger(__name__)

# datasetフォルダの場所
root = DEFAULT_RECOGNITION_CONFIG.raw_dataset_dir


def main() -> None:
    logger.info("ファイル名の正規化リネームを開始します: %s", root)
    if not root.exists():
        logger.warning("対象ディレクトリが存在しません: %s", root)
        return

    # 各文字フォルダを処理
    for folder in root.iterdir():
        if folder.is_dir():
            files = sorted(folder.glob("*.wav"))

            # 一時的な名前に変更（重複防止）
            temp_names = []
            for i, file in enumerate(files):
                temp = folder / f"temp_{i:03d}.wav"
                file.rename(temp)
                temp_names.append(temp)

            # 001.wav, 002.wav... に変更
            for i, temp in enumerate(temp_names, start=1):
                new_name = folder / f"{i:03d}.wav"
                temp.rename(new_name)

            logger.info(
                "フォルダ [%s]: %d 個のファイル名を正規化しました", folder.name, len(temp_names)
            )

    logger.info("リネーム完了")


if __name__ == "__main__":
    main()
