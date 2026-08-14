"""
Collected Dataset Directory Structure & Manifest Migration Script

Usage:
  uv run python script/migrate_collected_dataset.py
"""

import logging

from config import DEFAULT_RECOGNITION_CONFIG
from utils.machine_id import get_machine_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def migrate_collected_dataset() -> None:
    collected_dir = DEFAULT_RECOGNITION_CONFIG.collected_dataset_dir
    if not collected_dir.exists():
        logger.warning("対象ディレクトリが存在しません: %s", collected_dir)
        return

    target_dir = collected_dir / get_machine_id()
    target_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = target_dir / "metadata.csv"
    old_log_file = collected_dir / "predicted_text.txt"

    # 旧 predicted_text.txt の読み込み
    records: dict[str, str] = {}
    if old_log_file.exists():
        with open(old_log_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and "," in line:
                    parts = line.split(",")
                    timestamp = parts[0].strip()
                    predicted = parts[1].strip()
                    records[timestamp] = predicted

    # ログファイル (log) から認識成功データを補完
    app_log = DEFAULT_RECOGNITION_CONFIG.log_path
    if app_log.exists():
        import re

        with open(app_log, encoding="utf-8", errors="ignore") as f:
            log_lines = f.readlines()
        for i, line in enumerate(log_lines):
            m = re.search(r"([0-9]+_[0-9]+)(?:_\w+)?\.wav", line)
            if m:
                ts = m.group(1)
                for j in range(i, min(i + 5, len(log_lines))):
                    pm = re.search(r"認識成功:\s*(\w+)", log_lines[j])
                    if pm:
                        if ts not in records:
                            records[ts] = pm.group(1)
                        break

    # 既存 metadata.csv の読み込み・更新
    existing_entries: dict[str, list[str]] = {}
    if metadata_file.exists():
        with open(metadata_file, encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) >= 2 and parts[0]:
                    existing_entries[parts[0]] = parts

    wav_files = list(collected_dir.glob("*.wav")) + list(target_dir.glob("*.wav"))
    seen_stems = set()

    for wav_path in sorted(wav_files):
        stem = wav_path.stem
        if stem in seen_stems:
            continue
        seen_stems.add(stem)

        parts = stem.split("_")
        if len(parts) >= 3:
            timestamp = f"{parts[0]}_{parts[1]}"
            predicted = parts[2]
        elif len(parts) == 2:
            timestamp = stem
            predicted = records.get(timestamp, "")
        else:
            continue

        new_filename = f"{timestamp}.wav"
        new_wav_path = target_dir / new_filename

        if wav_path != new_wav_path and wav_path.exists():
            if new_wav_path.exists():
                wav_path.unlink()
            else:
                wav_path.rename(new_wav_path)
            logger.info("リネーム完了: %s -> %s", wav_path.name, new_filename)

        # metadata.csv エントリの作成・更新
        if timestamp in existing_entries:
            row = existing_entries[timestamp]
            # 既存の predicted が空で、records から取得できた場合は補完
            if len(row) > 2 and not row[2] and predicted:
                row[2] = predicted
            elif len(row) <= 2:
                row.append(predicted)
        else:
            existing_entries[timestamp] = [timestamp, new_filename, predicted, ""]

    # metadata.csv の保存
    with open(metadata_file, "w", encoding="utf-8") as meta_f:
        for timestamp, row in sorted(existing_entries.items()):
            time_val = row[0] if len(row) > 0 else timestamp
            file_val = row[1] if len(row) > 1 else f"{timestamp}.wav"
            pred_val = row[2] if len(row) > 2 else ""
            gt_val = row[3] if len(row) > 3 else ""
            meta_f.write(f"{time_val},{file_val},{pred_val},{gt_val}\n")

    # 旧ログファイルの削除
    if old_log_file.exists():
        old_log_file.unlink()
        logger.info("旧ログファイル %s を削除しました", old_log_file.name)

    logger.info("collected データセットの移行が完了しました (件数: %d)", len(existing_entries))


if __name__ == "__main__":
    migrate_collected_dataset()
