"""
Apply relabel decisions from review_decisions.json by moving wav files to their target label directory.

Usage:
  # Check planned moves without modifying files (Dry Run):
  python script/apply_relabel_decisions.py

  # Execute the actual file moves:
  python script/apply_relabel_decisions.py --execute
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voicerecognizer.evaluation.review import load_review_decisions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_next_available_path(target_dir: Path, original_stem: str, reserved: set[Path]) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    candidate = target_dir / f"{original_stem}.wav"
    if not candidate.exists() and candidate not in reserved:
        reserved.add(candidate)
        return candidate

    # Find highest numeric stem in target_dir and reserved
    max_num = 0
    for f in target_dir.glob("*.wav"):
        if f.stem.isdigit():
            max_num = max(max_num, int(f.stem))
    for f in reserved:
        if f.parent == target_dir and f.stem.isdigit():
            max_num = max(max_num, int(f.stem))

    next_num = max_num + 1
    width = len(original_stem) if original_stem.isdigit() else 3
    new_candidate = target_dir / f"{next_num:0{width}d}.wav"
    while new_candidate.exists() or new_candidate in reserved:
        next_num += 1
        new_candidate = target_dir / f"{next_num:0{width}d}.wav"

    reserved.add(new_candidate)
    return new_candidate


def apply_relabels(
    decisions_path: Path,
    project_root: Path = PROJECT_ROOT,
    execute: bool = False,
) -> None:
    if not decisions_path.exists():
        logger.error("Decisions file not found: %s", decisions_path)
        return

    decisions = load_review_decisions(decisions_path)
    relabel_items = []
    for d in decisions.values():
        if d.decision == "other":
            relabel_items.append((d, "other"))
        elif d.decision == "relabel" and d.new_label.strip():
            relabel_items.append((d, d.new_label.strip()))

    if not relabel_items:
        logger.info("No 'relabel' or 'other' decisions found in %s", decisions_path)
        return

    logger.info("Found %d relocation decisions (relabel/other)", len(relabel_items))
    mode_str = "[EXECUTE]" if execute else "[DRY-RUN]"

    reserved_paths: set[Path] = set()
    moved_count = 0
    for d, target_label in relabel_items:
        src_path = project_root / d.filepath
        if not src_path.exists():
            logger.warning("Source file not found: %s", src_path)
            continue

        old_label = d.label
        new_label = target_label.strip()
        if old_label == new_label:
            logger.info("Skipping same label: %s (%s == %s)", d.filepath, old_label, new_label)
            continue

        # Parent is label dir, grandparent is speaker dir (e.g. dataset/take/a -> dataset/take/i)
        speaker_dir = src_path.parent.parent
        target_dir = speaker_dir / new_label
        dest_path = find_next_available_path(target_dir, src_path.stem, reserved_paths)

        logger.info(
            "%s %s (%s -> %s) ==> %s",
            mode_str,
            src_path.relative_to(project_root),
            old_label,
            new_label,
            dest_path.relative_to(project_root),
        )

        if execute:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src_path, dest_path)
            moved_count += 1

    if execute:
        logger.info("Successfully moved %d files.", moved_count)
    else:
        logger.info("Dry run completed. To perform actual moves, run with --execute")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply relabel decisions to audio files")
    parser.add_argument(
        "--decisions",
        type=Path,
        default=PROJECT_ROOT / "evaluation_results" / "review_decisions.json",
        help="Path to review_decisions.json",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move files. If omitted, runs in dry-run mode.",
    )
    args = parser.parse_args()
    apply_relabels(args.decisions.resolve(), execute=args.execute)


if __name__ == "__main__":
    main()
