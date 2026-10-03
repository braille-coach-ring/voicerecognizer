"""
Generate Speaker-Independent Dataset Splits (Train / Val / Test)
according to speech recognition best practices.

Rules:
- Speakers(Train) ∩ Speakers(Val) ∩ Speakers(Test) = ∅ (Strictly Zero Data Leakage)
- Test: take (female, 104 labels) + yu-ota (male, 5 labels) + reon (male, 5 labels)
- Val:  yumike (male, 103 labels) + mikeryu (male, 5 labels)
- Train: rinry (female, 105 labels) + rikutomike (male, 104 labels) + haruyamike (male, 104 labels)
         + collected (PC mics, 5 labels) + rikuto (male, 5 labels) + ryu (male, 5 labels)
"""

import csv
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_SPEAKERS = {"take", "yu-ota", "reon"}
VAL_SPEAKERS = {"yumike", "mikeryu"}
TRAIN_SPEAKERS = {"rinry", "rikutomike", "haruyamike", "collected", "rikuto", "ryu"}


def extract_speaker(filepath: str) -> str:
    parts = Path(filepath).parts
    if len(parts) >= 2 and parts[0] == "dataset":
        return parts[1]
    return "unknown"


def main():
    merged_index_path = PROJECT_ROOT / "merged_dataset" / "index.csv"
    if not merged_index_path.exists():
        logger.error("merged_dataset/index.csv not found at %s", merged_index_path)
        return

    splits_dir = PROJECT_ROOT / "data_splits" / "speaker_independent"
    splits_dir.mkdir(parents=True, exist_ok=True)

    with open(merged_index_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_samples = list(reader)

    logger.info("Loaded %d total samples from %s", len(all_samples), merged_index_path)

    train_samples = []
    val_samples = []
    test_samples = []
    unknown_samples = []

    for sample in all_samples:
        filepath = sample["filepath"]
        speaker = extract_speaker(filepath)
        sample["speaker"] = speaker

        if speaker in TEST_SPEAKERS:
            test_samples.append(sample)
        elif speaker in VAL_SPEAKERS:
            val_samples.append(sample)
        elif speaker in TRAIN_SPEAKERS:
            train_samples.append(sample)
        else:
            unknown_samples.append(sample)

    logger.info("Split counts:")
    logger.info("  Train: %d samples (Speakers: %s)", len(train_samples), sorted(TRAIN_SPEAKERS))
    logger.info("  Val:   %d samples (Speakers: %s)", len(val_samples), sorted(VAL_SPEAKERS))
    logger.info("  Test:  %d samples (Speakers: %s)", len(test_samples), sorted(TEST_SPEAKERS))
    if unknown_samples:
        logger.warning("  Unknown speaker samples: %d", len(unknown_samples))

    # Verify zero overlap between speaker sets
    train_sp = {s["speaker"] for s in train_samples}
    val_sp = {s["speaker"] for s in val_samples}
    test_sp = {s["speaker"] for s in test_samples}

    assert not (train_sp & val_sp), f"Train and Val share speakers: {train_sp & val_sp}"
    assert not (train_sp & test_sp), f"Train and Test share speakers: {train_sp & test_sp}"
    assert not (val_sp & test_sp), f"Val and Test share speakers: {val_sp & test_sp}"
    logger.info("Validation passed: ZERO speaker leakage across Train, Val, and Test!")

    # Verify label coverage
    train_labels = Counter(s["label"] for s in train_samples)
    val_labels = Counter(s["label"] for s in val_samples)
    test_labels = Counter(s["label"] for s in test_samples)
    logger.info("Label counts: Train=%d labels, Val=%d labels, Test=%d labels",
                len(train_labels), len(val_labels), len(test_labels))

    def write_csv(output_path: Path, samples: list[dict[str, str]]):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            fieldnames = ["filepath", "label", "speaker", "predicted_text"]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for s in samples:
                writer.writerow(s)
        logger.info("Wrote %s (%d rows)", output_path, len(samples))

    write_csv(splits_dir / "train.csv", train_samples)
    write_csv(splits_dir / "val.csv", val_samples)
    write_csv(splits_dir / "test.csv", test_samples)

    # For evaluator CLI (which expects a directory with index.csv)
    test_eval_dir = splits_dir / "test_eval"
    write_csv(test_eval_dir / "index.csv", test_samples)

    # Also write a female-only test set (take) for specific female zero-shot benchmark
    take_samples = [s for s in test_samples if s["speaker"] == "take"]
    female_eval_dir = splits_dir / "test_female_take"
    write_csv(female_eval_dir / "index.csv", take_samples)

    logger.info("All speaker-independent splits generated successfully!")


if __name__ == "__main__":
    main()
