"""Prepare fixed manifests from reviewed r audio and CSV-labelled new recordings."""

import argparse
import csv
import hashlib
import json
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def prepare(
    root: Path, output: Path, accept_unreviewed: bool = False, include_tts: bool = False
) -> dict:
    if output.exists():
        raise FileExistsError(output)
    decisions_path = root / "evaluation_results/r_label_review_decisions.json"
    decisions_bytes = decisions_path.read_bytes()
    decisions = {
        row["filepath"].replace("\\", "/"): row for row in json.loads(decisions_bytes)["decisions"]
    }
    candidates = json.loads(
        (root / "evaluation_results/r_model_review_candidates.json").read_text(encoding="utf-8")
    )["candidates"]
    pending = {
        row["filepath"]
        for row in candidates
        if row["true_label"] != row["predicted_label"] and row["filepath"] not in decisions
    }
    inventory = json.loads(
        (root / "Docs/dataset_audit_2026-10-04/inventory.json").read_text(encoding="utf-8")
    )
    labels = set(inventory["expected_labels"])
    conflicts = {path for group in inventory["exact_duplicate_groups"] for path in group}
    rows, excluded = [], []
    base_manifests = {}
    for split in ("train", "val", "test"):
        source = root / f"data_splits/speaker_independent/{split}.csv"
        source_bytes = source.read_bytes()
        with source.open(encoding="utf-8-sig", newline="") as stream:
            records = list(csv.DictReader(stream))
        for row in records:
            if row["speaker"] in {f"r{i}" for i in range(1, 11)}:
                raise ValueError(f"Baseline already contains r speakers: {source}")
            if row["label"] not in labels:
                raise ValueError(f"Invalid baseline label: {row}")
            rows.append(
                {
                    "filepath": row["filepath"].replace("\\", "/"),
                    "label": row["label"],
                    "speaker": row["speaker"],
                    "split": split,
                }
            )
        base_manifests[split] = {
            "path": source.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "preserved_rows": len(records),
        }
    for speaker in (f"r{i}" for i in range(1, 11)):
        for wav in sorted((root / "dataset" / speaker).glob("*/*.wav")):
            path, label = wav.relative_to(root).as_posix(), wav.parent.name
            decision = decisions.get(path)
            reason = ""
            if path in conflicts:
                reason = "legacy_conflicting_labels"
            elif path in pending and not accept_unreviewed:
                reason = "missing_review_decision"
            elif decision and decision["decision"] in {"maybe", "delete_candidate"}:
                reason = decision["decision"]
            if reason:
                excluded.append({"filepath": path, "reason": reason})
                continue
            if decision:
                if decision["label"] != label:
                    raise ValueError(f"Review source label changed: {path}")
                if decision["decision"] == "other":
                    label = "other"
                elif decision["decision"] == "relabel":
                    label = decision["new_label"].strip()
            if label not in labels:
                raise ValueError(f"Invalid label: {path}: {label}")
            split = (
                "val" if speaker in {"r2", "r4"} else "test" if speaker in {"r3", "r5"} else "train"
            )
            rows.append({"filepath": path, "label": label, "speaker": speaker, "split": split})

    added = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--diff-filter=A",
            "--name-only",
            "60e25076",
            "HEAD",
            "--",
            "dataset",
        ],
        text=True,
    ).splitlines()
    added = {path for path in added if path.endswith(".wav")}
    new_rows = {}
    for metadata in (root / "dataset/collected").glob("*/metadata.csv"):
        with metadata.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.reader(stream):
                if len(row) < 3:
                    continue
                path = (metadata.parent / row[1].strip()).relative_to(root).as_posix()
                if path not in added:
                    continue
                label = row[3].strip() if len(row) >= 4 else row[2].strip()
                if path in new_rows or label not in labels or not (root / path).is_file():
                    raise ValueError(f"Invalid new CSV recording: {path}")
                new_rows[path] = {
                    "filepath": path,
                    "label": label,
                    "speaker": metadata.parent.name,
                    "split": "train",
                }
    if set(new_rows) != added:
        raise ValueError("New recordings and CSV rows do not match")
    ambient = sorted(path for path, row in new_rows.items() if row["label"] == "other")
    random.Random(42).shuffle(ambient)
    for index, path in enumerate(ambient):
        # Environmental recordings have recording IDs, not human speaker IDs.
        new_rows[path]["speaker"] = "environment:" + Path(path).stem
        new_rows[path]["split"] = "val" if index < 10 else "test" if index < 20 else "train"
    rows.extend(new_rows.values())
    tts_rows = []
    if include_tts:
        for wav in sorted((root / "dataset/ai_female_nanami").glob("*/*.wav")):
            if wav.parent.name not in labels:
                raise ValueError(f"Invalid TTS label: {wav}")
            tts_rows.append(
                {
                    "filepath": wav.relative_to(root).as_posix(),
                    "label": wav.parent.name,
                    "speaker": "tts_nanami",
                    "split": "train",
                }
            )
        rows.extend(tts_rows)

    seen_paths, seen_hashes = set(), set()
    splits = defaultdict(list)
    for row in rows:
        path = root / row["filepath"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if row["filepath"] in seen_paths or digest in seen_hashes:
            raise ValueError(f"Duplicate training recording: {row['filepath']}")
        seen_paths.add(row["filepath"])
        seen_hashes.add(digest)
        splits[row["split"]].append(row)
    speaker_sets = [{row["speaker"] for row in splits[name]} for name in ("train", "val", "test")]
    if any(speaker_sets[i] & speaker_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Speaker overlap")
    if not all(splits[name] for name in ("train", "val", "test")):
        raise ValueError("Empty split")
    output.mkdir(parents=True)
    fields = ["filepath", "label", "speaker"]
    for name in ("train", "val", "test"):
        with (output / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sorted(splits[name], key=lambda row: row["filepath"]))
    (output / "review_decisions_snapshot.json").write_bytes(decisions_bytes)
    summary = {
        "source_head": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "decision_counts": dict(Counter(row["decision"] for row in decisions.values())),
        "review_sha256": hashlib.sha256(decisions_bytes).hexdigest(),
        "accept_unreviewed": accept_unreviewed,
        "base_manifests": base_manifests,
        "tts_train_rows": len(tts_rows),
        "new_csv_labels": dict(Counter(row["label"] for row in new_rows.values())),
        "splits": {
            name: {
                "files": len(records),
                "speakers": dict(Counter(row["speaker"] for row in records)),
                "labels": dict(Counter(row["label"] for row in records)),
            }
            for name, records in splits.items()
        },
        "excluded": excluded,
        "environment_holdout": "10 validation and 10 test recordings from the same collected session; not an unseen acoustic environment benchmark",
        "legacy_policy": "All existing speaker_independent CSV rows retain their split, label and speaker; r speakers and new CSV-labelled recordings are appended",
    }
    (output / "preparation_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--accept-unreviewed", action="store_true")
    parser.add_argument("--include-tts", action="store_true")
    args = parser.parse_args()
    summary = prepare(ROOT, args.output_dir, args.accept_unreviewed, args.include_tts)
    print(
        json.dumps(
            {
                name: {"files": split["files"], "other": split["labels"].get("other", 0)}
                for name, split in summary["splits"].items()
            },
            indent=2,
        )
    )
    print("Excluded:", dict(Counter(row["reason"] for row in summary["excluded"])))
