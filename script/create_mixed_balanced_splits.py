"""Generate balanced speaker & mixed-pool splits (80/10/10 for rinry & collected)."""

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def make_splits():
    source_dir = ROOT / "data_splits/speaker_independent_full_20261004"
    out_dir = ROOT / "data_splits/speaker_mixed_balanced_20261004"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for split in ["train", "val", "test"]:
        with open(source_dir / f"{split}.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                all_rows.append(dict(r, orig_split=split))

    rinry_rows = [r for r in all_rows if r["filepath"].startswith("dataset/rinry")]
    collected_speech = [
        r for r in all_rows if r["filepath"].startswith("dataset/collected") and r["label"] != "other"
    ]
    collected_env = [
        r for r in all_rows if r["filepath"].startswith("dataset/collected") and r["label"] == "other"
    ]
    fixed_rows = [
        r for r in all_rows
        if not r["filepath"].startswith("dataset/rinry") and not r["filepath"].startswith("dataset/collected")
    ]

    def strat_split(items, val_r=0.1, test_r=0.1, seed=42):
        rng = random.Random(seed)
        by_label = defaultdict(list)
        for it in items:
            by_label[it["label"]].append(it)
        train, val, test = [], [], []
        for _lbl, group in sorted(by_label.items()):
            rng.shuffle(group)
            n = len(group)
            n_val = max(1, round(n * val_r)) if n >= 10 else (1 if n >= 5 else 0)
            n_test = max(1, round(n * test_r)) if n >= 10 else (1 if n >= 5 else 0)
            if n_val + n_test >= n:
                n_val = 1 if n >= 2 else 0
                n_test = 1 if n >= 3 else 0
            val.extend(group[:n_val])
            test.extend(group[n_val : n_val + n_test])
            train.extend(group[n_val + n_test :])
        return train, val, test

    rinry_tr, rinry_va, rinry_te = strat_split(rinry_rows)
    coll_sp_tr, coll_sp_va, coll_sp_te = strat_split(collected_speech)
    coll_env_tr, coll_env_va, coll_env_te = strat_split(collected_env)

    fixed_by_split = defaultdict(list)
    for r in fixed_rows:
        fixed_by_split[r["orig_split"]].append(r)

    splits = {
        "train": fixed_by_split["train"] + rinry_tr + coll_sp_tr + coll_env_tr,
        "val": fixed_by_split["val"] + rinry_va + coll_sp_va + coll_env_va,
        "test": fixed_by_split["test"] + rinry_te + coll_sp_te + coll_env_te,
    }

    # Verify no overlap
    all_paths = [r["filepath"] for s in splits.values() for r in s]
    assert len(all_paths) == len(set(all_paths)), "Duplicate filepath across splits!"
    assert len(all_paths) == len(all_rows), f"Total count mismatch: {len(all_paths)} vs {len(all_rows)}"

    fieldnames = ["filepath", "label", "speaker"]
    manifest = {"counts": {}, "speaker_counts": {}}
    for name, rows in splits.items():
        rows.sort(key=lambda x: x["filepath"])
        out_csv = out_dir / f"{name}.csv"
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        manifest["counts"][name] = len(rows)
        manifest["speaker_counts"][name] = dict(Counter(r["speaker"] for r in rows).most_common())
        print(f"Wrote {out_csv} ({len(rows)} samples)")

    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Done generating splits.")

if __name__ == "__main__":
    make_splits()
