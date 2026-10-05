"""Run three controlled experiments using the same fixed data and trainer."""

import argparse
import csv
import json
import os
import platform
from collections import Counter
from pathlib import Path
from typing import Any

from script.evaluate_speaker_independent import evaluate, hash_file, read_manifest
from voicerecognizer.config import DEFAULT_SPEAKER_SPLIT_DIR, PROJECT_ROOT
from voicerecognizer.config_labels import ALL_HIRAGANA_LABELS


def audit_splits(split_dir: Path) -> dict[str, Any]:
    rows = {name: read_manifest(split_dir / f"{name}.csv") for name in ("train", "val", "test")}
    # Fixed speakers that MUST be 100% disjoint
    STRICT_SPEAKERS = {
        f"r{i}" for i in range(1, 11)
    } | {"take", "reon", "yu-ota", "yumike", "mikeryu", "rikutomike", "haruyamike", "rikuto", "ryu"}
    seen_paths, seen_strict_speakers, seen_hashes = set(), set(), set()
    summaries = {}
    for name, records in rows.items():
        paths = {row["filepath"] for row in records}
        speakers = {row["speaker"] for row in records}
        strict = speakers & STRICT_SPEAKERS
        digests = [hash_file(PROJECT_ROOT / row["filepath"]) for row in records]
        if paths & seen_paths or strict & seen_strict_speakers or set(digests) & seen_hashes:
            raise ValueError(f"File/speaker/content overlaps earlier split: {name}")
        if len(set(digests)) != len(digests):
            raise ValueError(f"Duplicate audio content within {name}")
        seen_paths |= paths
        seen_strict_speakers |= strict
        seen_hashes |= set(digests)
        summaries[name] = {
            "samples": len(records),
            "speakers": dict(Counter(row["speaker"] for row in records)),
            "missing_labels": sorted(set(ALL_HIRAGANA_LABELS) - {row["label"] for row in records}),
            "csv_sha256": hash_file(split_dir / f"{name}.csv"),
            "audio": dict(zip([row["filepath"] for row in records], digests, strict=True)),
        }
    return summaries


def prepare_dataset(split_dir: Path, run_root: Path) -> Path:
    from voicerecognizer.preprocessing.dataset_builder import DatasetBuilder

    source_dir = run_root / "source"
    source_dir.mkdir()
    rows = read_manifest(split_dir / "train.csv") + read_manifest(split_dir / "val.csv")
    with (source_dir / "index.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("filepath", "label", "speaker"), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    destination = run_root / "processed"
    DatasetBuilder().preprocess_dataset(source_dir, destination)
    return destination


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from voicerecognizer.models.wav2vec2.train import build_parser, train

    run_root = args.output_dir.resolve()
    run_root.relative_to(PROJECT_ROOT.resolve())
    if run_root.exists():
        raise ValueError("Use a new output directory; existing experiments are never overwritten")
    manifest: dict[str, Any] = {
        "splits": audit_splits(args.split_dir),
        "checkpoint_provenance": {
            "status": "unverified historical training data; warm-start results are exploratory",
            "initial_model_sha256": hash_file(args.initial_model / "model.safetensors"),
        },
        "selection_metric": "normalized validation macro_f1 over true-supported labels",
        "test_role": "fixed retrospective benchmark; previously inspected, not a new blind final test",
        "settings": vars(args) | {"python": platform.python_version(), "torch": torch.__version__},
        "runs": {},
    }
    labels = json.loads((args.initial_model / "labels.json").read_text(encoding="utf-8"))
    if labels != sorted(ALL_HIRAGANA_LABELS):
        raise ValueError("Initial checkpoint label order differs from training dataset")
    bundle_manifest = PROJECT_ROOT / "experiments/bundle_manifest.json"
    if bundle_manifest.exists():
        captured = json.loads(bundle_manifest.read_text(encoding="utf-8"))
        if captured["splits"] != manifest["splits"]:
            raise ValueError("Remote splits/audio differ from captured local inputs")
        manifest["source"] = captured
    run_root.mkdir(parents=True)

    def save_manifest():
        temporary = run_root / "manifest.tmp"
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(run_root / "manifest.json")

    save_manifest()
    initial = evaluate(
        args.initial_model, args.split_dir / "test.csv", run_root / "initial_test.json"
    )
    manifest["initial_metrics"] = initial["metrics"]
    save_manifest()
    processed = prepare_dataset(args.split_dir, run_root)
    for name in getattr(args, "runs", ("fresh", "warm_start", "phoneme_multi")):
        target = run_root / name
        checkpoint = target / "checkpoint"
        cli = [
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--freeze-transformer-layers",
            "4",
            "--patience",
            str(getattr(args, "patience", 6)),
            "--num-workers",
            "0",
            "--skip-prep",
            "--no-hf-upload",
            "--no-onnx-export",
            "--no-confusion-pair-sampler",
            "--no-from-scratch-auto-tune",
            "--normalize-homophones",
            "--dataset-dir",
            str(processed),
            "--train-csv",
            str(args.split_dir / "train.csv"),
            "--val-csv",
            str(args.split_dir / "val.csv"),
            "--best-model-path",
            str(checkpoint),
            "--last-model-path",
            str(checkpoint),
        ]
        if name in ("fresh", "phoneme_multi"):
            cli += ["--no-resume"]
        else:
            cli += ["--resume-from", str(args.initial_model)]
        if name == "phoneme_multi":
            cli += [
                "--phoneme-multitask",
                "--lambda-cons",
                "0.5",
                "--lambda-vowel",
                "0.2",
            ]
        parsed = build_parser().parse_args(cli)
        parsed.seed = args.seed
        parsed.target_acc = 1.1  # Use validation F1/patience rather than stopping on raw accuracy.
        manifest["runs"][name] = {"status": "training", "arguments": cli, "seed": args.seed}
        save_manifest()
        train(parsed)
        validation = evaluate(checkpoint, args.split_dir / "val.csv", target / "validation.json")
        test = evaluate(checkpoint, args.split_dir / "test.csv", target / "test.json")
        manifest["runs"][name].update(
            status="complete",
            model_sha256=test["model_sha256"],
            validation_metrics=validation["metrics"],
            test_metrics=test["metrics"],
        )
        save_manifest()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # ONNX export is intentionally a separate confirmation after the common PyTorch comparison.
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPEAKER_SPLIT_DIR)
    parser.add_argument("--initial-model", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "experiments/speaker-independent"
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--runs",
        nargs="+",
        choices=("fresh", "warm_start", "phoneme_multi"),
        default=["fresh", "warm_start", "phoneme_multi"],
    )
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(audit_splits(args.split_dir), ensure_ascii=False, indent=2))
    elif not args.initial_model:
        parser.error("--initial-model is required for training")
    else:
        if not os.environ.get("VOICERECOGNIZER_CACHE_DIR"):
            raise ValueError("Set VOICERECOGNIZER_CACHE_DIR inside this worktree before training")
        Path(os.environ["VOICERECOGNIZER_CACHE_DIR"]).resolve().relative_to(PROJECT_ROOT.resolve())
        run(args)


if __name__ == "__main__":
    main()
