"""CPU probe: frozen encoder/projector, all Train, classifier-only device weighting."""

import argparse
import copy
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

from script.evaluate_speaker_independent import hash_file, read_manifest
from voicerecognizer.config import DEFAULT_SPEAKER_SPLIT_DIR, PROJECT_ROOT
from voicerecognizer.evaluation.evaluator import HOMOPHONE_MAP
from voicerecognizer.preprocessing.audio_preprocessor import AudioPreprocessor


def device_flags(rows):
    # Historical device manifest is evidence, not a complete recent-Rinry mapping.
    identified = set()
    for split in ("train", "val", "test"):
        with (PROJECT_ROOT / f"data_splits/speakerphone_{split}.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            identified.update(row["filepath"] for row in csv.DictReader(stream))
    return torch.tensor(
        [
            row["speaker"] in {"r1", "r6", "r7", "r8", "r9", "r10"} or row["filepath"] in identified
            for row in rows
        ]
    )


def metrics(rows, predictions, common):
    report = {}
    for normalized in (False, True):
        mapping = HOMOPHONE_MAP if normalized else {}
        truth = [mapping.get(row["label"], row["label"]) for row in rows]
        pred = [mapping.get(p, p) for p in predictions]
        per_speaker = {}
        for speaker in sorted({row["speaker"] for row in rows}):
            ids = [i for i, row in enumerate(rows) if row["speaker"] == speaker]
            voice = [i for i in ids if truth[i] != "other"]
            classes = sorted({truth[i] for i in voice})
            recalls = [
                sum(pred[i] == c for i in voice if truth[i] == c)
                / sum(truth[i] == c for i in voice)
                for c in classes
            ]
            per_speaker[speaker] = {
                "samples": len(ids),
                "voice_classes": len(classes),
                "voice_ba": float(np.mean(recalls)) if recalls else None,
                "accuracy": float(np.mean([truth[i] == pred[i] for i in ids])),
                "voice_macro_f1": float(
                    f1_score(
                        [truth[i] for i in voice],
                        [pred[i] for i in voice],
                        labels=classes,
                        average="macro",
                        zero_division=0,
                    )
                )
                if voice
                else None,
                "speech_false_reject": float(np.mean([pred[i] == "other" for i in voice]))
                if voice
                else None,
            }
        cohort = []
        for speaker in ("r2", "r4"):
            ids = [
                i
                for i, row in enumerate(rows)
                if row["speaker"] == speaker and truth[i] in common[normalized]
            ]
            cohort.append(
                float(
                    np.mean(
                        [
                            sum(pred[i] == c for i in ids if truth[i] == c)
                            / sum(truth[i] == c for i in ids)
                            for c in common[normalized]
                        ]
                    )
                )
            )
        other = [i for i, row in enumerate(rows) if row["label"] == "other"]
        report["normalized" if normalized else "raw"] = {
            "speakerphone_ba": float(np.mean(cohort)),
            "speakerphone_per_person_common_ba": dict(zip(("r2", "r4"), cohort, strict=True)),
            "common_classes": common[normalized],
            "per_speaker": per_speaker,
            "other_recall": float(np.mean([pred[i] == "other" for i in other])),
            "other_samples": len(other),
            "environment_false_accept": float(
                np.mean([pred[i] != "other" for i in other if rows[i]["speaker"].startswith("env")])
            )
            if any(rows[i]["speaker"].startswith("env") for i in other)
            else None,
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    train = read_manifest(DEFAULT_SPEAKER_SPLIT_DIR / "train.csv")
    val = read_manifest(DEFAULT_SPEAKER_SPLIT_DIR / "val.csv")
    assert not ({row["filepath"] for row in train} & {row["filepath"] for row in val})
    assert not ({row["speaker"] for row in train} & {row["speaker"] for row in val})
    common = {}
    for normalized in (False, True):
        mapping = HOMOPHONE_MAP if normalized else {}
        common[normalized] = sorted(
            set.intersection(
                *[
                    {
                        mapping.get(row["label"], row["label"])
                        for row in val
                        if row["speaker"] == speaker and row["label"] != "other"
                    }
                    for speaker in ("r2", "r4")
                ]
            )
        )
        assert common[normalized]
    flags = device_flags(train)
    provenance = {
        "scope": "classifier-only exploratory CPU experiment; no Test accessed",
        "initial_model_sha256": hash_file(args.model / "model.safetensors"),
        "split_sha256": {
            split: hash_file(DEFAULT_SPEAKER_SPLIT_DIR / f"{split}.csv")
            for split in ("train", "val")
        },
        "train_samples": len(train),
        "val_samples": len(val),
        "boosted_train_samples": int(flags.sum()),
        "boosted_groups": dict(
            Counter(row["speaker"] for row, flag in zip(train, flags, strict=True) if flag)
        ),
        "device_evidence": "r speakers: user; Rinry subset: historical speakerphone CSVs",
        "limitations": [
            "Exact recent Rinry range unknown",
            "Other-device mapping unresolved",
            "Rinry/collected person aliases unresolved",
            "No female Val person",
            "Encoder/projector frozen; no audio augmentation",
        ],
        "seeds": [42, 43, 44],
        "epochs": 20,
        "batch_size": 128,
        "learning_rate": 0.0001,
        "sampling": "Every Train row exactly once per epoch; common order across weights",
        "common_classes": {str(k): v for k, v in common.items()},
    }
    audio_hashes = {
        row["filepath"]: hash_file(PROJECT_ROOT / row["filepath"]) for row in train + val
    }
    assert not (
        {audio_hashes[row["filepath"]] for row in train}
        & {audio_hashes[row["filepath"]] for row in val}
    )
    (args.output / "inputs.json").write_text(
        json.dumps(
            {"settings": provenance, "rows": train + val, "audio_sha256": audio_hashes},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    labels = json.loads((args.model / "labels.json").read_text(encoding="utf-8"))
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        args.model, local_files_only=True
    ).eval()
    extractor = AutoFeatureExtractor.from_pretrained(args.model, local_files_only=True)
    preprocessor = AudioPreprocessor()
    captured = []
    hook = model.classifier.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().clone())
    )
    start = time.time()
    rows = train + val
    # Frozen features are identical for all weights/seeds; extract once, all files.
    with torch.inference_mode():
        for offset in range(0, len(rows), 16):
            waves = [
                preprocessor.preprocess_waveform(PROJECT_ROOT / row["filepath"])
                for row in rows[offset : offset + 16]
            ]
            inputs = extractor(waves, sampling_rate=16000, return_tensors="pt", padding=True)
            model(**inputs)
            if offset % 320 == 0 or offset + 16 >= len(rows):
                print(
                    f"features {min(offset + 16, len(rows))}/{len(rows)} "
                    f"elapsed={time.time() - start:.1f}s",
                    flush=True,
                )
    hook.remove()
    features = torch.cat(captured).clone()
    initial_head = copy.deepcopy(model.classifier)
    captured.clear()
    del model
    torch.save(features, args.output / "features.pt")
    train_x, val_x = features[: len(train)], features[len(train) :]
    target = torch.tensor([labels.index(row["label"]) for row in train])
    initial_predictions = [labels[i] for i in initial_head(val_x).argmax(-1).tolist()]
    result = {
        "settings": provenance,
        "initial": metrics(val, initial_predictions, common),
        "runs": [],
        "elapsed_seconds": None,
    }
    for seed in provenance["seeds"]:
        for weight in (1, 2, 4):
            head = copy.deepcopy(initial_head)
            optimizer = torch.optim.AdamW(head.parameters(), lr=0.0001)
            generator = torch.Generator().manual_seed(seed)
            weights = torch.where(flags, float(weight), 1.0)
            weights /= weights.mean()
            best, history = None, []
            for epoch in range(1, 21):
                order = torch.randperm(len(train), generator=generator)
                assert order.unique().numel() == len(train)
                for ids in order.split(128):
                    loss = (
                        torch.nn.functional.cross_entropy(
                            head(train_x[ids]), target[ids], reduction="none"
                        )
                        * weights[ids]
                    ).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                with torch.no_grad():
                    predictions = [labels[i] for i in head(val_x).argmax(-1).tolist()]
                report = metrics(val, predictions, common)
                score = report["normalized"]["speakerphone_ba"]
                history.append(
                    {
                        "epoch": epoch,
                        "speakerphone_ba": score,
                        "train_unique_used": len(train),
                        "train_unused": 0,
                    }
                )
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "epoch": epoch,
                        "metrics": report,
                        "predictions": predictions,
                        "state": copy.deepcopy(head.state_dict()),
                    }
            torch.save(best.pop("state"), args.output / f"head_seed{seed}_weight{weight}.pt")
            best.update(seed=seed, weight=weight, history=history)
            result["runs"].append(best)
            print(
                f"seed={seed} weight={weight} best_epoch={best['epoch']} "
                f"speakerphone_ba={best['score']:.4f}",
                flush=True,
            )
            result["elapsed_seconds"] = time.time() - start
            (args.output / "results.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )


if __name__ == "__main__":
    main()
