import csv
import json

from script.prepare_reviewed_training import prepare


def test_review_overrides_csv_labels_without_moving_audio(tmp_path, monkeypatch):
    root = tmp_path
    audit = root / "Docs/dataset_audit_2026-10-04"
    audit.mkdir(parents=True)
    evaluation = root / "evaluation_results"
    evaluation.mkdir()
    originals = {}
    baseline = {}
    for split, speaker in [("train", "old_train"), ("val", "old_val"), ("test", "take")]:
        path = root / f"dataset/{speaker}/a/001.wav"
        path.parent.mkdir(parents=True)
        path.write_bytes(speaker.encode())
        originals[path] = path.read_bytes()
        record = {"filepath": path.relative_to(root).as_posix(), "label": "a", "speaker": speaker}
        source = root / f"data_splits/speaker_independent/{split}.csv"
        source.parent.mkdir(parents=True, exist_ok=True)
        with source.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(record))
            writer.writeheader()
            writer.writerow(record)
        baseline[split] = record
    for number in range(1, 11):
        path = root / f"dataset/r{number}/a/001.wav"
        path.parent.mkdir(parents=True)
        path.write_bytes(bytes([number]))
        originals[path] = path.read_bytes()
    decisions = [
        {"filepath": "dataset/r2/a/001.wav", "label": "a", "decision": "relabel", "new_label": "i"},
        {"filepath": "dataset/r3/a/001.wav", "label": "a", "decision": "other"},
    ]
    (evaluation / "r_label_review_decisions.json").write_text(json.dumps({"decisions": decisions}))
    (evaluation / "r_model_review_candidates.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {"filepath": "dataset/r1/a/001.wav", "true_label": "a", "predicted_label": "i"},
                ]
            }
        )
    )
    (audit / "inventory.json").write_text(
        json.dumps({"expected_labels": ["a", "i", "other"], "exact_duplicate_groups": []})
    )
    folder = root / "dataset/collected/pc_test"
    folder.mkdir(parents=True)
    (folder / "new.wav").write_bytes(b"new")
    with (folder / "metadata.csv").open("w", newline="") as stream:
        csv.writer(stream).writerow(["new", "new.wav", "other", "a"])
    monkeypatch.setattr(
        "script.prepare_reviewed_training.subprocess.check_output",
        lambda arguments, **kwargs: (
            "dataset/collected/pc_test/new.wav\n" if "diff" in arguments else "test-head\n"
        ),
    )
    output = root / "prepared"
    summary = prepare(root, output)
    records = {}
    for name in ("train", "val", "test"):
        with (output / f"{name}.csv").open(newline="") as stream:
            split_records = {row["filepath"]: row for row in csv.DictReader(stream)}
            assert split_records[baseline[name]["filepath"]] == baseline[name]
            records.update(split_records)
    assert records["dataset/r2/a/001.wav"]["label"] == "i"
    assert records["dataset/r3/a/001.wav"]["label"] == "other"
    assert records["dataset/collected/pc_test/new.wav"]["label"] == "a"
    assert "dataset/r1/a/001.wav" not in records
    assert summary["excluded"] == [
        {"filepath": "dataset/r1/a/001.wav", "reason": "missing_review_decision"}
    ]
    assert all(path.read_bytes() == content for path, content in originals.items())
    assert all(value["preserved_rows"] == 1 for value in summary["base_manifests"].values())
    tts = root / "dataset/ai_female_nanami/a/001.wav"
    tts.parent.mkdir(parents=True)
    tts.write_bytes(b"tts")
    full_output = root / "all_prepared"
    full = prepare(root, full_output, accept_unreviewed=True, include_tts=True)
    with (full_output / "train.csv").open(newline="") as stream:
        full_train = {row["filepath"]: row for row in csv.DictReader(stream)}
    assert full_train["dataset/r1/a/001.wav"]["label"] == "a"
    assert full_train["dataset/ai_female_nanami/a/001.wav"]["speaker"] == "tts_nanami"
    assert full["tts_train_rows"] == 1
    assert not full["excluded"]
