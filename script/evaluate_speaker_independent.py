"""Evaluate a fixed CSV, preserving every prediction and both label conventions."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

from voicerecognizer.config import PROJECT_ROOT
from voicerecognizer.config_labels import ALL_HIRAGANA_LABELS
from voicerecognizer.evaluation.evaluator import score_predictions
from voicerecognizer.preprocessing.audio_preprocessor import AudioPreprocessor


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    seen = set()
    for row in rows:
        relative = Path(row["filepath"].replace("\\", "/"))
        audio = (relative if relative.is_absolute() else PROJECT_ROOT / relative).resolve()
        audio.relative_to(PROJECT_ROOT.resolve())
        if not audio.is_file() or audio in seen:
            raise ValueError(f"Missing or repeated audio: {audio}")
        if row["label"] not in ALL_HIRAGANA_LABELS or not row.get("speaker"):
            raise ValueError(f"Invalid label/speaker: {row}")
        seen.add(audio)
        row["filepath"] = audio.relative_to(PROJECT_ROOT).as_posix()
    if not rows:
        raise ValueError(f"Empty split: {path}")
    return rows


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(
    model_path: str | Path,
    split_csv: str | Path,
    output_path: str | Path,
    backend: str = "pytorch",
) -> dict[str, Any]:
    rows = read_manifest(split_csv)
    model_path = Path(model_path)
    labels = json.loads((model_path / "labels.json").read_text(encoding="utf-8"))
    if set(labels) != set(ALL_HIRAGANA_LABELS):
        raise ValueError("Checkpoint must cover the declared 105 labels")
    if backend == "onnx":
        from voicerecognizer.recognizers.wav2vec2_recognizer import Wav2Vec2Recognizer

        recognizer = Wav2Vec2Recognizer(model_path=model_path, auto_download=False)

        def predict(path: Path) -> tuple[str, float | None]:
            prediction = recognizer.recognize(str(path))
            return prediction, getattr(recognizer, "last_confidence", None)

        artifact = recognizer.onnx_model_path
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = (
            Wav2Vec2ForSequenceClassification.from_pretrained(
                model_path,
                use_safetensors=True,
            )
            .to(device)
            .eval()
        )
        extractor = AutoFeatureExtractor.from_pretrained(model_path)
        preprocessor = AudioPreprocessor()

        def predict(path: Path) -> tuple[str, float | None]:
            waveform = preprocessor.preprocess_waveform(path)
            batch = extractor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.inference_mode():
                probabilities = torch.softmax(model(**batch).logits.float(), dim=-1)[0]
            confidence, index = probabilities.max(dim=-1)
            return labels[int(index)], float(confidence)

        artifact = model_path / "model.safetensors"
    if artifact is None:
        raise ValueError("Model artifact path is unavailable")
    records: list[dict[str, Any]] = []
    for row in rows:
        predicted, confidence = predict(PROJECT_ROOT / row["filepath"])
        records.append({**row, "predicted_label": predicted, "confidence": confidence})
    report = {
        "split_sha256": hash_file(split_csv),
        "model_sha256": hash_file(artifact),
        "backend": backend,
        "metrics": score_predictions(
            [row["label"] for row in records],
            [row["predicted_label"] for row in records],
            [row["speaker"] for row in records],
            ALL_HIRAGANA_LABELS,
        ),
        "records": records,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--split-csv", type=Path, default=PROJECT_ROOT / "data_splits/speaker_independent/test.csv"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("pytorch", "onnx"), default="pytorch")
    parser.add_argument("--normalize-homophones", action="store_true")
    args = parser.parse_args()
    report = evaluate(args.model_path, args.split_csv, args.output, args.backend)
    mode = "normalized" if args.normalize_homophones else "raw"
    print(json.dumps(report["metrics"][mode], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
