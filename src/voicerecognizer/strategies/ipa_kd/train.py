"""Training module for Strategy 01: Wav2Vec2 + XLS-R IPA Knowledge Distillation."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig

from voicerecognizer.config import (
    DEFAULT_RECOGNITION_CONFIG,
    PROJECT_ROOT,
)
from voicerecognizer.models.wav2vec2.export_onnx import export_and_benchmark
from voicerecognizer.strategies.ipa_kd.model import Wav2Vec2ForIPAKDClassification
from voicerecognizer.strategies.ipa_kd.teacher import (
    DEFAULT_IPA_TEACHER_MODEL_ID,
    IPA_VOCAB_SIZE,
    IPATeacher,
    precompute_ipa_posteriors_cache,
)
from voicerecognizer.strategies.registry import STRATEGIES_DIR

logger = logging.getLogger(__name__)

DEFAULT_IPA_STRATEGY_DIR = STRATEGIES_DIR / "wav2vec2_ipa_kd"
DEFAULT_CACHE_PATH = PROJECT_ROOT / "checkpoints" / "ipa_teacher_cache.pt"


def fix_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IPAKDSampleDataset(Dataset):
    """Dataset providing audio waveform, character label, and teacher IPA posterior."""

    def __init__(
        self,
        csv_path: Path,
        labels: tuple[str, ...] | list[str],
        posteriors_cache: dict[str, torch.Tensor] | None = None,
        sample_rate: int = 16000,
        target_length_seconds: float = 0.6,
        max_samples_per_class: int | None = None,
    ):
        self.labels = list(labels)
        self.label_to_idx = {lb: i for i, lb in enumerate(self.labels)}
        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * target_length_seconds)
        self.posteriors_cache = posteriors_cache or {}
        self.samples: list[tuple[Path, int]] = []
        counts: dict[int, int] = {}

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filepath_str = row.get("filepath") or row.get("source_filepath") or ""
                label_str = row.get("label") or ""
                if not filepath_str or label_str not in self.label_to_idx:
                    continue
                label_id = self.label_to_idx[label_str]
                if max_samples_per_class is not None and max_samples_per_class > 0:
                    if counts.get(label_id, 0) >= max_samples_per_class:
                        continue
                    counts[label_id] = counts.get(label_id, 0) + 1

                p = Path(filepath_str)
                if not (p.is_absolute() and p.exists()):
                    if (csv_path.parent / p).exists():
                        p = csv_path.parent / p
                    elif (PROJECT_ROOT / p).exists():
                        p = PROJECT_ROOT / p
                    elif (PROJECT_ROOT / "processed_dataset" / p).exists():
                        p = PROJECT_ROOT / "processed_dataset" / p
                    else:
                        continue
                self.samples.append((p, label_id))

        if not self.samples:
            raise ValueError(f"No valid audio samples found in {csv_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        p, label_id = self.samples[idx]
        waveform, _ = sf.read(p, dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)

        if len(waveform) > self.target_samples:
            waveform = waveform[: self.target_samples]
        elif len(waveform) < self.target_samples:
            waveform = np.pad(waveform, (0, self.target_samples - len(waveform)))

        resolved_key = str(p.resolve())
        cached_post = self.posteriors_cache.get(resolved_key)
        if cached_post is None:
            # Fallback uniform distribution if missing
            cached_post = torch.full((IPA_VOCAB_SIZE,), 1.0 / IPA_VOCAB_SIZE, dtype=torch.float32)
        else:
            cached_post = cached_post.float()

        return {
            "waveform": waveform,
            "label": label_id,
            "teacher_ipa": cached_post,
        }


def collate_kd_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    waveforms = [item["waveform"] for item in batch]
    labels = [item["label"] for item in batch]
    teacher_ipas = [item["teacher_ipa"] for item in batch]

    return {
        "input_values": torch.tensor(np.stack(waveforms), dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
        "teacher_ipa_posteriors": torch.stack(teacher_ipas),
    }


def train_ipa_kd(args: argparse.Namespace) -> Path:
    fix_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training Strategy 01: Wav2Vec2 + XLS-R IPA KD on device: %s", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model_path = Path(args.base_model)
    labels_file = base_model_path / "labels.json"
    if labels_file.exists():
        labels = json.loads(labels_file.read_text(encoding="utf-8"))
        logger.info("Loaded %d labels matching base checkpoint: %s", len(labels), labels_file)
    else:
        labels = sorted(DEFAULT_RECOGNITION_CONFIG.labels)

    # 1. Load or build teacher posteriors cache
    cache_path = Path(args.teacher_cache)
    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)

    posteriors_cache: dict[str, torch.Tensor] = {}
    if cache_path.exists():
        logger.info("Loading existing IPA teacher cache: %s", cache_path)
        posteriors_cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    else:
        logger.info(
            "IPA teacher cache not found at %s. Initializing Teacher to precompute...", cache_path
        )
        teacher = IPATeacher(
            model_id=args.teacher_model_id,
            device=device,
            temperature=args.temperature,
        )
        precompute_ipa_posteriors_cache(
            index_csv_path=train_csv,
            output_cache_path=cache_path,
            teacher=teacher,
            batch_size=args.teacher_batch_size,
            max_samples_per_class=args.max_samples_per_class,
        )
        posteriors_cache = torch.load(cache_path, map_location="cpu", weights_only=False)

    # 2. Build Datasets & DataLoaders
    train_ds = IPAKDSampleDataset(
        csv_path=train_csv,
        labels=labels,
        posteriors_cache=posteriors_cache,
        max_samples_per_class=args.max_samples_per_class,
    )
    val_ds = IPAKDSampleDataset(
        csv_path=val_csv,
        labels=labels,
        posteriors_cache=posteriors_cache,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_kd_batch,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_kd_batch,
        num_workers=args.num_workers,
    )

    # 3. Initialize Student Model
    base_model_source = args.base_model
    logger.info("Initializing Student model from: %s", base_model_source)

    config = AutoConfig.from_pretrained(
        base_model_source,
        num_labels=len(labels),
        finetuning_task="audio-classification",
    )
    student = Wav2Vec2ForIPAKDClassification.from_pretrained(
        base_model_source,
        config=config,
        ipa_vocab_size=IPA_VOCAB_SIZE,
        temperature=args.temperature,
    ).to(device)

    # Freeze lower transformer layers if specified
    if args.freeze_layers > 0:
        student.freeze_feature_encoder()
        for layer in student.wav2vec2.encoder.layers[: args.freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False
        logger.info("Froze feature extractor and bottom %d encoder layers.", args.freeze_layers)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        student.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"IPA KD Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            output = student(
                input_values=batch["input_values"],
                labels=batch["labels"],
                teacher_ipa_posteriors=batch["teacher_ipa_posteriors"],
                alpha_kd=args.alpha_kd,
            )

            loss = output.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = output.logits.argmax(dim=-1)
            correct_train += (preds == batch["labels"]).sum().item()
            total_train += batch["labels"].size(0)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct_train / max(1, total_train):.2%}",
            )

        # Validation
        student.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = student(input_values=batch["input_values"], labels=batch["labels"])
                preds = out.logits.argmax(dim=-1)
                correct_val += (preds == batch["labels"]).sum().item()
                total_val += batch["labels"].size(0)

        val_acc = correct_val / max(1, total_val)
        logger.info(
            "Epoch %d/%d - Train Loss: %.4f, Train Acc: %.2f%%, Val Acc: %.2f%%",
            epoch,
            args.epochs,
            total_loss / len(train_loader),
            (correct_train / total_train) * 100,
            val_acc * 100,
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            logger.info(
                "--> New best validation accuracy: %.2f%% (Saving checkpoint...)", val_acc * 100
            )

            # Save clean inference model for standard pipeline and ONNX export
            clean_model = student.extract_inference_model()
            clean_model.save_pretrained(output_dir)

            # Save labels.json
            (output_dir / "labels.json").write_text(
                json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            # Save metrics
            metrics = {
                "strategy": "wav2vec2_ipa_kd",
                "best_epoch": best_epoch,
                "best_val_acc": round(best_val_acc * 100, 2),
                "alpha_kd": args.alpha_kd,
                "temperature": args.temperature,
                "teacher_model_id": args.teacher_model_id,
            }
            (output_dir / "strategy_metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    logger.info(
        "Training complete. Best Val Acc: %.2f%% at Epoch %d.", best_val_acc * 100, best_epoch
    )

    # 4. Auto-export to INT8 ONNX
    logger.info("Exporting strategy model to ONNX format (model_mel_int8.onnx)...")
    try:
        export_and_benchmark(model_dir=output_dir, export_int8=True)
        logger.info("ONNX export succeeded: %s", output_dir / "model_mel_int8.onnx")
    except Exception as e:
        logger.error("ONNX export failed: %s", e, exc_info=True)

    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Strategy 01: Wav2Vec2 + XLS-R IPA Knowledge Distillation"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--freeze-layers", type=int, default=8, help="Bottom layers to freeze")
    parser.add_argument(
        "--alpha-kd", type=float, default=0.5, help="Weight for KD loss (0.0=CE only, 1.0=KD only)"
    )
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Limit number of training samples per class for fast local calibration",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=str(DEFAULT_RECOGNITION_CONFIG.wav2vec2_best_model_dir),
        help="Base model checkpoint or Hugging Face ID",
    )
    parser.add_argument(
        "--teacher-model-id",
        type=str,
        default=DEFAULT_IPA_TEACHER_MODEL_ID,
        help="Teacher model ID on Hugging Face",
    )
    parser.add_argument(
        "--teacher-cache",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Path to precomputed teacher posteriors cache",
    )
    parser.add_argument(
        "--teacher-batch-size",
        type=int,
        default=16,
        help="Batch size when precomputing teacher posteriors",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=PROJECT_ROOT / "data_splits" / "combined_train.csv",
        help="Training dataset split CSV",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=PROJECT_ROOT / "data_splits" / "combined_val.csv",
        help="Validation dataset split CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_IPA_STRATEGY_DIR,
        help="Destination directory for fine-tuned weights",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train_ipa_kd(args)


if __name__ == "__main__":
    main()
