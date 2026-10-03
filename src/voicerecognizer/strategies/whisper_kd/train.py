"""Training module for Strategy 02: Wav2Vec2 + Whisper Hidden Feature Distillation."""

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
from voicerecognizer.strategies.registry import STRATEGIES_DIR
from voicerecognizer.strategies.whisper_kd.model import Wav2Vec2ForWhisperKDClassification
from voicerecognizer.strategies.whisper_kd.teacher import (
    DEFAULT_WHISPER_TEACHER_MODEL_ID,
    WhisperTeacher,
    precompute_whisper_embeddings_cache,
)

logger = logging.getLogger(__name__)

DEFAULT_WHISPER_STRATEGY_DIR = STRATEGIES_DIR / "wav2vec2_whisper_kd"
DEFAULT_WHISPER_CACHE_PATH = PROJECT_ROOT / "checkpoints" / "whisper_teacher_cache.pt"


def fix_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WhisperKDSampleDataset(Dataset):
    """Dataset providing audio waveform, character label, and teacher Whisper embedding."""

    def __init__(
        self,
        csv_path: Path,
        labels: tuple[str, ...] | list[str],
        whisper_dim: int = 1280,
        embeddings_cache: dict[str, torch.Tensor] | None = None,
        sample_rate: int = 16000,
        target_length_seconds: float = 0.6,
        max_samples_per_class: int | None = None,
    ):
        self.labels = list(labels)
        self.label_to_idx = {lb: i for i, lb in enumerate(self.labels)}
        self.whisper_dim = whisper_dim
        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * target_length_seconds)
        self.embeddings_cache = embeddings_cache or {}
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
        cached_emb = self.embeddings_cache.get(resolved_key)
        if cached_emb is None:
            cached_emb = torch.zeros(self.whisper_dim, dtype=torch.float32)
        else:
            cached_emb = cached_emb.float()

        return {
            "waveform": waveform,
            "label": label_id,
            "teacher_whisper": cached_emb,
        }


def collate_whisper_kd_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    waveforms = [item["waveform"] for item in batch]
    labels = [item["label"] for item in batch]
    teacher_whispers = [item["teacher_whisper"] for item in batch]

    return {
        "input_values": torch.tensor(np.stack(waveforms), dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
        "teacher_whisper_embeddings": torch.stack(teacher_whispers),
    }


def train_whisper_kd(args: argparse.Namespace) -> Path:
    fix_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training Strategy 02: Wav2Vec2 + Whisper KD on device: %s", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model_path = Path(args.base_model)
    if (
        not (base_model_path / "model.safetensors").exists()
        and not (base_model_path / "pytorch_model.bin").exists()
    ):
        logger.info(
            "Base model weights not found at %s. Attempting to download from HF Hub...",
            base_model_path,
        )
        try:
            from voicerecognizer.utils.model_uploader import (
                download_latest_team_weights_if_needed,
            )

            download_latest_team_weights_if_needed(model_type="wav2vec2")
        except Exception as e:
            logger.warning("Could not auto-download weights from HF Hub: %s", e)

    labels_file = base_model_path / "labels.json"
    if labels_file.exists():
        labels = json.loads(labels_file.read_text(encoding="utf-8"))
        logger.info("Loaded %d labels matching base checkpoint: %s", len(labels), labels_file)
    else:
        labels = sorted(DEFAULT_RECOGNITION_CONFIG.labels)

    # 1. Load or build teacher embeddings cache
    cache_path = Path(args.teacher_cache)
    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)

    embeddings_cache: dict[str, torch.Tensor] = {}
    if cache_path.exists():
        logger.info("Loading existing Whisper teacher cache: %s", cache_path)
        embeddings_cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        # Determine dimension from sample
        whisper_dim = next(iter(embeddings_cache.values())).shape[0] if embeddings_cache else 1280
    else:
        logger.info(
            "Whisper teacher cache not found at %s. Initializing Whisper Teacher to precompute...",
            cache_path,
        )
        teacher = WhisperTeacher(
            model_id=args.teacher_model_id,
            device=device,
        )
        whisper_dim = teacher.hidden_dim
        precompute_whisper_embeddings_cache(
            index_csv_path=train_csv,
            output_cache_path=cache_path,
            teacher=teacher,
            batch_size=args.teacher_batch_size,
            max_samples_per_class=args.max_samples_per_class,
        )
        embeddings_cache = torch.load(cache_path, map_location="cpu", weights_only=False)

    # 2. Build Datasets & DataLoaders
    train_ds = WhisperKDSampleDataset(
        csv_path=train_csv,
        labels=labels,
        whisper_dim=whisper_dim,
        embeddings_cache=embeddings_cache,
        max_samples_per_class=args.max_samples_per_class,
    )
    val_ds = WhisperKDSampleDataset(
        csv_path=val_csv,
        labels=labels,
        whisper_dim=whisper_dim,
        embeddings_cache=embeddings_cache,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_whisper_kd_batch,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_whisper_kd_batch,
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
    student = Wav2Vec2ForWhisperKDClassification.from_pretrained(
        base_model_source,
        config=config,
        whisper_dim=whisper_dim,
    ).to(device)

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
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        student.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"Whisper KD Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            output = student(
                input_values=batch["input_values"],
                labels=batch["labels"],
                teacher_whisper_embeddings=batch["teacher_whisper_embeddings"],
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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            logger.info(
                "--> New best validation accuracy: %.2f%% (Saving checkpoint...)", val_acc * 100
            )

            clean_model = student.extract_inference_model()
            clean_model.save_pretrained(output_dir)

            (output_dir / "labels.json").write_text(
                json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            metrics = {
                "strategy": "wav2vec2_whisper_kd",
                "best_epoch": best_epoch,
                "best_val_acc": round(best_val_acc * 100, 2),
                "alpha_kd": args.alpha_kd,
                "teacher_model_id": args.teacher_model_id,
            }
            (output_dir / "strategy_metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        else:
            patience_counter += 1
            logger.info(
                "Validation accuracy did not improve (current: %.2f%%, best: %.2f%%, patience: %d/%d).",
                val_acc * 100,
                best_val_acc * 100,
                patience_counter,
                args.patience,
            )
            if args.patience > 0 and patience_counter >= args.patience:
                logger.info(
                    "Early stopping triggered at epoch %d after %d epochs without improvement.",
                    epoch,
                    patience_counter,
                )
                break

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
        description="Train Strategy 02: Wav2Vec2 + Whisper Hidden Feature Distillation"
    )
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Early stopping patience (number of epochs without improvement). Set to 0 to disable.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--freeze-layers", type=int, default=4, help="Bottom layers to freeze")
    parser.add_argument(
        "--alpha-kd",
        type=float,
        default=0.5,
        help="Weight for KD cosine loss (0.0=CE only, 1.0=KD only)",
    )
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
        default=DEFAULT_WHISPER_TEACHER_MODEL_ID,
        help="Teacher model ID on Hugging Face",
    )
    parser.add_argument(
        "--teacher-cache",
        type=Path,
        default=DEFAULT_WHISPER_CACHE_PATH,
        help="Path to precomputed teacher embeddings cache",
    )
    parser.add_argument(
        "--teacher-batch-size",
        type=int,
        default=32,
        help="Batch size when precomputing teacher embeddings",
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
        default=DEFAULT_WHISPER_STRATEGY_DIR,
        help="Destination directory for fine-tuned weights",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train_whisper_kd(args)


if __name__ == "__main__":
    main()
