"""Training module for Strategy 04: Wav2Vec2 + ArcFace Angular Margin Loss."""

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
from voicerecognizer.strategies.arcface.model import Wav2Vec2ForArcFaceClassification
from voicerecognizer.strategies.registry import STRATEGIES_DIR

logger = logging.getLogger(__name__)

DEFAULT_ARCFACE_STRATEGY_DIR = STRATEGIES_DIR / "wav2vec2_arcface"


def fix_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ArcFaceSampleDataset(Dataset):
    """Dataset providing audio waveform and syllable label."""

    def __init__(
        self,
        csv_path: Path,
        labels: tuple[str, ...] | list[str],
        sample_rate: int = 16000,
        target_length_seconds: float = 0.6,
        max_samples_per_class: int | None = None,
    ):
        self.labels = list(labels)
        self.label_to_idx = {lb: i for i, lb in enumerate(self.labels)}
        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * target_length_seconds)
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

        return {
            "waveform": waveform,
            "label": label_id,
        }


def collate_arcface_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    waveforms = [item["waveform"] for item in batch]
    labels = [item["label"] for item in batch]

    return {
        "input_values": torch.tensor(np.stack(waveforms), dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def train_arcface(args: argparse.Namespace) -> Path:
    fix_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training Strategy 04: Wav2Vec2 + ArcFace Angular Margin on device: %s", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model_path = Path(args.base_model)
    if not (base_model_path / "model.safetensors").exists() and not (
        base_model_path / "pytorch_model.bin"
    ).exists():
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

    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)

    train_ds = ArcFaceSampleDataset(
        csv_path=train_csv,
        labels=labels,
        max_samples_per_class=args.max_samples_per_class,
    )
    val_ds = ArcFaceSampleDataset(
        csv_path=val_csv,
        labels=labels,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_arcface_batch,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_arcface_batch,
        num_workers=args.num_workers,
    )

    base_model_source = args.base_model
    logger.info("Initializing ArcFace Model from: %s", base_model_source)

    config = AutoConfig.from_pretrained(
        base_model_source,
        num_labels=len(labels),
        finetuning_task="audio-classification",
    )
    student = Wav2Vec2ForArcFaceClassification(
        config=config,
        s=args.scale,
        m=args.margin,
    ).to(device)

    # Initialize ArcFace weight from base model classifier if available
    try:
        from transformers import Wav2Vec2ForSequenceClassification

        base_clf = Wav2Vec2ForSequenceClassification.from_pretrained(base_model_source)
        with torch.no_grad():
            if base_clf.classifier.weight.shape == student.arcface_head.weight.shape:
                student.arcface_head.weight.copy_(base_clf.classifier.weight)
                logger.info("Initialized ArcFace head weights from base model classifier.")
    except Exception as e:
        logger.warning("Could not warm-start ArcFace head from base classifier: %s", e)

    if args.freeze_layers > 0:
        student.freeze_feature_encoder()
        for layer in student.wav2vec2.encoder.layers[: args.freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False
        logger.info("Froze feature extractor and bottom %d encoder layers.", args.freeze_layers)

    head_lr = args.head_lr if args.head_lr is not None else args.lr * 20.0
    backbone_params = [
        p
        for n, p in student.named_parameters()
        if not n.startswith("arcface_head") and p.requires_grad
    ]
    head_params = [p for p in student.arcface_head.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    logger.info("Optimizer configured: backbone_lr=%.2e, head_lr=%.2e", args.lr, head_lr)

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    warmup_epochs = max(1, args.margin_warmup_epochs)
    target_margin = args.margin

    for epoch in range(1, args.epochs + 1):
        # Margin Warmup schedule
        if epoch <= warmup_epochs:
            current_margin = target_margin * (epoch / warmup_epochs)
        else:
            current_margin = target_margin
        student.arcface_head.set_margin(current_margin)
        logger.info(
            "Epoch %d/%d - Active ArcFace margin m = %.3f rad (%.1f deg)",
            epoch,
            args.epochs,
            current_margin,
            np.degrees(current_margin),
        )

        student.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"ArcFace Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            output = student(
                input_values=batch["input_values"],
                labels=batch["labels"],
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
                out = student(
                    input_values=batch["input_values"],
                    labels=batch["labels"],
                )
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
                "strategy": "wav2vec2_arcface",
                "best_epoch": best_epoch,
                "best_val_acc": round(best_val_acc * 100, 2),
                "scale_s": args.scale,
                "margin_m": args.margin,
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

    # Auto-export to INT8 ONNX
    logger.info("Exporting strategy model to ONNX format (model_mel_int8.onnx)...")
    try:
        export_and_benchmark(model_dir=output_dir, export_int8=True)
        logger.info("ONNX export succeeded: %s", output_dir / "model_mel_int8.onnx")
    except Exception as e:
        logger.error("ONNX export failed: %s", e, exc_info=True)

    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Strategy 04: Wav2Vec2 + ArcFace Angular Margin Loss"
    )
    parser.add_argument("--epochs", type=int, default=18, help="Number of training epochs")
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience (number of epochs without improvement). Set to 0 to disable.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for backbone")
    parser.add_argument(
        "--head-lr",
        type=float,
        default=None,
        help="Learning rate for ArcFace head (defaults to lr * 20.0)",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--freeze-layers", type=int, default=4, help="Bottom layers to freeze")
    parser.add_argument("--scale", "-s", type=float, default=24.0, help="ArcFace radius scale s")
    parser.add_argument(
        "--margin", "-m", type=float, default=0.20, help="ArcFace angular margin m in radians"
    )
    parser.add_argument(
        "--margin-warmup-epochs",
        type=int,
        default=3,
        help="Number of epochs to linearly ramp up margin from 0 to target margin",
    )
    parser.add_argument(
        "--alpha-kd",
        type=float,
        default=None,
        help="Ignored (for Colab runner CLI compatibility)",
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
        default=DEFAULT_ARCFACE_STRATEGY_DIR,
        help="Destination directory for fine-tuned weights",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train_arcface(args)


if __name__ == "__main__":
    main()
