"""Training module for Strategy 07: Wav2Vec2 + RIR Room Acoustics & Microphone Distortion Simulation."""

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
from voicerecognizer.preprocessing.audio_augmentor import AudioAugmentor
from voicerecognizer.strategies.phoneme_multi.model import (
    MultiTaskUncertaintyLoss,
    Wav2Vec2ForPhonemeMultiTaskClassification,
)
from voicerecognizer.strategies.phoneme_multi.phoneme_mapping import (
    build_label_phoneme_tables,
)
from voicerecognizer.strategies.registry import STRATEGIES_DIR

logger = logging.getLogger(__name__)

DEFAULT_RIR_STRATEGY_DIR = STRATEGIES_DIR / "wav2vec2_rir_simulation"


def fix_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RIRAugmentedDataset(Dataset):
    """Dataset providing audio waveforms augmented with RIR, mic EQ, and saturation."""

    def __init__(
        self,
        csv_path: Path,
        labels: tuple[str, ...] | list[str],
        sample_rate: int = 16000,
        target_length_seconds: float = 0.6,
        max_samples_per_class: int | None = None,
        augmentor: AudioAugmentor | None = None,
    ):
        self.labels = list(labels)
        self.label_to_idx = {lb: i for i, lb in enumerate(self.labels)}
        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * target_length_seconds)
        self.augmentor = augmentor
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

        logger.info(
            "Dataset %s loaded %d samples (augmentor=%s).",
            csv_path.name,
            len(self.samples),
            bool(augmentor),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        p, label_id = self.samples[idx]
        waveform, _ = sf.read(p, dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)

        # Apply online acoustic simulation (RIR, speakerphone bandpass, mic saturation)
        if self.augmentor is not None:
            waveform = self.augmentor.augment(waveform)

        if len(waveform) > self.target_samples:
            waveform = waveform[: self.target_samples]
        elif len(waveform) < self.target_samples:
            waveform = np.pad(waveform, (0, self.target_samples - len(waveform)))

        return {
            "waveform": waveform,
            "label": label_id,
        }


def collate_rir_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    waveforms = [item["waveform"] for item in batch]
    labels = [item["label"] for item in batch]

    return {
        "input_values": torch.tensor(np.stack(waveforms), dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def train_rir_simulation(args: argparse.Namespace) -> Path:
    fix_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Training Strategy 07: Wav2Vec2 + RIR Simulation & Mic Distortion on device: %s", device
    )

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

    label_source = (
        base_model_path
        if (base_model_path / "labels.json").exists()
        else DEFAULT_RECOGNITION_CONFIG.wav2vec2_best_model_dir
    )
    with open(label_source / "labels.json", encoding="utf-8") as f:
        labels = json.load(f)
    logger.info("Loaded %d classification labels.", len(labels))

    # Initialize Augmentor for training dataset
    augmentor = AudioAugmentor(
        noise_level=0.005,
        gain_range=(0.8, 1.2),
        shift_max_ratio=0.1,
        speed_range=(0.95, 1.05),
        pitch_shift_steps=(-0.5, 0.5),
        sample_rate=16000,
        p=args.augmentation_prob,
        seed=args.seed,
    )
    logger.info(
        "Configured AudioAugmentor with p=%.2f (RIR, speakerphone EQ, mic saturation, pitch/speed).",
        args.augmentation_prob,
    )

    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)

    train_ds = RIRAugmentedDataset(
        csv_path=train_csv,
        labels=labels,
        max_samples_per_class=args.max_samples_per_class,
        augmentor=augmentor,
    )
    val_ds = RIRAugmentedDataset(
        csv_path=val_csv,
        labels=labels,
        augmentor=None,  # Validation is always clean
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_rir_batch,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_rir_batch,
        num_workers=args.num_workers,
    )

    base_model_source = args.base_model
    logger.info("Initializing Model from: %s", base_model_source)

    config = AutoConfig.from_pretrained(
        base_model_source,
        num_labels=len(labels),
        finetuning_task="audio-classification",
    )
    student = Wav2Vec2ForPhonemeMultiTaskClassification.from_pretrained(
        base_model_source,
        config=config,
    ).to(device)

    # Phoneme label lookup tables
    c_indices, v_indices = build_label_phoneme_tables(labels)
    c_table = torch.tensor(c_indices, dtype=torch.long, device=device)
    v_table = torch.tensor(v_indices, dtype=torch.long, device=device)

    # Homoscedastic uncertainty weighting
    uncertainty_loss_fn = MultiTaskUncertaintyLoss(init_log_var=0.0).to(device)

    if args.freeze_layers > 0:
        student.freeze_feature_encoder()
        for layer in student.wav2vec2.encoder.layers[: args.freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False
        logger.info("Froze feature extractor and bottom %d encoder layers.", args.freeze_layers)

    params = list(student.parameters()) + list(uncertainty_loss_fn.parameters())
    optimizer = torch.optim.AdamW(
        [p for p in params if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        student.train()
        uncertainty_loss_fn.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"RIR Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            output = student(
                input_values=batch["input_values"],
                labels=batch["labels"],
                c_table=c_table,
                v_table=v_table,
                uncertainty_loss_fn=uncertainty_loss_fn,
            )

            loss = output.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(output.logits, dim=-1)
            correct_train += (preds == batch["labels"]).sum().item()
            total_train += len(batch["labels"])
            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{correct_train / total_train:.2%}"}
            )

        train_loss = total_loss / len(train_loader)
        train_acc = correct_train / total_train

        # Validation on clean split
        student.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                output = student(
                    input_values=batch["input_values"],
                    labels=batch["labels"],
                    c_table=c_table,
                    v_table=v_table,
                    uncertainty_loss_fn=uncertainty_loss_fn,
                )
                preds = torch.argmax(output.logits, dim=-1)
                correct_val += (preds == batch["labels"]).sum().item()
                total_val += len(batch["labels"])

        val_acc = correct_val / total_val
        weights_info = uncertainty_loss_fn.get_effective_weights()
        logger.info(
            "Epoch %d/%d - Train Loss: %.4f, Train Acc: %.2f%%, Val Acc: %.2f%% | "
            "Task Weights: Char=%.3f, Cons=%.3f, Vow=%.3f",
            epoch,
            args.epochs,
            train_loss,
            train_acc * 100.0,
            val_acc * 100.0,
            weights_info["weight_char"],
            weights_info["weight_cons"],
            weights_info["weight_vow"],
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            logger.info(
                "--> New best validation accuracy: %.2f%% (Saving checkpoint...)", val_acc * 100.0
            )

            inference_model = student.extract_inference_model()
            inference_model.save_pretrained(output_dir)

            with open(output_dir / "labels.json", "w", encoding="utf-8") as f:
                json.dump(labels, f, ensure_ascii=False, indent=2)

            metric_record = {
                "epoch": epoch,
                "val_acc": float(val_acc),
                "train_acc": float(train_acc),
                "train_loss": float(train_loss),
                "augmentation_prob": float(args.augmentation_prob),
                "uncertainty_weights": weights_info,
            }
            with open(output_dir / "best_metric.json", "w", encoding="utf-8") as f:
                json.dump(metric_record, f, indent=2)
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    args.patience,
                )
                break

    logger.info(
        "Training complete. Best Val Acc: %.2f%% at Epoch %d.",
        best_val_acc * 100.0,
        best_epoch,
    )

    logger.info("Exporting strategy model to ONNX format (model_mel_int8.onnx)...")
    try:
        export_and_benchmark(model_dir=output_dir, export_int8=True)
        logger.info("ONNX export succeeded: %s", output_dir / "model_mel_int8.onnx")
    except Exception as e:
        logger.error("ONNX export failed: %s", e, exc_info=True)

    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Strategy 07: Wav2Vec2 + RIR Room Acoustics & Microphone Distortion Simulation"
    )
    parser.add_argument("--epochs", type=int, default=14, help="Number of training epochs")
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience (number of epochs without improvement). Set to 0 to disable.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--freeze-layers", type=int, default=4, help="Bottom layers to freeze")
    parser.add_argument(
        "--augmentation-prob",
        type=float,
        default=0.5,
        help="Probability of applying each audio augmentation transform",
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
        help="Limit number of training samples per class for fast calibration",
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
        default=DEFAULT_RIR_STRATEGY_DIR,
        help="Destination directory for fine-tuned weights",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train_rir_simulation(args)


if __name__ == "__main__":
    main()
