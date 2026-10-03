"""Training module for Strategy 06: Wav2Vec2 Consonant/Vowel Multi-Task Learning."""

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
from voicerecognizer.strategies.phoneme_multi.model import (
    MultiTaskUncertaintyLoss,
    Wav2Vec2ForPhonemeMultiTaskClassification,
)
from voicerecognizer.strategies.phoneme_multi.phoneme_mapping import (
    CONSONANTS,
    VOWELS,
    build_label_phoneme_tables,
)
from voicerecognizer.strategies.registry import STRATEGIES_DIR

logger = logging.getLogger(__name__)

DEFAULT_PHONEME_STRATEGY_DIR = STRATEGIES_DIR / "wav2vec2_phoneme_multi"


def fix_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PhonemeMultiDataset(Dataset):
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


def collate_multi_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    waveforms = [item["waveform"] for item in batch]
    labels = [item["label"] for item in batch]

    return {
        "input_values": torch.tensor(np.stack(waveforms), dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def train_phoneme_multi(args: argparse.Namespace) -> Path:
    fix_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training Strategy 06: Wav2Vec2 Consonant/Vowel Multi-Task on device: %s", device)

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

    # Build lookup tables mapping label_id -> consonant_id, vowel_id
    c_indices, v_indices = build_label_phoneme_tables(labels)
    c_table = torch.tensor(c_indices, dtype=torch.long, device=device)
    v_table = torch.tensor(v_indices, dtype=torch.long, device=device)

    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)

    train_ds = PhonemeMultiDataset(
        csv_path=train_csv,
        labels=labels,
        max_samples_per_class=args.max_samples_per_class,
    )
    val_ds = PhonemeMultiDataset(
        csv_path=val_csv,
        labels=labels,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_multi_batch,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_multi_batch,
        num_workers=args.num_workers,
    )

    base_model_source = args.base_model
    logger.info("Initializing Multi-Task Model from: %s", base_model_source)

    config = AutoConfig.from_pretrained(
        base_model_source,
        num_labels=len(labels),
        finetuning_task="audio-classification",
    )
    config.num_consonants = len(CONSONANTS)
    config.num_vowels = len(VOWELS)

    model = Wav2Vec2ForPhonemeMultiTaskClassification.from_pretrained(
        base_model_source,
        config=config,
    ).to(device)

    if args.freeze_layers > 0:
        model.freeze_feature_encoder()
        for layer in model.wav2vec2.encoder.layers[: args.freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False
        logger.info("Froze feature extractor and bottom %d encoder layers.", args.freeze_layers)

    uncertainty_loss_fn = MultiTaskUncertaintyLoss().to(device) if args.use_uncertainty else None
    params = list(model.parameters())
    if uncertainty_loss_fn is not None:
        params.extend(list(uncertainty_loss_fn.parameters()))

    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct_char = 0
        correct_cons = 0
        correct_vow = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"Phoneme Multi Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            output = model(
                input_values=batch["input_values"],
                labels=batch["labels"],
                c_table=c_table,
                v_table=v_table,
                uncertainty_loss_fn=uncertainty_loss_fn,
                lambda_cons=args.lambda_cons,
                lambda_vowel=args.lambda_vowel,
            )

            loss = output.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            b_labels = batch["labels"]
            target_cons = c_table[b_labels]
            target_vow = v_table[b_labels]

            preds_char = output.char_logits.argmax(dim=-1)
            preds_cons = output.cons_logits.argmax(dim=-1)
            preds_vow = output.vowel_logits.argmax(dim=-1)

            correct_char += (preds_char == b_labels).sum().item()
            correct_cons += (preds_cons == target_cons).sum().item()
            correct_vow += (preds_vow == target_vow).sum().item()
            total_train += b_labels.size(0)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                c_acc=f"{correct_char / max(1, total_train):.1%}",
                cons=f"{correct_cons / max(1, total_train):.1%}",
                vow=f"{correct_vow / max(1, total_train):.1%}",
            )

        # Validation
        model.eval()
        val_correct_char = 0
        val_correct_cons = 0
        val_correct_vow = 0
        total_val = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(
                    input_values=batch["input_values"],
                    labels=batch["labels"],
                    c_table=c_table,
                    v_table=v_table,
                )
                b_labels = batch["labels"]
                target_cons = c_table[b_labels]
                target_vow = v_table[b_labels]

                val_correct_char += (out.char_logits.argmax(dim=-1) == b_labels).sum().item()
                val_correct_cons += (out.cons_logits.argmax(dim=-1) == target_cons).sum().item()
                val_correct_vow += (out.vowel_logits.argmax(dim=-1) == target_vow).sum().item()
                total_val += b_labels.size(0)

        val_acc = val_correct_char / max(1, total_val)
        cons_acc = val_correct_cons / max(1, total_val)
        vow_acc = val_correct_vow / max(1, total_val)

        weights_info = (
            uncertainty_loss_fn.get_effective_weights() if uncertainty_loss_fn else {}
        )
        logger.info(
            "Epoch %d/%d - Loss: %.4f | Val Char Acc: %.2f%%, Cons Acc: %.2f%%, Vow Acc: %.2f%% | Weights: %s",
            epoch,
            args.epochs,
            total_loss / len(train_loader),
            val_acc * 100,
            cons_acc * 100,
            vow_acc * 100,
            weights_info,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            logger.info(
                "--> New best validation accuracy: %.2f%% (Saving checkpoint...)", val_acc * 100
            )

            clean_model = model.extract_inference_model()
            clean_model.save_pretrained(output_dir)

            (output_dir / "labels.json").write_text(
                json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            metrics = {
                "strategy": "wav2vec2_phoneme_multi",
                "best_epoch": best_epoch,
                "best_val_acc": round(best_val_acc * 100, 2),
                "val_cons_acc": round(cons_acc * 100, 2),
                "val_vow_acc": round(vow_acc * 100, 2),
                "use_uncertainty": args.use_uncertainty,
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
        description="Train Strategy 06: Wav2Vec2 Consonant/Vowel Multi-Task Learning"
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
        default=None,
        help="Ignored (for Colab runner CLI compatibility)",
    )
    parser.add_argument(
        "--use-uncertainty",
        action="store_true",
        default=True,
        help="Use Kendall homoscedastic uncertainty weighting",
    )
    parser.add_argument("--lambda-cons", type=float, default=0.5, help="Fixed consonant loss weight")
    parser.add_argument("--lambda-vowel", type=float, default=0.5, help="Fixed vowel loss weight")
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
        default=DEFAULT_PHONEME_STRATEGY_DIR,
        help="Destination directory for fine-tuned weights",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train_phoneme_multi(args)


if __name__ == "__main__":
    main()
