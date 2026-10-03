"""Training module for Strategy: Wav2Vec2 Consonant/Vowel Multi-Task + XLS-R IPA KD Hybrid."""

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
from voicerecognizer.strategies.ipa_kd.teacher import (
    DEFAULT_IPA_TEACHER_MODEL_ID,
    IPA_VOCAB_SIZE,
    IPATeacher,
)
from voicerecognizer.strategies.phoneme_ipa_hybrid.model import (
    MultiTaskUncertaintyLoss,
    Wav2Vec2ForPhonemeIPAHybridClassification,
)
from voicerecognizer.strategies.phoneme_multi.phoneme_mapping import (
    build_label_phoneme_tables,
)
from voicerecognizer.strategies.registry import STRATEGIES_DIR

logger = logging.getLogger(__name__)

DEFAULT_HYBRID_STRATEGY_DIR = STRATEGIES_DIR / "wav2vec2_phoneme_ipa_hybrid"
DEFAULT_IPA_CACHE_PATH = STRATEGIES_DIR / "ipa_kd_cache" / "train_val_posteriors.pt"


def fix_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PhonemeIPAHybridDataset(Dataset):
    """Dataset providing audio waveform, label, and cached teacher IPA posteriors."""

    def __init__(
        self,
        csv_path: Path,
        labels: tuple[str, ...] | list[str],
        posteriors_cache: dict[str, torch.Tensor],
        sample_rate: int = 16000,
        target_length_seconds: float = 0.6,
        max_samples_per_class: int | None = None,
        augmentor: AudioAugmentor | None = None,
    ):
        self.labels = list(labels)
        self.label_to_idx = {lb: i for i, lb in enumerate(self.labels)}
        self.posteriors_cache = posteriors_cache
        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * target_length_seconds)
        self.augmentor = augmentor
        self.samples: list[tuple[Path, int]] = []
        counts: dict[int, int] = {}

        cache_hits = 0
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

                # Check cache hit
                post = self._lookup_cache(p)
                if post is not None:
                    cache_hits += 1
                self.samples.append((p, label_id))

        if not self.samples:
            raise ValueError(f"No valid audio samples found in {csv_path}")

        hit_rate = (cache_hits / len(self.samples)) * 100.0 if self.samples else 0.0
        logger.info(
            "Dataset %s loaded %d samples. Teacher IPA cache hit rate: %.1f%% (%d/%d)",
            csv_path.name,
            len(self.samples),
            hit_rate,
            cache_hits,
            len(self.samples),
        )

    def _lookup_cache(self, p: Path) -> torch.Tensor | None:
        """Robust multi-key cache lookup supporting Windows and Linux path formats."""
        # 1. Exact string
        k1 = str(p.resolve())
        if k1 in self.posteriors_cache:
            return self.posteriors_cache[k1]
        # 2. Relative to project root
        try:
            k2 = p.resolve().relative_to(PROJECT_ROOT).as_posix()
            if k2 in self.posteriors_cache:
                return self.posteriors_cache[k2]
        except Exception:
            pass
        # 3. Posix path
        k3 = p.as_posix()
        if k3 in self.posteriors_cache:
            return self.posteriors_cache[k3]
        # 4. Filename
        if p.name in self.posteriors_cache:
            return self.posteriors_cache[p.name]
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        p, label_id = self.samples[idx]
        waveform, _ = sf.read(p, dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)

        if self.augmentor is not None:
            waveform = self.augmentor.augment(waveform)

        if len(waveform) > self.target_samples:
            waveform = waveform[: self.target_samples]
        elif len(waveform) < self.target_samples:
            waveform = np.pad(waveform, (0, self.target_samples - len(waveform)))

        post = self._lookup_cache(p)
        if post is None:
            # Fallback uniform distribution if missing
            post = torch.full((IPA_VOCAB_SIZE,), 1.0 / IPA_VOCAB_SIZE, dtype=torch.float32)
        else:
            post = post.float()

        return {
            "waveform": waveform,
            "label": label_id,
            "teacher_ipa": post,
        }


def collate_hybrid_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    waveforms = [item["waveform"] for item in batch]
    labels = [item["label"] for item in batch]
    teacher_ipas = [item["teacher_ipa"] for item in batch]

    return {
        "input_values": torch.tensor(np.stack(waveforms), dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
        "teacher_ipa_posteriors": torch.stack(teacher_ipas, dim=0),
    }


def precompute_ipa_cache_if_needed(
    csv_paths: list[Path],
    cache_path: Path,
    device: torch.device,
    teacher_model_name: str = DEFAULT_IPA_TEACHER_MODEL_ID,
    temperature: float = 2.0,
    batch_size: int = 16,
) -> dict[str, torch.Tensor]:
    """Precompute and cache teacher posteriors for all dataset files."""
    if cache_path.exists():
        logger.info("Loading existing IPA teacher posteriors cache from %s", cache_path)
        try:
            return torch.load(cache_path, map_location="cpu", weights_only=True)
        except Exception:
            return torch.load(cache_path, map_location="cpu", weights_only=False)

    logger.info("IPA cache not found at %s. Precomputing via %s...", cache_path, teacher_model_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    teacher = IPATeacher(model_name=teacher_model_name, device=device, temperature=temperature)

    unique_paths: set[Path] = set()
    for cp in csv_paths:
        if not cp.exists():
            continue
        with open(cp, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fp = row.get("filepath") or row.get("source_filepath") or ""
                if not fp:
                    continue
                p = Path(fp)
                if not (p.is_absolute() and p.exists()):
                    if (cp.parent / p).exists():
                        p = cp.parent / p
                    elif (PROJECT_ROOT / p).exists():
                        p = PROJECT_ROOT / p
                    elif (PROJECT_ROOT / "processed_dataset" / p).exists():
                        p = PROJECT_ROOT / "processed_dataset" / p
                    else:
                        continue
                unique_paths.add(p)

    target_samples = int(16000 * 0.6)
    posteriors_by_key: dict[str, torch.Tensor] = {}
    batch_wavs: list[torch.Tensor] = []
    batch_paths: list[Path] = []

    for p in tqdm(sorted(unique_paths), desc="Precomputing Teacher Posteriors"):
        try:
            wav, _ = sf.read(p, dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = np.mean(wav, axis=1)
            if len(wav) > target_samples:
                wav = wav[:target_samples]
            elif len(wav) < target_samples:
                wav = np.pad(wav, (0, target_samples - len(wav)))

            batch_wavs.append(torch.from_numpy(wav).float())
            batch_paths.append(p)

            if len(batch_wavs) >= batch_size:
                posts = teacher.extract_pooled_posteriors(torch.stack(batch_wavs, dim=0))
                for bp, post in zip(batch_paths, posts, strict=True):
                    fp16_post = post.half()
                    # Store multiple keys for robust cross-environment portability
                    posteriors_by_key[str(bp.resolve())] = fp16_post
                    posteriors_by_key[bp.as_posix()] = fp16_post
                    posteriors_by_key[bp.name] = fp16_post
                    try:
                        posteriors_by_key[bp.resolve().relative_to(PROJECT_ROOT).as_posix()] = fp16_post
                    except Exception:
                        pass
                batch_wavs = []
                batch_paths = []
        except Exception as e:
            logger.warning("Error processing %s: %s", p, e)

    if batch_wavs:
        posts = teacher.extract_pooled_posteriors(torch.stack(batch_wavs, dim=0))
        for bp, post in zip(batch_paths, posts, strict=True):
            fp16_post = post.half()
            posteriors_by_key[str(bp.resolve())] = fp16_post
            posteriors_by_key[bp.as_posix()] = fp16_post
            posteriors_by_key[bp.name] = fp16_post
            try:
                posteriors_by_key[bp.resolve().relative_to(PROJECT_ROOT).as_posix()] = fp16_post
            except Exception:
                pass

    torch.save(posteriors_by_key, cache_path)
    logger.info("Successfully cached %d entries into %s", len(posteriors_by_key), cache_path)
    return posteriors_by_key


def train_phoneme_ipa_hybrid(args: argparse.Namespace) -> Path:
    fix_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training Strategy: Phoneme Multi-Task + XLS-R IPA KD Hybrid on %s", device)

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

    label_source = (
        base_model_path
        if (base_model_path / "labels.json").exists()
        else DEFAULT_RECOGNITION_CONFIG.wav2vec2_best_model_dir
    )
    with open(label_source / "labels.json", encoding="utf-8") as f:
        labels = json.load(f)
    logger.info("Loaded %d classification labels.", len(labels))

    # Precompute or load IPA cache
    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)
    cache_path = Path(args.ipa_cache_path)
    posteriors_cache = precompute_ipa_cache_if_needed(
        csv_paths=[train_csv, val_csv],
        cache_path=cache_path,
        device=device,
        teacher_model_name=args.teacher_model,
        temperature=args.temperature,
    )

    augmentor = None
    if getattr(args, "augmentation", False):
        augmentor = AudioAugmentor(p=getattr(args, "augmentation_prob", 0.5))
        logger.info("Audio Data Augmentation enabled (speakerphone EQ, reverb, noise, pitch/speed).")

    train_ds = PhonemeIPAHybridDataset(
        csv_path=train_csv,
        labels=labels,
        posteriors_cache=posteriors_cache,
        max_samples_per_class=args.max_samples_per_class,
        augmentor=augmentor,
    )
    val_ds = PhonemeIPAHybridDataset(
        csv_path=val_csv,
        labels=labels,
        posteriors_cache=posteriors_cache,
        augmentor=None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_hybrid_batch,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_hybrid_batch,
        num_workers=args.num_workers,
    )

    base_model_source = args.base_model
    logger.info("Initializing Hybrid model from: %s", base_model_source)

    config = AutoConfig.from_pretrained(
        base_model_source,
        num_labels=len(labels),
        finetuning_task="audio-classification",
    )
    student = Wav2Vec2ForPhonemeIPAHybridClassification.from_pretrained(
        base_model_source,
        config=config,
        ipa_vocab_size=IPA_VOCAB_SIZE,
        temperature=args.temperature,
    ).to(device)

    # Phoneme label lookup tables
    c_indices, v_indices = build_label_phoneme_tables(labels)
    c_table = torch.tensor(c_indices, dtype=torch.long, device=device)
    v_table = torch.tensor(v_indices, dtype=torch.long, device=device)

    # Homoscedastic uncertainty weighting module for multi-task loss
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

        pbar = tqdm(train_loader, desc=f"Hybrid Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            output = student(
                input_values=batch["input_values"],
                labels=batch["labels"],
                c_table=c_table,
                v_table=v_table,
                teacher_ipa_posteriors=batch["teacher_ipa_posteriors"],
                alpha_kd=args.alpha_kd,
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
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct_train / total_train:.2%}"})

        train_loss = total_loss / len(train_loader)
        train_acc = correct_train / total_train

        # Validation
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
                    teacher_ipa_posteriors=batch["teacher_ipa_posteriors"],
                    alpha_kd=args.alpha_kd,
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
            logger.info("--> New best validation accuracy: %.2f%% (Saving checkpoint...)", val_acc * 100.0)

            # Extract standard inference model (stripping consonant, vowel, and ipa auxiliary heads)
            inference_model = student.extract_inference_model()
            inference_model.save_pretrained(output_dir)

            with open(output_dir / "labels.json", "w", encoding="utf-8") as f:
                json.dump(labels, f, ensure_ascii=False, indent=2)

            metric_record = {
                "epoch": epoch,
                "val_acc": float(val_acc),
                "train_acc": float(train_acc),
                "train_loss": float(train_loss),
                "alpha_kd": float(args.alpha_kd),
                "temperature": float(args.temperature),
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
        description="Train Strategy: Phoneme Multi-Task + XLS-R IPA KD Hybrid"
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
        default=0.20,
        help="Weight for XLS-R IPA distillation loss",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="Distillation temperature scaling for soft targets",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default=DEFAULT_IPA_TEACHER_MODEL_ID,
        help="Teacher model Hugging Face ID",
    )
    parser.add_argument(
        "--ipa-cache-path",
        type=Path,
        default=DEFAULT_IPA_CACHE_PATH,
        help="Path to precomputed teacher posteriors tensor cache",
    )
    parser.add_argument(
        "--augmentation",
        action="store_true",
        default=False,
        help="Enable audio data augmentation (speakerphone EQ, reverb, pitch/speed)",
    )
    parser.add_argument(
        "--augmentation-prob",
        type=float,
        default=0.5,
        help="Probability of applying each augmentation transform",
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
        default=DEFAULT_HYBRID_STRATEGY_DIR,
        help="Destination directory for fine-tuned weights",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train_phoneme_ipa_hybrid(args)


if __name__ == "__main__":
    main()
