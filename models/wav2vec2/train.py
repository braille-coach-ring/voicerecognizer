import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from config import DEFAULT_RECOGNITION_CONFIG
from dataset.hiragana_dataset import HiraganaDataset
from evaluation.evaluator import compute_evaluation_result
from preprocessing.audio_preprocessor import AudioPreprocessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class Wav2Vec2ClassificationDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        sample_rate: int,
        target_length_seconds: float,
        top_db: float,
    ):
        source_dataset = HiraganaDataset(
            root_dir=root_dir,
            sample_rate=sample_rate,
        )
        self.labels = source_dataset.labels
        self.data = source_dataset.data
        self.preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            target_length_seconds=target_length_seconds,
            top_db=top_db,
        )

        if not self.data:
            raise ValueError(f"No training samples were found in {root_dir}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        wav_path, label = self.data[index]
        waveform = self.preprocessor.preprocess_waveform(wav_path)
        return waveform, label


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(
    dataset: Wav2Vec2ClassificationDataset, val_rate: float, seed: int
) -> tuple[Subset, Subset]:
    labels = [label for _, label in dataset.data]
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_rate,
        random_state=seed,
    )
    train_idx, val_idx = next(splitter.split(range(len(labels)), labels))
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def build_collate_fn(feature_extractor: Any, sample_rate: int) -> Callable:
    def collate(batch: list[tuple[np.ndarray, int]]) -> dict[str, torch.Tensor]:
        waveforms = [waveform for waveform, _ in batch]
        labels = torch.tensor([label for _, label in batch], dtype=torch.long)
        inputs = feature_extractor(
            waveforms,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs["labels"] = labels
        return inputs

    return collate


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    epoch: int,
    epochs: int,
    max_grad_norm: float,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    progress = tqdm(loader)

    for batch in progress:
        batch = move_batch(batch, device)

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        logits = outputs.logits.detach()
        pred = logits.argmax(dim=-1)
        labels = batch["labels"]
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()

        progress.set_description(f"Wav2Vec2 Epoch {epoch + 1}/{epochs}")
        progress.set_postfix(loss=f"{loss.item():.3f}")

    return total_loss / max(len(loader), 1), correct / max(total, 1)


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    labels: tuple[str, ...] | list[str],
) -> tuple[float, float, list[str], list[str]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_true: list[str] = []
    all_pred: list[str] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            pred = outputs.logits.argmax(dim=-1)
            batch_labels = batch["labels"]

            total_loss += loss.item()
            correct += (pred == batch_labels).sum().item()
            total += batch_labels.size(0)

            for true_index, pred_index in zip(
                batch_labels.cpu().numpy(), pred.cpu().numpy()
            ):
                all_true.append(labels[int(true_index)])
                all_pred.append(labels[int(pred_index)])

    return total_loss / max(len(loader), 1), correct / max(total, 1), all_true, all_pred


def save_plots(
    history: dict[str, list[float]], loss_path: Path, accuracy_path: Path
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_path)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.plot(history["val_macro_f1"], label="Val Macro-F1", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(accuracy_path)
    plt.close()


def save_pretrained_model(
    model: Any,
    feature_extractor: Any,
    output_dir: Path,
    labels: tuple[str, ...] | list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    feature_extractor.save_pretrained(output_dir)
    (output_dir / "labels.json").write_text(
        json.dumps(list(labels), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def import_transformers() -> tuple[Any, Any, Any]:
    try:
        from transformers import (
            AutoFeatureExtractor,
            Wav2Vec2ForSequenceClassification,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:
        raise ImportError(
            "Wav2Vec2 training requires the 'transformers' package. "
            "Install project dependencies with: uv sync"
        ) from exc

    return (
        AutoFeatureExtractor,
        Wav2Vec2ForSequenceClassification,
        get_linear_schedule_with_warmup,
    )


def train(args: argparse.Namespace) -> None:
    fix_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        AutoFeatureExtractor,
        Wav2Vec2ForSequenceClassification,
        get_linear_schedule_with_warmup,
    ) = import_transformers()

    dataset = Wav2Vec2ClassificationDataset(
        root_dir=args.root_dir,
        sample_rate=args.sample_rate,
        target_length_seconds=args.target_length_seconds,
        top_db=args.top_db,
    )
    train_dataset, val_dataset = split_dataset(dataset, args.val_rate, args.seed)

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.pretrained_model_name
    )
    feature_sample_rate = getattr(feature_extractor, "sampling_rate", None)
    if feature_sample_rate and feature_sample_rate != args.sample_rate:
        logger.warning(
            "Feature extractor expects %s Hz but training uses %s Hz",
            feature_sample_rate,
            args.sample_rate,
        )

    collate_fn = build_collate_fn(feature_extractor, args.sample_rate)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    label2id = {label: index for index, label in enumerate(dataset.labels)}
    id2label = {index: label for label, index in label2id.items()}
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        args.pretrained_model_name,
        num_labels=len(dataset.labels),
        label2id=label2id,
        id2label=id2label,
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True,
    )
    if args.freeze_feature_encoder and hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
        logger.info("Wav2Vec2 feature encoder is frozen.")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    training_steps = max(len(train_loader) * args.epochs, 1)
    warmup_steps = int(training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=training_steps,
    )
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
    }
    best_macro_f1 = -1.0

    logger.info("Device: %s", device)
    logger.info("Labels: %s", dataset.labels)
    logger.info("Train: %d, Validation: %d", len(train_dataset), len(val_dataset))

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch,
            args.epochs,
            args.max_grad_norm,
        )
        val_loss, val_acc, val_true, val_pred = validate(
            model, val_loader, device, labels=dataset.labels
        )
        eval_result = compute_evaluation_result(
            val_true, val_pred, labels=dataset.labels
        )
        macro_f1 = eval_result.overall.macro_f1

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(macro_f1)

        logger.info(
            "Epoch %d/%d - Train Loss: %.4f, Train Acc: %.4f | "
            "Val Loss: %.4f, Val Acc: %.4f, Val Macro-F1: %.4f",
            epoch + 1,
            args.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            macro_f1,
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            save_pretrained_model(
                model,
                feature_extractor,
                args.best_model_path,
                dataset.labels,
            )
            logger.info(
                "Best Wav2Vec2 model saved: %s (Val Macro-F1: %.4f, Val Acc: %.4f)",
                args.best_model_path,
                best_macro_f1,
                val_acc,
            )

        if val_acc >= args.target_acc:
            logger.info(
                "Target validation accuracy reached: %.2f%%",
                args.target_acc * 100,
            )
            break

    save_pretrained_model(
        model,
        feature_extractor,
        args.last_model_path,
        dataset.labels,
    )
    save_plots(history, args.loss_plot_path, args.accuracy_plot_path)
    logger.info(
        "Training finished. Last Wav2Vec2 model saved to %s",
        args.last_model_path,
    )


def build_parser() -> argparse.ArgumentParser:
    default_root = (
        DEFAULT_RECOGNITION_CONFIG.processed_dataset_dir
        if DEFAULT_RECOGNITION_CONFIG.processed_dataset_dir.exists()
        else DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir
    )
    parser = argparse.ArgumentParser(
        description="Fine-tune Wav2Vec2 for hiragana label classification."
    )
    parser.add_argument("--root-dir", type=Path, default=default_root)
    parser.add_argument(
        "--pretrained-model-name",
        default=DEFAULT_RECOGNITION_CONFIG.wav2vec2_pretrained_model_name,
    )
    parser.add_argument(
        "--best-model-path",
        type=Path,
        default=DEFAULT_RECOGNITION_CONFIG.wav2vec2_best_model_dir,
    )
    parser.add_argument(
        "--last-model-path",
        type=Path,
        default=DEFAULT_RECOGNITION_CONFIG.wav2vec2_last_model_dir,
    )
    parser.add_argument(
        "--loss-plot-path",
        type=Path,
        default=DEFAULT_RECOGNITION_CONFIG.wav2vec2_loss_plot_path,
    )
    parser.add_argument(
        "--accuracy-plot-path",
        type=Path,
        default=DEFAULT_RECOGNITION_CONFIG.wav2vec2_accuracy_plot_path,
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_RECOGNITION_CONFIG.sample_rate,
    )
    parser.add_argument(
        "--target-length-seconds",
        type=float,
        default=DEFAULT_RECOGNITION_CONFIG.target_length_seconds,
    )
    parser.add_argument(
        "--top-db",
        type=float,
        default=DEFAULT_RECOGNITION_CONFIG.top_db,
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-rate", type=float, default=0.2)
    parser.add_argument("--target-acc", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--freeze-feature-encoder", action="store_true")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
