"""
Wav2Vec2 Fine-Tuning Script with Layer Freezing and On-Memory Caching

役割:
  Wav2Vec2 プリトレイニードモデルの下位層フリーズ ＋ オンメモリキャッシュにより、
  転移学習ファインチューニングを高速実行し、best_model ディレクトリおよび labels.json を保存します。

使い方:
  uv run python models/wav2vec2/train.py           # デフォルト設定で Wav2Vec2 を継続ファインチューニング
  uv run python models/wav2vec2/train.py --no-resume # ベースモデル (facebook/wav2vec2-base) から新規学習
"""

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

        logger.info("オンメモリキャッシュ作成中: %d 件の音声データを前処理しています...", len(self.data))
        self.cached_data: list[tuple[np.ndarray, int]] = []
        for wav_path, label in self.data:
            waveform = self.preprocessor.preprocess_waveform(wav_path)
            self.cached_data.append((waveform, label))

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        return self.cached_data[index]


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


from utils.split_helper import safe_stratified_split


def split_dataset(
    dataset: Wav2Vec2ClassificationDataset, val_rate: float, seed: int
) -> tuple[Subset, Subset]:
    labels = [label for _, label in dataset.data]
    train_idx, val_idx = safe_stratified_split(labels, val_rate=val_rate, seed=seed)
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


from utils.plot_saver import save_history_plots


def save_plots(
    history: dict[str, list[float]], loss_path: Path, accuracy_path: Path
) -> None:
    save_history_plots(
        history=history,
        model_name="wav2vec2",
        legacy_loss_path=loss_path,
        legacy_accuracy_path=accuracy_path,
    )


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


def freeze_wav2vec2_layers(
    model: Any,
    freeze_feature_encoder: bool = True,
    freeze_transformer_layers: int = 10,
) -> None:
    """
    Wav2Vec2 の下位レイヤーをフリーズ（勾配計算対象外化）して学習を高速化・過学習防止する。

    ・Feature Encoder (7層CNN): 常にフリーズ
    ・Transformer Encoder (12層のうち下位 10 層): 勾配計算をオフ (requires_grad = False)
    ・最上位 2 層 + Classifier Head: 勾配計算を継続 (requires_grad = True)
    """
    if freeze_feature_encoder and hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
        logger.info("Wav2Vec2 feature encoder (7-layer CNN) is frozen.")

    if freeze_transformer_layers > 0 and hasattr(model, "wav2vec2") and hasattr(model.wav2vec2, "encoder"):
        layers = model.wav2vec2.encoder.layers
        num_total_layers = len(layers)
        num_frozen = min(freeze_transformer_layers, num_total_layers)
        for i in range(num_frozen):
            for param in layers[i].parameters():
                param.requires_grad = False
        logger.info(
            "Wav2Vec2 Transformer 下位 %d 層 (全 %d 層中) をフリーズしました。上位 %d 層と分類ヘッドのみ学習します。",
            num_frozen,
            num_total_layers,
            num_total_layers - num_frozen,
        )


from preprocessing.dataset_builder import DatasetBuilder, ensure_merged_and_preprocessed


def train(args: argparse.Namespace) -> None:
    ensure_merged_and_preprocessed(skip_prep=getattr(args, "skip_prep", False))
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

    has_weights = (
        args.best_model_path.exists()
        and (
            (args.best_model_path / "model.safetensors").exists()
            or (args.best_model_path / "pytorch_model.bin").exists()
        )
    )

    model_source = args.pretrained_model_name
    if args.resume and has_weights:
        model_source = str(args.best_model_path)
        logger.info("既存のチェックポイント (%s) を再利用 (reuse) して継続学習を行います。", model_source)
    elif not args.resume:
        logger.info(
            "=== [--from-scratch / --no-resume が指定されたため、既存チェックポイントを読み込まずベースモデル (%s) から新規ファインチューニングを開始します] ===",
            model_source,
        )
    elif args.resume:
        logger.info(
            "過去のチェックポイント (%s) が存在しないため、ベースモデル (%s) から新規学習を開始します。",
            args.best_model_path,
            model_source,
        )

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_source)
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
        model_source,
        num_labels=len(dataset.labels),
        label2id=label2id,
        id2label=id2label,
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True,
    )
    freeze_wav2vec2_layers(
        model,
        freeze_feature_encoder=args.freeze_feature_encoder,
        freeze_transformer_layers=args.freeze_transformer_layers,
    )
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
    best_macro_f1 = -1.0

    if has_weights:
        try:
            eval_fe = AutoFeatureExtractor.from_pretrained(str(args.best_model_path))
            eval_collate = build_collate_fn(eval_fe, args.sample_rate)
            eval_val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=eval_collate,
            )
            eval_m = Wav2Vec2ForSequenceClassification.from_pretrained(
                str(args.best_model_path),
                num_labels=len(dataset.labels),
                label2id=label2id,
                id2label=id2label,
                problem_type="single_label_classification",
                ignore_mismatched_sizes=True,
            ).to(device)
            _, val_acc, val_true, val_pred = validate(
                eval_m, eval_val_loader, device, labels=dataset.labels
            )
            init_result = compute_evaluation_result(
                val_true, val_pred, labels=dataset.labels
            )
            best_macro_f1 = init_result.overall.macro_f1
            logger.info(
                "保存済み Wav2Vec2 ベストモデル (%s) の評価スコア - Val Acc: %.4f, Val Macro-F1: %.4f",
                args.best_model_path,
                val_acc,
                best_macro_f1,
            )
            del eval_m
            del eval_fe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning("既存 Wav2Vec2 モデルの評価に失敗しました: %s", e)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
    }
    is_best_updated = False

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
            is_best_updated = True
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
    save_history_plots(
        history=history,
        model_name="wav2vec2",
        legacy_loss_path=args.loss_plot_path,
        legacy_accuracy_path=args.accuracy_plot_path,
        num_classes=len(dataset.labels),
        num_samples=len(dataset),
    )
    logger.info(
        "Training finished. Last Wav2Vec2 model saved to %s",
        args.last_model_path,
    )

    # Hugging Face 自動アップロード判定 (ベストモデルが更新された場合のみ)
    from config import DEFAULT_HUGGINGFACE_CONFIG
    if getattr(args, "upload_hf", False) or DEFAULT_HUGGINGFACE_CONFIG.auto_upload:
        if is_best_updated:
            from utils.model_uploader import upload_weights_to_hf
            upload_weights_to_hf(model_type="wav2vec2")
        else:
            logger.info("ベストモデルが更新されなかったため、Hugging Face へのアップロードをスキップします。")



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
    parser.add_argument(
        "--freeze-feature-encoder",
        action="store_true",
        default=True,
        help="Freeze Wav2Vec2 feature encoder layers for faster training (default: True)",
    )
    parser.add_argument(
        "--no-freeze-feature-encoder",
        action="store_false",
        dest="freeze_feature_encoder",
        help="Unfreeze Wav2Vec2 feature encoder layers",
    )
    parser.add_argument(
        "--freeze-transformer-layers",
        type=int,
        default=10,
        help="Number of bottom Transformer layers to freeze (default: 10 out of 12)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Reuse existing trained model checkpoint by default if available (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        "--from-scratch",
        action="store_false",
        dest="resume",
        help="Train from base pretrained model without reusing existing checkpoint",
    )
    parser.add_argument(
        "--skip-prep",
        action="store_true",
        help="Skip automatic data merging (merge_data) and preprocessing before training (default: False, auto-prep is enabled by default)",
    )
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
