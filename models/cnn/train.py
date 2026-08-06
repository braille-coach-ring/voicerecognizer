"""
Hiragana CNN Model Training Script

【設計理由: 学習パイプラインの構成】
・`processed_dataset/` (または `merged_dataset/index.csv`) から前処理済みの音声・メルスペクトログラムをロード。
・Stratified 80/20 分割によりクラス割合を均等に保持しながら Train / Validation データセットを作成。
・CrossEntropyLoss および Adam オプティマイザで CNN モデルを自動学習し、最高精度の重み (best_model.pth)
  および最新の重み (last_model.pth)、Loss/Accuracy グラフを出力保存します。
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import (
    DEFAULT_AUDIO_CONFIG,
    DEFAULT_PREPROCESS_CONFIG,
    DEFAULT_RECOGNITION_CONFIG,
)
from dataset.hiragana_dataset import HiraganaDataset
from preprocessing.dataset_builder import DatasetBuilder, ensure_merged_and_preprocessed
from models.cnn.hiragana_cnn import HiraganaCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


from utils.split_helper import safe_stratified_split


def split_dataset(dataset: HiraganaDataset, val_rate: float, seed: int):
    labels = [label for _, label in dataset.data]
    train_idx, val_idx = safe_stratified_split(labels, val_rate=val_rate, seed=seed)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def train_epoch(model, loader, criterion, optimizer, device, epoch: int, epochs: int):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    progress = tqdm(loader)

    for mel, label in progress:
        mel = mel.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(mel)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.size(0)

        progress.set_description(f"Epoch {epoch + 1}/{epochs}")
        progress.set_postfix(loss=f"{loss.item():.3f}")

    return total_loss / len(loader), correct / total


from evaluation.evaluator import compute_evaluation_result


def validate(model, loader, criterion, device, labels: tuple[str, ...]):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_true: list[str] = []
    all_pred: list[str] = []

    with torch.no_grad():
        for mel, label in loader:
            mel = mel.to(device)
            label = label.to(device)
            output = model(mel)
            loss = criterion(output, label)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

            for t, p in zip(label.cpu().numpy(), pred.cpu().numpy()):
                all_true.append(labels[t])
                all_pred.append(labels[p])

    return total_loss / len(loader), correct / total, all_true, all_pred


from utils.plot_saver import save_history_plots


def save_plots(
    history: dict[str, list[float]], loss_path: Path, accuracy_path: Path
) -> None:
    save_history_plots(
        history=history,
        model_name="cnn",
        legacy_loss_path=loss_path,
        legacy_accuracy_path=accuracy_path,
    )


def train(args: argparse.Namespace) -> None:
    ensure_merged_and_preprocessed(skip_prep=getattr(args, "skip_prep", False))
    fix_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HiraganaDataset(
        root_dir=args.root_dir,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
    )
    train_dataset, val_dataset = split_dataset(dataset, args.val_rate, args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info("Device: %s", device)
    logger.info("Train: %d, Validation: %d", len(train_dataset), len(val_dataset))

    model = HiraganaCNN(num_classes=len(dataset.labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_macro_f1": []}
    best_macro_f1 = 0.0

    if args.resume and args.best_model_path.exists():
        try:
            state_dict = torch.load(args.best_model_path, map_location=device, weights_only=True)
            model_dict = model.state_dict()
            matched_dict = {
                k: v for k, v in state_dict.items()
                if k in model_dict and model_dict[k].shape == v.shape
            }
            model_dict.update(matched_dict)
            model.load_state_dict(model_dict)

            # Evaluate loaded model score if full weights matched
            if len(matched_dict) == len(state_dict):
                eval_model = HiraganaCNN(num_classes=len(dataset.labels)).to(device)
                eval_model.load_state_dict(state_dict)
                _, val_acc, val_true, val_pred = validate(
                    eval_model, val_loader, criterion, device, labels=dataset.labels
                )
                init_result = compute_evaluation_result(val_true, val_pred, labels=dataset.labels)
                best_macro_f1 = init_result.overall.macro_f1
                logger.info("既存モデル重み (%s) を再利用 (reuse) して継続学習します - Val Acc: %.4f, Val Macro-F1: %.4f", args.best_model_path, val_acc, best_macro_f1)
            else:
                logger.info("既存モデル (%s) の特徴抽出層重みを再利用 (reuse) し、分類ヘッドを更新して継続学習を開始します。", args.best_model_path)
        except Exception as e:
            logger.warning("既存モデル %s の再利用に失敗したため新規学習を行います (%s)。", args.best_model_path, e)
    elif not args.resume:
        logger.info("=== [--from-scratch / --no-resume が指定されたため、既存重みを破棄して 0 から新規学習を開始します] ===")
    else:
        logger.info("過去のチェックポイント (%s) が存在しないため、0 から新規学習を開始します。", args.best_model_path)

    args.best_model_path.parent.mkdir(parents=True, exist_ok=True)
    args.last_model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            args.epochs,
        )
        val_loss, val_acc, val_true, val_pred = validate(
            model, val_loader, criterion, device, labels=dataset.labels
        )
        eval_result = compute_evaluation_result(val_true, val_pred, labels=dataset.labels)
        macro_f1 = eval_result.overall.macro_f1

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(macro_f1)

        logger.info(
            "Epoch %d/%d - Train Loss: %.4f, Train Acc: %.4f | Val Loss: %.4f, Val Acc: %.4f, Val Macro-F1: %.4f",
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
            torch.save(model.state_dict(), args.best_model_path)
            labels_path = args.best_model_path.parent / "labels.json"
            with open(labels_path, "w", encoding="utf-8") as f:
                json.dump(list(dataset.labels), f, ensure_ascii=False, indent=2)
            logger.info(
                "Best model saved: %s (Val Macro-F1: %.4f, Val Acc: %.4f)",
                args.best_model_path,
                best_macro_f1,
                val_acc,
            )

        if val_acc >= args.target_acc:
            logger.info("Target validation accuracy reached: %.2f%%", args.target_acc * 100)
            break

    torch.save(model.state_dict(), args.last_model_path)
    save_plots(history, args.loss_plot_path, args.accuracy_plot_path)
    logger.info("Training finished. Last model saved to %s", args.last_model_path)


def build_parser() -> argparse.ArgumentParser:
    default_root = (
        DEFAULT_RECOGNITION_CONFIG.processed_dataset_dir
        if DEFAULT_RECOGNITION_CONFIG.processed_dataset_dir.exists()
        else DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir
    )
    parser = argparse.ArgumentParser(description="Train the Hiragana CNN recognizer.")
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=default_root,
    )
    parser.add_argument(
        "--sample-rate", type=int, default=DEFAULT_AUDIO_CONFIG.sample_rate
    )
    parser.add_argument("--n-mels", type=int, default=DEFAULT_PREPROCESS_CONFIG.n_mels)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--val-rate", type=float, default=0.2)
    parser.add_argument("--target-acc", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--best-model-path",
        type=Path,
        default=DEFAULT_RECOGNITION_CONFIG.cnn_weight_path,
    )
    parser.add_argument(
        "--last-model-path",
        type=Path,
        default=DEFAULT_RECOGNITION_CONFIG.last_model_path,
    )
    parser.add_argument(
        "--loss-plot-path", type=Path, default=DEFAULT_RECOGNITION_CONFIG.loss_plot_path
    )
    parser.add_argument(
        "--accuracy-plot-path",
        type=Path,
        default=DEFAULT_RECOGNITION_CONFIG.accuracy_plot_path,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Reuse existing trained model weights by default if available (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        "--from-scratch",
        action="store_false",
        dest="resume",
        help="Train from scratch without reusing existing model weights",
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
