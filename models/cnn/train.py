"""
Hiragana CNN Model Training Script

役割:
  前処理済みデータセット (processed_dataset/ や index.csv) から Stratified 分割により
  CNN モデルの学習・検証を行い、best_model.pth および labels.json を保存します。

使い方:
  uv run python models/cnn/train.py                # デフォルト設定で CNN を継続学習 (チーム最新モデルを自動取得)
  uv run python models/cnn/train.py --from-scratch  # 0 から新規学習
"""

import argparse
import json
import logging
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import (
    DEFAULT_AUDIO_CONFIG,
    DEFAULT_PREPROCESS_CONFIG,
    DEFAULT_RECOGNITION_CONFIG,
)
from dataset.hiragana_dataset import HiraganaDataset
from evaluation.evaluator import compute_evaluation_result
from models.cnn.hiragana_cnn import HiraganaCNN
from preprocessing.dataset_builder import ensure_merged_and_preprocessed
from utils.plot_saver import save_history_plots

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def fix_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(
    dataset: HiraganaDataset, val_rate: float = 0.2, seed: int = 42
) -> tuple[Subset, Subset]:
    from utils.split_helper import safe_stratified_split

    labels = [label for _, label in dataset.data]
    train_idx, val_idx = safe_stratified_split(labels, val_rate=val_rate, seed=seed)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
    for data, labels in progress:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

        progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")

    return total_loss / max(total, 1), correct / max(total, 1)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
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
        for data, target_labels in loader:
            data, target_labels = data.to(device), target_labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, target_labels)
            total_loss += loss.item() * data.size(0)

            _, pred = outputs.max(1)
            correct += pred.eq(target_labels).sum().item()
            total += target_labels.size(0)

            for t_idx, p_idx in zip(target_labels.cpu().numpy(), pred.cpu().numpy(), strict=False):
                all_true.append(labels[int(t_idx)])
                all_pred.append(labels[int(p_idx)])

    return total_loss / max(total, 1), correct / max(total, 1), all_true, all_pred


def train(args: argparse.Namespace) -> None:
    ensure_merged_and_preprocessed(skip_prep=getattr(args, "skip_prep", False))

    root_dir = getattr(args, "root_dir", None) or (
        DEFAULT_RECOGNITION_CONFIG.processed_dataset_dir
        if DEFAULT_RECOGNITION_CONFIG.processed_dataset_dir.exists()
        else DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir
    )
    sample_rate = getattr(args, "sample_rate", DEFAULT_AUDIO_CONFIG.sample_rate)
    n_mels = getattr(args, "n_mels", DEFAULT_PREPROCESS_CONFIG.n_mels)
    val_rate = getattr(args, "val_rate", 0.2)
    seed = getattr(args, "seed", 42)
    best_model_path = getattr(args, "best_model_path", DEFAULT_RECOGNITION_CONFIG.cnn_weight_path)
    last_model_path = getattr(args, "last_model_path", DEFAULT_RECOGNITION_CONFIG.last_model_path)
    target_acc = getattr(args, "target_acc", 0.97)
    resume = getattr(args, "resume", True)

    # チーム共有の最新モデルを手元と比較し、必要に応じて自動ダウンロード (SHA256事前判定)
    if resume:
        from utils.model_uploader import download_latest_team_weights_if_needed

        download_latest_team_weights_if_needed(model_type="cnn")

    fix_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HiraganaDataset(
        root_dir=root_dir,
        sample_rate=sample_rate,
        n_mels=n_mels,
    )
    train_dataset, val_dataset = split_dataset(dataset, val_rate, seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info("Device: %s", device)
    logger.info("Train: %d, Validation: %d", len(train_dataset), len(val_dataset))

    model = HiraganaCNN(num_classes=len(dataset.labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_macro_f1": []}
    best_macro_f1 = 0.0
    is_best_updated = False

    if resume and best_model_path and best_model_path.exists():
        try:
            state_dict = torch.load(best_model_path, map_location=device, weights_only=True)
            model_dict = model.state_dict()
            matched_dict = {
                k: v
                for k, v in state_dict.items()
                if k in model_dict and model_dict[k].shape == v.shape
            }
            model_dict.update(matched_dict)
            model.load_state_dict(model_dict)

            # Evaluate loaded model score if full weights matched
            if len(matched_dict) == len(state_dict):
                eval_model = HiraganaCNN(num_classes=len(dataset.labels)).to(device)
                eval_model.load_state_dict(state_dict)
                _, val_acc_init, val_true_init, val_pred_init = validate(
                    eval_model, val_loader, criterion, device, labels=dataset.labels
                )
                init_result = compute_evaluation_result(
                    val_true_init, val_pred_init, labels=dataset.labels
                )
                best_macro_f1 = init_result.overall.macro_f1
                logger.info(
                    "既存のチェックポイント (%s) を再利用して継続学習を行います。", best_model_path
                )
                logger.info(
                    "保存済みベストモデルの評価スコア - Val Acc: %.4f, Val Macro-F1: %.4f",
                    val_acc_init,
                    best_macro_f1,
                )
        except Exception as e:
            logger.warning("既存チェックポイントの読み込み/評価に失敗しました: %s", e)
    elif not resume:
        logger.info(
            "=== [--from-scratch が指定されたため、既存重みを破棄して 0 から新規学習を開始します] ==="
        )

    patience = getattr(args, "patience", 10)
    patience_counter = 0

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
            is_best_updated = True
            patience_counter = 0
            if best_model_path:
                torch.save(model.state_dict(), best_model_path)
                labels_path = best_model_path.parent / "labels.json"
                with open(labels_path, "w", encoding="utf-8") as f:
                    json.dump(list(dataset.labels), f, ensure_ascii=False, indent=2)
                logger.info(
                    "Best model saved: %s (Val Macro-F1: %.4f, Val Acc: %.4f)",
                    best_model_path,
                    best_macro_f1,
                    val_acc,
                )
        else:
            patience_counter += 1
            if patience > 0 and patience_counter >= patience:
                logger.info(
                    "🛑 Early stopping: Validation Macro-F1 が %d エポック連続で向上しなかったため、頭打ちと判断して学習を自動終了します (Best Macro-F1: %.4f)",
                    patience,
                    best_macro_f1,
                )
                break

        if val_acc >= target_acc:
            logger.info("Target validation accuracy reached: %.2f%%", target_acc * 100)
            break

    if last_model_path:
        torch.save(model.state_dict(), last_model_path)
    save_history_plots(
        history=history,
        model_name="cnn",
        num_classes=len(dataset.labels),
        num_samples=len(dataset),
    )
    logger.info("Training finished. Last model saved to %s", last_model_path)

    # Hugging Face 自動アップロード判定 (チーム最高精度を更新した場合のみ)
    if is_best_updated:
        from utils.model_uploader import upload_weights_to_hf

        upload_weights_to_hf(model_type="cnn")
    else:
        logger.info(
            "チーム最高精度が更新されなかったため、Hugging Face へのアップロードをスキップします。"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Hiragana CNN recognizer.")
    parser.add_argument(
        "--epochs", type=int, default=150, help="Number of training epochs (default: 150)"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of epochs with no validation improvement after which training stops early (default: 10, set 0 to disable)",
    )
    parser.add_argument(
        "--from-scratch",
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Train from scratch without reusing team/local checkpoint",
    )

    parser.add_argument(
        "--skip-prep",
        action="store_true",
        help="Skip automatic data merging and preprocessing before training",
    )
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
