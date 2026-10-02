"""
Train / Fine-tune HiraganaCNN on the combined dataset including the new speakerphone data.
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from voicerecognizer.config import DEFAULT_AUDIO_CONFIG, DEFAULT_PREPROCESS_CONFIG, PROJECT_ROOT
from voicerecognizer.dataset.hiragana_dataset import HiraganaDataset
from voicerecognizer.evaluation.evaluator import Evaluator, compute_evaluation_result
from voicerecognizer.models.cnn.hiragana_cnn import HiraganaCNN
from voicerecognizer.recognizers.cnn_recognizer import CNNRecognizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("speakerphone_adaptation")


def fix_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, labels in loader:
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

    return total_loss / max(total, 1), correct / max(total, 1)


def validate(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    labels: list[str],
) -> tuple[float, float, float]:
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

    eval_result = compute_evaluation_result(all_true, all_pred, labels=labels)
    acc = correct / max(total, 1)
    return total_loss / max(total, 1), acc, eval_result.overall.macro_f1


def main():
    seed = 42
    fix_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    splits_dir = PROJECT_ROOT / "data_splits"
    train_csv = splits_dir / "combined_train.csv"
    val_csv = splits_dir / "combined_val.csv"
    test_eval_dir = splits_dir / "speakerphone_test_eval"

    weights_dir = PROJECT_ROOT / "weights"
    initial_weights = weights_dir / "best_model.pth"
    adapted_weights_path = weights_dir / "speakerphone_adapted_model.pth"
    best_weights_path = weights_dir / "best_model.pth"

    labels_json_path = weights_dir / "labels.json"
    with open(labels_json_path, encoding="utf-8") as f:
        master_labels = json.load(f)

    logger.info("Loading training dataset (combined_train: 5260 samples)...")
    train_dataset = HiraganaDataset(
        root_dir=train_csv,
        sample_rate=DEFAULT_AUDIO_CONFIG.sample_rate,
        n_mels=DEFAULT_PREPROCESS_CONFIG.n_mels,
        cache_in_memory=True,
    )

    logger.info("Loading validation dataset (combined_val: 1290 samples)...")
    val_dataset = HiraganaDataset(
        root_dir=val_csv,
        sample_rate=DEFAULT_AUDIO_CONFIG.sample_rate,
        n_mels=DEFAULT_PREPROCESS_CONFIG.n_mels,
        cache_in_memory=True,
    )

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(master_labels)
    model = HiraganaCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    # 既存の事前学習済み重みをロード
    if initial_weights.exists():
        logger.info("Loading existing pretrained weights from %s", initial_weights)
        state_dict = torch.load(initial_weights, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        logger.warning("No pretrained weights found at %s. Training from scratch.", initial_weights)

    # 初期スコア測定
    init_loss, init_acc, init_macro_f1 = validate(
        model, val_loader, criterion, device, train_dataset.labels
    )
    logger.info(
        "Pre-training Val Evaluation - Loss: %.4f, Acc: %.4f, Macro-F1: %.4f",
        init_loss,
        init_acc,
        init_macro_f1,
    )

    # ファインチューニング用設定
    learning_rate = 0.0003
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 25
    patience = 7
    patience_counter = 0
    best_val_macro_f1 = init_macro_f1

    logger.info("Starting fine-tuning for %d epochs...", epochs)
    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_macro_f1 = validate(
            model, val_loader, criterion, device, train_dataset.labels
        )
        elapsed = time.time() - t0

        logger.info(
            "Epoch %02d/%d [%.1fs] - Train Loss: %.4f, Train Acc: %.4f | Val Loss: %.4f, Val Acc: %.4f, Val F1: %.4f",
            epoch + 1,
            epochs,
            elapsed,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_macro_f1,
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            patience_counter = 0
            torch.save(model.state_dict(), adapted_weights_path)
            torch.save(model.state_dict(), best_weights_path)
            # キャッシュも更新
            cache_weight = Path.home() / ".cache" / "voicerecognizer" / "weights" / "best_model.pth"
            if cache_weight.parent.exists():
                torch.save(model.state_dict(), cache_weight)
            logger.info("  ★ New Best Model saved! (Val Macro-F1: %.4f)", best_val_macro_f1)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", epoch + 1)
                break

    logger.info("=== Fine-tuning completed! Best Val Macro-F1: %.4f ===", best_val_macro_f1)

    # 最終評価: 未見のスピーカーホン Test セット (156件) で適応性能を検証
    logger.info("=== Evaluating on unseen Speakerphone Test Set (156 samples) ===")
    test_recognizer = CNNRecognizer(model_path=best_weights_path)
    evaluator = Evaluator(
        model=test_recognizer,
        dataset_path=test_eval_dir,
    )
    test_result = evaluator.evaluate()

    logger.info("--- Speakerphone Test Set Final Results ---")
    logger.info("Accuracy    : %.4f (Before was: 0.0705)", test_result.overall.accuracy)
    logger.info("Macro F1    : %.4f (Before was: 0.0371)", test_result.overall.macro_f1)
    logger.info("Weighted F1 : %.4f (Before was: 0.0436)", test_result.overall.weighted_f1)
    logger.info("Total       : %d samples", test_result.overall.total_samples)

    evaluator.export_json(PROJECT_ROOT / "evaluation_results" / "speakerphone_test_after.json")
    evaluator.export_html(
        PROJECT_ROOT / "evaluation_results" / "speakerphone_test_after.html",
        title="Speakerphone Adaptation Test Report (After Training)",
    )
    logger.info("Report saved to evaluation_results/speakerphone_test_after.html and .json")

    # 未見の女性話者 Test セット (52件) の評価
    fem_test_dir = splits_dir / "female_test_eval"
    if fem_test_dir.exists():
        logger.info("=== Evaluating on unseen Female Test Set (52 samples) ===")
        fem_evaluator = Evaluator(
            model=test_recognizer,
            dataset_path=fem_test_dir,
        )
        fem_result = fem_evaluator.evaluate()
        logger.info("--- Female Test Set Final Results ---")
        logger.info("Accuracy    : %.4f (Before was: 0.0865)", fem_result.overall.accuracy)
        logger.info("Macro F1    : %.4f", fem_result.overall.macro_f1)
        logger.info("Total       : %d samples", fem_result.overall.total_samples)

        fem_evaluator.export_json(PROJECT_ROOT / "evaluation_results" / "female_test_after.json")
        fem_evaluator.export_html(
            PROJECT_ROOT / "evaluation_results" / "female_test_after.html",
            title="Female Voice Adaptation Test Report (After Training)",
        )
        logger.info("Report saved to evaluation_results/female_test_after.html and .json")


if __name__ == "__main__":
    main()
