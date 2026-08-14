"""
Wav2Vec2 Base Model A/B Test Script

役割:
  異なるベースモデル × フリーズ層数の組み合わせを短エポック (5ep) で比較し、
  このプロジェクトに最適なベースモデルとフリーズ設定を実データで検証する。

検証条件:
  A: facebook/wav2vec2-base       freeze=6  (上位6層学習)
  B: facebook/wav2vec2-base       freeze=10 (上位2層学習, 現行設定)
  C: reazon-research/japanese-wav2vec2-base  freeze=6

特徴:
  - Ctrl+C で安全に中断可能（完了済みの条件は結果に残る）
  - 各条件の結果を1条件ごとに即座にJSONに保存（中断しても途中結果が残る）
  - weights/ を汚さない（一時ディレクトリに保存）

使い方:
  uv run python script/ab_test_base_models.py
  uv run python script/ab_test_base_models.py --epochs 3       # エポック数を変更
"""

import argparse
import contextlib
import gc
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from voicerecognizer.config import (
    DEFAULT_AUDIO_CONFIG,
    DEFAULT_PREPROCESS_CONFIG,
    DEFAULT_RECOGNITION_CONFIG,
    PROJECT_ROOT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# A/B テスト条件の定義
DEFAULT_CONDITIONS = [
    {
        "name": "A: facebook/wav2vec2-base (freeze=6)",
        "model_name": "facebook/wav2vec2-base",
        "freeze_transformer_layers": 6,
    },
    {
        "name": "B: facebook/wav2vec2-base (freeze=10, current)",
        "model_name": "facebook/wav2vec2-base",
        "freeze_transformer_layers": 10,
    },
    {
        "name": "C: reazon-research/japanese-wav2vec2-base (freeze=6)",
        "model_name": "reazon-research/japanese-wav2vec2-base",
        "freeze_transformer_layers": 6,
    },
]


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_condition(
    condition: dict,
    epochs: int,
    seed: int,
    device: torch.device,
    learning_rate: float,
    batch_size: int,
    output_dir: Path,
) -> dict[str, Any]:
    """1つの条件（ベースモデル + フリーズ設定）でファインチューニングを実行する"""
    from transformers import (
        AutoConfig,
        AutoFeatureExtractor,
        Wav2Vec2ForSequenceClassification,
        get_linear_schedule_with_warmup,
    )

    from voicerecognizer.evaluation.evaluator import compute_evaluation_result
    from voicerecognizer.models.wav2vec2.train import (
        AugmentedSubset,
        Wav2Vec2ClassificationDataset,
        build_collate_fn,
        compute_class_weights,
        determine_optimal_num_workers,
        freeze_wav2vec2_layers,
        load_wav2vec2_classifier,
        split_dataset,
        train_epoch,
        validate,
    )
    from voicerecognizer.preprocessing.audio_augmentor import AudioAugmentor

    fix_seed(seed)

    cond_name = condition["name"]
    model_name = condition["model_name"]
    freeze_layers = condition["freeze_transformer_layers"]

    logger.info("=" * 70)
    logger.info("  [%s]", cond_name)
    logger.info(
        "  Model: %s | Freeze: %d layers | LR: %s | Epochs: %d",
        model_name,
        freeze_layers,
        learning_rate,
        epochs,
    )
    logger.info("=" * 70)

    # Dataset
    root_dir = (
        DEFAULT_RECOGNITION_CONFIG.processed_dataset_dir
        if DEFAULT_RECOGNITION_CONFIG.processed_dataset_dir.exists()
        else DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir
    )
    dataset = Wav2Vec2ClassificationDataset(
        root_dir=root_dir,
        sample_rate=DEFAULT_AUDIO_CONFIG.sample_rate,
        target_length_seconds=DEFAULT_RECOGNITION_CONFIG.target_length_seconds,
        top_db=DEFAULT_PREPROCESS_CONFIG.top_db,
    )
    train_subset, val_subset = split_dataset(dataset, val_rate=0.2, seed=seed)

    augmentor = AudioAugmentor()
    train_dataset = AugmentedSubset(train_subset, augmentor)

    # Class weights
    train_labels = [dataset.data[i][1] for i in train_subset.indices]
    class_weights = compute_class_weights(
        train_labels, num_classes=len(dataset.labels), power=0.5
    ).to(device)
    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Model
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    collate_fn = build_collate_fn(feature_extractor, DEFAULT_AUDIO_CONFIG.sample_rate)

    label2id = {label: idx for idx, label in enumerate(dataset.labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    model = load_wav2vec2_classifier(
        Wav2Vec2ForSequenceClassification,
        AutoConfig,
        model_name,
        dataset.labels,
        label2id,
        id2label,
    )
    freeze_wav2vec2_layers(
        model,
        freeze_feature_encoder=True,
        freeze_transformer_layers=freeze_layers,
    )
    model.to(device)

    # Count trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable params: %s / %s (%.1f%%)",
        f"{trainable_params:,}",
        f"{total_params:,}",
        trainable_params / total_params * 100,
    )

    # DataLoaders & Hardware Optimization (CUDA vs CPU dynamic settings)
    num_workers = determine_optimal_num_workers() if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0

    logger.info(
        "Hardware Optimization: device=%s | num_workers=%d | pin_memory=%s | AMP=%s",
        device,
        num_workers,
        pin_memory,
        (device.type == "cuda"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    training_steps = max(len(train_loader) * epochs, 1)
    warmup_steps = int(training_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=training_steps,
    )
    scaler = GradScaler() if device.type == "cuda" else None

    # Training loop
    history = []
    best_val_macro_f1 = -1.0
    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch,
            epochs,
            max_grad_norm=1.0,
            loss_fct=loss_fct,
            scaler=scaler,
        )

        val_loss, val_acc, val_true, val_pred = validate(
            model,
            val_loader,
            device,
            labels=dataset.labels,
        )
        eval_result = compute_evaluation_result(val_true, val_pred, labels=dataset.labels)
        macro_f1 = eval_result.overall.macro_f1

        if macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = macro_f1

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_macro_f1": round(macro_f1, 4),
        }
        history.append(epoch_result)

        logger.info(
            "[%s] Epoch %d/%d - Train Acc: %.4f | Val Acc: %.4f | Val F1: %.4f",
            cond_name,
            epoch + 1,
            epochs,
            train_acc,
            val_acc,
            macro_f1,
        )

    elapsed = time.time() - start_time

    # Cleanup
    del model, optimizer, scheduler, feature_extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "condition": cond_name,
        "model_name": condition["model_name"],
        "freeze_transformer_layers": freeze_layers,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "best_val_macro_f1": round(best_val_macro_f1, 4),
        "final_val_acc": history[-1]["val_acc"] if history else 0,
        "final_val_macro_f1": history[-1]["val_macro_f1"] if history else 0,
        "elapsed_seconds": round(elapsed, 1),
        "history": history,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="A/B test for Wav2Vec2 base model and freeze layer selection"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per condition (default: 5)")
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate (default: 2e-5)"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "evaluation_results" / "ab_test_results.json",
        help="Output path for A/B test results",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        with contextlib.suppress(Exception):
            sys.stdout.reconfigure(encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    # 既に途中結果があれば読み込む（中断後の再開用）
    if args.output_json.exists():
        try:
            existing = json.loads(args.output_json.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                all_results = existing
                completed_names = {r["condition"] for r in all_results}
                logger.info("Resuming: %d conditions already completed.", len(all_results))
        except Exception:
            pass

    completed_names = {r["condition"] for r in all_results}

    print("\n" + "=" * 70)
    print("  Wav2Vec2 A/B Test: Base Model x Freeze Layer Comparison")
    print(f"  Epochs: {args.epochs} | LR: {args.learning_rate} | Batch: {args.batch_size}")
    print("  Press Ctrl+C to safely interrupt (completed conditions are saved)")
    print("=" * 70 + "\n")

    for _i, condition in enumerate(DEFAULT_CONDITIONS):
        if condition["name"] in completed_names:
            logger.info("Skipping already completed: %s", condition["name"])
            continue

        try:
            result = run_single_condition(
                condition=condition,
                epochs=args.epochs,
                seed=args.seed,
                device=device,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                output_dir=args.output_json.parent,
            )
            all_results.append(result)

            # 各条件完了ごとに即座に保存（中断しても途中結果が残る）
            args.output_json.write_text(
                json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.info(
                "Result saved to %s (%d/%d conditions done)",
                args.output_json,
                len(all_results),
                len(DEFAULT_CONDITIONS),
            )

        except KeyboardInterrupt:
            logger.warning("Ctrl+C detected. Saving completed results and exiting.")
            args.output_json.write_text(
                json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"\nSaved {len(all_results)} completed condition(s) to {args.output_json}")
            print("Re-run the same command to resume from where you left off.")
            return

    # Final summary
    successful = [r for r in all_results if "final_val_acc" in r]
    successful.sort(key=lambda x: x["best_val_macro_f1"], reverse=True)

    print("\n" + "=" * 90)
    print("  A/B Test Final Summary (sorted by Best Val Macro-F1)")
    print("=" * 90)
    header = f"{'Rank':<5} {'Condition':<55} {'Best F1':<10} {'Final Acc':<10} {'Time':<8}"
    print(header)
    print("-" * 90)
    for rank, r in enumerate(successful, 1):
        print(
            f"{rank:<5} {r['condition']:<55} {r['best_val_macro_f1']:<10.4f} {r['final_val_acc'] * 100:<9.2f}% {r['elapsed_seconds']:<8.1f}s"
        )

    # Epoch-by-epoch comparison
    print("\n" + "-" * 90)
    print("  Epoch-by-Epoch Val Macro-F1 Comparison")
    print("-" * 90)
    max_epochs = max(len(r.get("history", [])) for r in successful) if successful else 0
    header_parts = [f"{'Epoch':<7}"]
    for r in successful:
        short_name = r["condition"][:30]
        header_parts.append(f"{short_name:<32}")
    print("".join(header_parts))

    for ep in range(max_epochs):
        parts = [f"  {ep + 1:<5}"]
        for r in successful:
            hist = r.get("history", [])
            if ep < len(hist):
                f1 = hist[ep]["val_macro_f1"]
                acc = hist[ep]["val_acc"]
                parts.append(f"F1={f1:.4f} Acc={acc:.4f}          ")
            else:
                parts.append(f"{'N/A':<32}")
        print("".join(parts))

    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
