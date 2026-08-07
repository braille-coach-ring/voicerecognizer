"""
Plot Saving Utility with Versioning and Timestamped History

【設計理由: グラフのバージョン管理・比較可能化】
・`plots/{model_name}/history/{timestamp}_loss.png` および `accuracy.png` へタイムスタンプ付きで無期限保存。
・一目で直近の結果を確認できるよう `plots/{model_name}/loss_latest.png` および `accuracy_latest.png` へ出力。
・後方互換性のため、指定された旧パス (PROJECT_ROOT/loss.png 等) にも自動保存。
"""

import datetime
import logging
from pathlib import Path
import shutil
import matplotlib.pyplot as plt

from config import PROJECT_ROOT

logger = logging.getLogger(__name__)


def save_history_plots(
    history: dict[str, list[float]],
    model_name: str = "cnn",
    legacy_loss_path: Path | None = None,
    legacy_accuracy_path: Path | None = None,
) -> tuple[Path, Path]:
    """
    Loss および Accuracy/Macro-F1 グラフを最新 (latest) および履歴 (history/YYYYMMDD_HHMMSS) に出力保存します。
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    model_plot_dir = PROJECT_ROOT / "plots" / model_name
    history_dir = model_plot_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    # 履歴ファイルパス
    hist_loss_path = history_dir / f"{timestamp}_loss.png"
    hist_acc_path = history_dir / f"{timestamp}_accuracy.png"

    # 最新ファイルパス
    latest_loss_path = model_plot_dir / "loss_latest.png"
    latest_acc_path = model_plot_dir / "accuracy_latest.png"

    # --- Loss Plot ---
    plt.figure(figsize=(8, 5))
    if "train_loss" in history:
        plt.plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name.upper()} Training Loss ({timestamp})")
    plt.legend()
    plt.grid(True)

    plt.savefig(hist_loss_path, bbox_inches="tight")
    plt.savefig(latest_loss_path, bbox_inches="tight")
    if legacy_loss_path:
        try:
            legacy_loss_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(latest_loss_path, legacy_loss_path)
        except Exception as e:
            logger.warning("旧ロスグラフパスへの保存に失敗しました: %s", e)
    plt.close()

    # --- Accuracy / Macro-F1 Plot ---
    plt.figure(figsize=(8, 5))
    if "train_acc" in history:
        plt.plot(history["train_acc"], label="Train Acc")
    if "val_acc" in history:
        plt.plot(history["val_acc"], label="Val Acc")
    if "val_macro_f1" in history:
        plt.plot(history["val_macro_f1"], label="Val Macro-F1", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"{model_name.upper()} Validation Metrics ({timestamp})")
    plt.legend()
    plt.grid(True)

    plt.savefig(hist_acc_path, bbox_inches="tight")
    plt.savefig(latest_acc_path, bbox_inches="tight")
    if legacy_accuracy_path:
        try:
            legacy_accuracy_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(latest_acc_path, legacy_accuracy_path)
        except Exception as e:
            logger.warning("旧精度グラフパスへの保存に失敗しました: %s", e)
    plt.close()

    logger.info(
        "学習グラフを保存しました:\n  [最新] %s\n  [最新] %s\n  [履歴] %s",
        latest_loss_path,
        latest_acc_path,
        history_dir,
    )
    return latest_loss_path, latest_acc_path
