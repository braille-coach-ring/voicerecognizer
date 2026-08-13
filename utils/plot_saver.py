"""
Plot Saving Utility with Versioning and Timestamped History

【設計理由: グラフのバージョン管理・比較可能化】
・`plots/{model_name}/history/{timestamp}_loss.png` および `accuracy.png` へタイムスタンプ付きで無期限保存。
・一目で直近の結果を確認できるよう `plots/{model_name}/loss_latest.png` および `accuracy_latest.png` へ出力。
・後方互換性のため、指定された旧パス (PROJECT_ROOT/loss.png 等) にも自動保存。
"""

import datetime
import json
import logging
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import numpy as np

from config import PROJECT_ROOT

logger = logging.getLogger(__name__)


def record_and_plot_cumulative_progress(
    model_name: str,
    val_acc: float,
    val_macro_f1: float,
    val_weighted_f1: float = 0.0,
    val_loss: float = 0.0,
    epochs: int = 1,
    num_classes: int = 105,
    num_samples: int = 0,
    is_best_updated: bool | None = None,
) -> Path:
    """
    これまでの全学習ランの成績推移 (Val Acc, Val Macro-F1, Val Weighted-F1, Val Loss) を
    `experiment_history.json` に累積記録し、Best Model 更新点 (★) を明示した自己改善トレンドグラフ
    `long_term_progress_trend.png` を生成・更新します。
    """
    model_plot_dir = PROJECT_ROOT / "plots" / model_name
    model_plot_dir.mkdir(parents=True, exist_ok=True)
    history_json_path = model_plot_dir / "experiment_history.json"
    trend_plot_path = model_plot_dir / "long_term_progress_trend.png"

    records = []
    if history_json_path.exists():
        try:
            with open(history_json_path, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = []

    # Best Model 更新の自動判定 (指定がない場合は過去最高 Macro-F1 の更新有無)
    prev_max_f1 = max([r.get("val_macro_f1", 0.0) for r in records], default=0.0)
    if is_best_updated is None:
        is_best_updated = (val_macro_f1 > prev_max_f1) or len(records) == 0

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_record = {
        "run_id": len(records) + 1,
        "timestamp": timestamp,
        "epochs": epochs,
        "num_classes": num_classes,
        "num_samples": num_samples,
        "val_acc": round(float(val_acc), 4),
        "val_macro_f1": round(float(val_macro_f1), 4),
        "val_weighted_f1": round(float(val_weighted_f1), 4),
        "val_loss": round(float(val_loss), 4),
        "is_best_updated": bool(is_best_updated),
    }
    records.append(new_record)

    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    run_ids = [r["run_id"] for r in records]
    accs = [r["val_acc"] for r in records]
    f1s = [r["val_macro_f1"] for r in records]
    wf1s = [r.get("val_weighted_f1", 0.0) for r in records]
    losses = [r.get("val_loss", 0.0) for r in records]
    samples = [r.get("num_samples", 0) for r in records]

    # 過去最高スコアのステップ曲線 (Cumulative Peak Curve)
    running_peak_f1 = np.maximum.accumulate(f1s)
    ax1.step(
        run_ids,
        running_peak_f1,
        where="post",
        color="gold",
        linewidth=2.5,
        label="Cumulative Best Macro-F1 Peak",
        zorder=2,
    )

    # 上段: 4指標の長期的推移 (Acc, Macro-F1, Weighted-F1, Val Loss)
    ax1.plot(
        run_ids, accs, marker="o", color="navy", linewidth=1.5, alpha=0.7, label="Val Accuracy"
    )
    ax1.plot(
        run_ids,
        f1s,
        marker="s",
        color="darkorange",
        linewidth=2.0,
        linestyle="--",
        label="Val Macro-F1",
    )
    ax1.plot(
        run_ids,
        wf1s,
        marker="^",
        color="forestgreen",
        linewidth=1.5,
        linestyle=":",
        alpha=0.7,
        label="Val Weighted-F1",
    )

    # Best Model 更新ランのゴールドスター (★) ハイライト
    best_runs = [r for r in records if r.get("is_best_updated", False)]
    if best_runs:
        best_x = [r["run_id"] for r in best_runs]
        best_y = [r["val_macro_f1"] for r in best_runs]
        ax1.scatter(
            best_x,
            best_y,
            marker="*",
            s=260,
            color="gold",
            edgecolors="darkred",
            linewidth=1.5,
            zorder=6,
            label="Best Model Updated (★)",
        )
        for r in best_runs:
            ax1.annotate(
                f"★ Run #{r['run_id']}\n({r['val_macro_f1']:.4f})",
                xy=(r["run_id"], r["val_macro_f1"]),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                fontweight="bold",
                color="darkred",
                bbox=dict(
                    boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.3, edgecolor="goldenrod"
                ),
            )

    ax1.set_ylabel("Metric Score (0.0 - 1.0)", color="navy")
    ax1.set_title(f"{model_name.upper()} Self-Improvement Loop & Best Model Progress Trend")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.08)

    # ロス（右Y軸）
    ax1_loss = ax1.twinx()
    ax1_loss.plot(
        run_ids,
        losses,
        marker="d",
        color="crimson",
        linewidth=1.2,
        linestyle="-.",
        alpha=0.6,
        label="Val Loss",
    )
    ax1_loss.set_ylabel("Validation Loss", color="crimson")

    # 凡例統合
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_loss.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    # 下段: データセット成長
    ax2.bar(run_ids, samples, color="steelblue", alpha=0.5, label="Total Audio Samples")
    ax2.set_xlabel("Training Run #")
    ax2.set_ylabel("Sample Count")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(trend_plot_path, bbox_inches="tight")
    plt.close()

    logger.info(
        "自己改善ベストモデル更新ハイライト付きトレンドグラフを保存しました: %s", trend_plot_path
    )
    return trend_plot_path


def save_history_plots(
    history: dict[str, list[float]],
    model_name: str = "cnn",
    legacy_loss_path: Path | None = None,
    legacy_accuracy_path: Path | None = None,
    num_classes: int = 105,
    num_samples: int = 0,
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
        plt.plot(history["train_acc"], label="Train Acc", color="steelblue", linewidth=1.5)
    if "val_acc" in history:
        plt.plot(history["val_acc"], label="Val Acc", color="navy", linewidth=2.0)
    if "val_macro_f1" in history:
        plt.plot(
            history["val_macro_f1"],
            label="Val Macro-F1",
            color="darkorange",
            linestyle="--",
            linewidth=2.0,
        )
    if "val_weighted_f1" in history:
        plt.plot(
            history["val_weighted_f1"],
            label="Val Weighted-F1",
            color="forestgreen",
            linestyle=":",
            linewidth=2.0,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Score (0.0 - 1.0)")
    plt.title(f"{model_name.upper()} Training & Validation Metrics ({timestamp})")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)

    plt.savefig(hist_acc_path, bbox_inches="tight")
    plt.savefig(latest_acc_path, bbox_inches="tight")
    if legacy_accuracy_path:
        try:
            legacy_accuracy_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(latest_acc_path, legacy_accuracy_path)
        except Exception as e:
            logger.warning("旧精度グラフパスへの保存に失敗しました: %s", e)
    plt.close()

    # 累積トータル progress の記録・更新
    last_val_acc = history.get("val_acc", [0.0])[-1] if history.get("val_acc") else 0.0
    last_val_f1 = history.get("val_macro_f1", [0.0])[-1] if history.get("val_macro_f1") else 0.0
    last_val_wf1 = (
        history.get("val_weighted_f1", [0.0])[-1] if history.get("val_weighted_f1") else 0.0
    )
    last_val_loss = history.get("val_loss", [0.0])[-1] if history.get("val_loss") else 0.0
    epochs_count = len(history.get("train_loss", []))

    record_and_plot_cumulative_progress(
        model_name=model_name,
        val_acc=last_val_acc,
        val_macro_f1=last_val_f1,
        val_weighted_f1=last_val_wf1,
        val_loss=last_val_loss,
        epochs=epochs_count,
        num_classes=num_classes,
        num_samples=num_samples,
    )

    logger.info(
        "学習グラフを保存しました:\n  [最新] %s\n  [最新] %s\n  [履歴] %s",
        latest_loss_path,
        latest_acc_path,
        history_dir,
    )
    return latest_loss_path, latest_acc_path
