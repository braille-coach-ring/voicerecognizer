import csv
from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import sys

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_RECOGNITION_CONFIG
from core.interfaces import RecognitionStrategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PerClassMetrics:
    """文字（クラス）ごとの評価指標"""

    precision: float
    recall: float
    f1_score: float
    support: int


@dataclass(frozen=True)
class OverallMetrics:
    """全体評価指標"""

    accuracy: float
    macro_f1: float
    weighted_f1: float
    total_samples: int


@dataclass(frozen=True)
class MisclassifiedSample:
    """誤識別した音声サンプル"""

    true_label: str
    predicted_label: str
    filepath: str = ""
    confidence: float | None = None


@dataclass
class EvaluationResult:
    """評価レポート全体の統合オブジェクト"""

    overall: OverallMetrics
    per_class: dict[str, PerClassMetrics]
    confusion_matrix: dict[str, dict[str, int]]
    misclassified: list[MisclassifiedSample] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON保存用・Dict型変換"""
        return asdict(self)


class Evaluator:

    def __init__(
        self,
        model: RecognitionStrategy | None = None,
        labels: tuple[str, ...] = DEFAULT_RECOGNITION_CONFIG.labels,
        dataset_path: Path
        | str = DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir,
    ):
        """Evaluatorの初期化

        Args:
            model: RecognitionStrategyの実装モデル（推論なし集計時はNone可）
            labels: 評価対象ラベルのタプル
            dataset_path: データセットのルートディレクトリ
        """
        self.model = model
        self.labels = labels
        self.dataset_path = Path(dataset_path)
        self.index_file = self.dataset_path / "index.csv"

        if not self.dataset_path.exists():
            logger.warning(
                f"データセットパス {self.dataset_path} が存在しません"
            )
            raise FileNotFoundError(
                f"データセットパス {self.dataset_path} が存在しません"
            )
        if not self.index_file.exists():
            logger.warning(
                f"インデックスファイル {self.index_file} が存在しません"
            )
            raise FileNotFoundError(
                f"インデックスファイル {self.index_file} が存在しません"
            )

        self.y_true: list[str] = []
        self.y_pred: list[str] = []
        self.confidences: list[float] = []
        self.filepaths: list[str] = []
        self.result: EvaluationResult | None = None

    def reset(self) -> None:
        """評価用内部状態のリセット"""
        self.y_true.clear()
        self.y_pred.clear()
        self.confidences.clear()
        self.filepaths.clear()
        self.result = None

    def _add_prediction(
        self,
        true_label: str,
        pred_label: str,
        confidence: float | None = None,
        filepath: str | None = None,
    ) -> None:
        """1件ごとの推論/評価レコードを追加する共通処理"""
        self.y_true.append(true_label)
        self.y_pred.append(pred_label)
        if confidence is not None:
            self.confidences.append(confidence)
        if filepath is not None:
            self.filepaths.append(filepath)

    def _resolve_audio_path(self, rel_path: str) -> Path:
        p = Path(rel_path)
        if p.is_absolute() and p.exists():
            return p
        if (self.dataset_path / rel_path).exists():
            return self.dataset_path / rel_path
        if (PROJECT_ROOT / rel_path).exists():
            return PROJECT_ROOT / rel_path
        return self.dataset_path / rel_path

    def evaluate(self) -> EvaluationResult:
        """最新モデルで全データセットに対してリアルタイム推論を実行し、評価する"""
        if self.model is None:
            logger.warning("モデルがロードされていません")
            raise ValueError("モデルがロードされていません")

        if not self.index_file.exists():
            logger.warning(
                f"インデックスファイルが存在しません: {self.index_file}"
            )
            raise FileNotFoundError(
                f"インデックスファイルが存在しません: {self.index_file}"
            )

        self.reset()

        with open(self.index_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_path = row["filepath"]
                true_label = str(row["label"])
                audio_path = self._resolve_audio_path(rel_path)

                if not audio_path.exists():
                    logger.warning(
                        f"音声ファイルが存在しません: {audio_path}"
                    )
                    continue

                pred_label = self.model.recognize(str(audio_path))
                confidence = getattr(self.model, "last_confidence", None)

                self._add_prediction(
                    true_label=true_label,
                    pred_label=pred_label,
                    confidence=confidence,
                    filepath=rel_path,
                )

        self.result = self._compute_metrics()
        return self.result

    def update_index_with_predictions(self) -> None:
        if self.model is None:
            logger.warning("繝｢繝・Ν縺後Ο繝ｼ繝峨＆繧後※縺・∪縺帙ｓ")
            raise ValueError("繝｢繝・Ν縺後Ο繝ｼ繝峨＆繧後※縺・∪縺帙ｓ")

        rows: list[dict[str, str]] = []
        with open(self.index_file, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            if "predicted_text" not in fieldnames:
                fieldnames.append("predicted_text")

            for row in reader:
                rel_path = row.get("filepath", "")
                audio_path = self._resolve_audio_path(rel_path)

                if not audio_path.exists():
                    logger.warning(
                        f"髻ｳ螢ｰ繝輔ぃ繧､繝ｫ縺悟ｭ伜惠縺励∪縺帙ｓ: {audio_path}"
                    )
                    row["predicted_text"] = ""
                else:
                    row["predicted_text"] = self.model.recognize(str(audio_path))
                rows.append(row)

        with open(self.index_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def update_from_dataset(self) -> EvaluationResult:
        """データセット(index.csv)内の predicted_text を用いて再推論を行わずに即座に評価・集計する"""
        if not self.index_file.exists():
            logger.warning(
                f"インデックスファイルが存在しません: {self.index_file}"
            )
            raise FileNotFoundError(
                f"インデックスファイルが存在しません: {self.index_file}"
            )

        self.reset()

        with open(self.index_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "predicted_text" not in row or not row["predicted_text"]:
                    continue

                true_label = str(row["label"])
                pred_label = str(row["predicted_text"])
                rel_path = row.get("filepath", "")

                self._add_prediction(
                    true_label=true_label,
                    pred_label=pred_label,
                    filepath=rel_path,
                )

        self.result = self._compute_metrics()
        return self.result

    def _compute_metrics(self) -> EvaluationResult:
        """蓄積された y_true / y_pred から EvaluationResult を計算・構築する"""
        if not self.y_true:
            logger.warning("評価対象のデータが存在しません")
            return EvaluationResult(
                overall=OverallMetrics(
                    accuracy=0.0, macro_f1=0.0, weighted_f1=0.0, total_samples=0
                ),
                per_class={},
                confusion_matrix={},
                misclassified=[],
            )

        labels_list = list(self.labels)
        acc = float(accuracy_score(self.y_true, self.y_pred))
        report_dict = classification_report(
            self.y_true,
            self.y_pred,
            labels=labels_list,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(
            self.y_true,
            self.y_pred,
            labels=labels_list,
        )

        confusion_breakdown = {
            true_lbl: {
                pred_lbl: int(cm[i, j])
                for j, pred_lbl in enumerate(labels_list)
            }
            for i, true_lbl in enumerate(labels_list)
        }

        per_class = {
            lbl: PerClassMetrics(
                precision=round(float(report_dict[lbl]["precision"]), 4),
                recall=round(float(report_dict[lbl]["recall"]), 4),
                f1_score=round(float(report_dict[lbl]["f1-score"]), 4),
                support=int(report_dict[lbl]["support"]),
            )
            for lbl in labels_list
            if lbl in report_dict
        }

        overall = OverallMetrics(
            accuracy=round(acc, 4),
            macro_f1=round(float(report_dict["macro avg"]["f1-score"]), 4),
            weighted_f1=round(
                float(report_dict["weighted avg"]["f1-score"]), 4
            ),
            total_samples=len(self.y_true),
        )

        misclassified = []
        for i in range(len(self.y_true)):
            if self.y_true[i] != self.y_pred[i]:
                filepath = self.filepaths[i] if i < len(self.filepaths) else ""
                confidence = (
                    self.confidences[i] if i < len(self.confidences) else None
                )
                misclassified.append(
                    MisclassifiedSample(
                        true_label=self.y_true[i],
                        predicted_label=self.y_pred[i],
                        filepath=filepath,
                        confidence=confidence,
                    )
                )

        return EvaluationResult(
            overall=overall,
            per_class=per_class,
            confusion_matrix=confusion_breakdown,
            misclassified=misclassified,
        )

    def export_json(self, output_path: Path | str) -> bool:
        """評価結果(JSON)をファイルに出力する"""
        if self.result is None:
            logger.warning(
                "評価結果が存在しません。先に evaluate() または update_from_dataset() を実行してください。"
            )
            return False

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.result.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"評価結果JSONを保存しました: {output_path}")
        return True


def compute_evaluation_result(
    y_true: list[str],
    y_pred: list[str],
    labels: tuple[str, ...] | list[str] = DEFAULT_RECOGNITION_CONFIG.labels,
    filepaths: list[str] | None = None,
    confidences: list[float] | None = None,
) -> EvaluationResult:
    """スタンドアロンで y_true と y_pred から EvaluationResult を直接計算するユーティリティ関数"""
    evaluator = object.__new__(Evaluator)
    evaluator.labels = labels
    evaluator.y_true = list(y_true)
    evaluator.y_pred = list(y_pred)
    evaluator.filepaths = list(filepaths) if filepaths else []
    evaluator.confidences = list(confidences) if confidences else []
    evaluator.result = None
    return evaluator._compute_metrics()
