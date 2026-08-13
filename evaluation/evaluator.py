import csv
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_RECOGNITION_CONFIG  # noqa: E402
from core.interfaces import RecognitionStrategy  # noqa: E402

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
        dataset_path: Path | str = DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir,
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
            logger.warning(f"データセットパス {self.dataset_path} が存在しません")
            raise FileNotFoundError(f"データセットパス {self.dataset_path} が存在しません")
        if not self.index_file.exists():
            logger.warning(f"インデックスファイル {self.index_file} が存在しません")
            raise FileNotFoundError(f"インデックスファイル {self.index_file} が存在しません")

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
            logger.warning(f"インデックスファイルが存在しません: {self.index_file}")
            raise FileNotFoundError(f"インデックスファイルが存在しません: {self.index_file}")

        self.reset()

        with open(self.index_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_path = row["filepath"]
                true_label = str(row["label"])
                audio_path = self._resolve_audio_path(rel_path)

                if not audio_path.exists():
                    logger.warning(f"音声ファイルが存在しません: {audio_path}")
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
        with open(self.index_file, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            if "predicted_text" not in fieldnames:
                fieldnames.append("predicted_text")

            for row in reader:
                rel_path = row.get("filepath", "")
                audio_path = self._resolve_audio_path(rel_path)

                if not audio_path.exists():
                    logger.warning(f"髻ｳ螢ｰ繝輔ぃ繧､繝ｫ縺悟ｭ伜惠縺励∪縺帙ｓ: {audio_path}")
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
            logger.warning(f"インデックスファイルが存在しません: {self.index_file}")
            raise FileNotFoundError(f"インデックスファイルが存在しません: {self.index_file}")

        self.reset()

        with open(self.index_file, encoding="utf-8") as f:
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
            true_lbl: {pred_lbl: int(cm[i, j]) for j, pred_lbl in enumerate(labels_list)}
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
            weighted_f1=round(float(report_dict["weighted avg"]["f1-score"]), 4),
            total_samples=len(self.y_true),
        )

        misclassified = []
        for i in range(len(self.y_true)):
            if self.y_true[i] != self.y_pred[i]:
                filepath = self.filepaths[i] if i < len(self.filepaths) else ""
                confidence = self.confidences[i] if i < len(self.confidences) else None
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

    def export_html(
        self, output_path: Path | str, title: str = "音声認識モデル評価レポート"
    ) -> bool:
        """評価結果(HTMLレポート)をファイルに出力する"""
        if self.result is None:
            logger.warning(
                "評価結果が存在しません。先に evaluate() または update_from_dataset() を実行してください。"
            )
            return False

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html_content = generate_html_report(self.result, title=title)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"評価結果HTMLレポートを保存しました: {output_path}")
        return True


def generate_html_report(
    result: EvaluationResult, title: str = "音声認識モデル評価レポート"
) -> str:
    """
    EvaluationResult から、人間（エンジニア・開発者）が直感的に分析・改善アクションを起こせる
    リッチな HTML インタラクティブ・ダッシュボードレポートを生成する
    """
    overall = result.overall
    per_class = result.per_class
    cm = result.confusion_matrix
    misclassified = result.misclassified

    labels = list(per_class.keys())

    # --- 1. 自動診断・要改善ポイントのアクション抽出 (Executive Automated Insights) ---
    insights = []

    # 混同対（誤認識ペア）の集計
    confusion_pairs = []
    for true_lbl in labels:
        row_total = sum(cm.get(true_lbl, {}).values())
        for pred_lbl in labels:
            if true_lbl != pred_lbl:
                cnt = cm.get(true_lbl, {}).get(pred_lbl, 0)
                if cnt > 0:
                    pct = (cnt / row_total * 100) if row_total > 0 else 0
                    confusion_pairs.append((true_lbl, pred_lbl, cnt, pct))

    confusion_pairs.sort(key=lambda x: x[2], reverse=True)

    if confusion_pairs:
        top_pair = confusion_pairs[0]
        insights.append(
            {
                "type": "danger",
                "icon": "🚨",
                "title": f"最重点ボトルネック: 「{top_pair[0]}」 ➔ 「{top_pair[1]}」 の誤認識",
                "desc": f"正解が「{top_pair[0]}」のサンプルのうち {top_pair[2]} 件 ({top_pair[3]:.1f}%) が「{top_pair[1]}」と誤認されています。音量歪み（クリッピング）や『{top_pair[0]}』の生声サンプルの追加収集を推奨します。",
            }
        )

    # サンプル数の偏りチェック
    supports = [m.support for m in per_class.values() if m.support > 0]
    if supports:
        avg_supp = sum(supports) / len(supports)
        for lbl, m in per_class.items():
            if m.support < avg_supp * 0.3 and m.support > 0:
                insights.append(
                    {
                        "type": "warning",
                        "icon": "⚠️",
                        "title": f"データ偏りの警告: ラベル 「{lbl}」 のサンプル不足",
                        "desc": f"「{lbl}」のデータ数（{m.support}件）が全平均（{avg_supp:.0f}件）に対して著しく少ないため、誤判定や過学習のリスクがあります。",
                    }
                )

    # 高精度達成の称賛
    high_acc_labels = [lbl for lbl, m in per_class.items() if m.f1_score >= 0.95 and m.support >= 5]
    if high_acc_labels:
        insights.append(
            {
                "type": "success",
                "icon": "🎉",
                "title": f"優秀クラス: 「{', '.join(high_acc_labels)}」 は F1 95% 以上を維持",
                "desc": "これらの音階・音声特徴表現は安定して正しく学習されています。",
            }
        )

    insights_html = ""
    for ins in insights:
        insights_html += f"""
        <div class="insight-card insight-{ins["type"]}">
            <div class="insight-icon">{ins["icon"]}</div>
            <div class="insight-content">
                <div class="insight-title">{ins["title"]}</div>
                <div class="insight-desc">{ins["desc"]}</div>
            </div>
        </div>
        """

    # --- 2. 混同行列のパーセンテージ＆ヒートマップ表示 ---
    cm_headers_html = "".join([f"<th>予測: {lbl}</th>" for lbl in labels])
    cm_rows_html = ""
    for true_lbl in labels:
        row_total = sum(cm.get(true_lbl, {}).values())
        row_cells = f"<td class='row-label'>正解: {true_lbl}</td>"
        for pred_lbl in labels:
            count = cm.get(true_lbl, {}).get(pred_lbl, 0)
            pct = (count / row_total * 100) if row_total > 0 else 0.0

            if true_lbl == pred_lbl:
                alpha = min(1.0, max(0.15, pct / 100.0))
                cell_style = f"background-color: rgba(52, 211, 153, {alpha:.2f}); color: #ffffff;"
            else:
                if count > 0:
                    alpha = min(1.0, max(0.2, pct / 40.0))
                    cell_style = (
                        f"background-color: rgba(248, 113, 113, {alpha:.2f}); color: #ffffff;"
                    )
                else:
                    cell_style = "color: var(--text-muted); opacity: 0.3;"

            row_cells += f"<td style='{cell_style}' class='cm-cell'><strong>{count}</strong><br><span style='font-size:0.75rem;'>({pct:.1f}%)</span></td>"
        cm_rows_html += f"<tr>{row_cells}</tr>"

    # --- 3. クラス別メトリクス行 ＆ Chart.js データ構築 ---
    per_class_rows_html = ""
    chart_labels_js = json.dumps(labels, ensure_ascii=False)
    chart_f1_js = json.dumps([round(per_class[lbl].f1_score * 100, 1) for lbl in labels])
    chart_prec_js = json.dumps([round(per_class[lbl].precision * 100, 1) for lbl in labels])
    chart_rec_js = json.dumps([round(per_class[lbl].recall * 100, 1) for lbl in labels])

    for lbl, m in per_class.items():
        f1_pct = m.f1_score * 100
        prec_pct = m.precision * 100
        rec_pct = m.recall * 100
        per_class_rows_html += f"""
        <tr>
            <td class='class-name'><strong>{lbl}</strong></td>
            <td>{m.precision:.4f} ({prec_pct:.1f}%)</td>
            <td>{m.recall:.4f} ({rec_pct:.1f}%)</td>
            <td>
                <div class="progress-container">
                    <div class="progress-bar" style="width: {f1_pct:.1f}%;"></div>
                    <span class="progress-text">{m.f1_score:.4f}</span>
                </div>
            </td>
            <td>{m.support} 件</td>
        </tr>
        """

    # --- 4. 誤識別サンプル行 ＆ インライン HTML5 音声再生プレイヤー ---
    filter_buttons_html = f"<button class='btn-filter active' onclick='filterCategory(\"all\")'>全件 ({len(misclassified)})</button>"

    # 誤認識のある正解ラベルのユニークリスト
    mis_labels = sorted({m.true_label for m in misclassified})
    for ml in mis_labels:
        cnt = sum(1 for m in misclassified if m.true_label == ml)
        filter_buttons_html += f"<button class='btn-filter' onclick='filterCategory(\"{ml}\")'>正解「{ml}」 ({cnt})</button>"

    mis_rows_html = ""
    if misclassified:
        for i, sample in enumerate(misclassified[:150], 1):
            conf_str = f"{sample.confidence:.2f}" if sample.confidence is not None else "-"
            rel_audio = sample.filepath.replace("\\", "/")

            # ブラウザから相対パスで .wav を直接再生できるインラインプレーヤー
            audio_player_html = f"""
            <audio controls preload="none" style="height: 30px; width: 220px;">
                <source src="../{rel_audio}" type="audio/wav">
                <source src="../../{rel_audio}" type="audio/wav">
                <source src="{rel_audio}" type="audio/wav">
                お使いのブラウザは音声再生に対応していません。
            </audio>
            """

            mis_rows_html += f"""
            <tr class="mis-row category-{sample.true_label}">
                <td>{i}</td>
                <td><span class="badge badge-true">正解: {sample.true_label}</span></td>
                <td><span class="badge badge-pred">予測: {sample.predicted_label}</span></td>
                <td>{audio_player_html}</td>
                <td class="filepath">{rel_audio}</td>
                <td>{conf_str}</td>
            </tr>
            """
    else:
        mis_rows_html = "<tr><td colspan='6' style='text-align:center; color:#34d399; padding:20px;'>🎉 誤識別サンプルはありません（全件完全正解）！</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-green: #34d399;
            --accent-red: #f87171;
            --border-color: #334155;
        }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 30px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1240px;
            margin: 0 auto;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        h1 {{
            margin: 0;
            font-size: 1.8rem;
            color: var(--accent-blue);
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .kpi-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
        }}
        .kpi-title {{
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .kpi-value {{
            font-size: 2.2rem;
            font-weight: 700;
            margin: 10px 0 0 0;
            color: var(--accent-green);
        }}

        /* インサイトカード */
        .insights-section {{
            margin-bottom: 30px;
        }}
        .insight-card {{
            display: flex;
            align-items: flex-start;
            gap: 15px;
            padding: 16px 20px;
            border-radius: 10px;
            margin-bottom: 12px;
            border: 1px solid transparent;
        }}
        .insight-danger {{
            background-color: rgba(248, 113, 113, 0.1);
            border-color: rgba(248, 113, 113, 0.3);
        }}
        .insight-warning {{
            background-color: rgba(251, 191, 36, 0.1);
            border-color: rgba(251, 191, 36, 0.3);
        }}
        .insight-success {{
            background-color: rgba(52, 211, 153, 0.1);
            border-color: rgba(52, 211, 153, 0.3);
        }}
        .insight-icon {{
            font-size: 1.5rem;
            line-height: 1;
        }}
        .insight-title {{
            font-weight: bold;
            font-size: 1.05rem;
            margin-bottom: 4px;
        }}
        .insight-desc {{
            color: var(--text-muted);
            font-size: 0.95rem;
        }}

        .grid-2col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
            margin-bottom: 30px;
        }}

        .section-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
        }}
        h2 {{
            margin-top: 0;
            font-size: 1.25rem;
            color: var(--text-main);
            border-left: 4px solid var(--accent-blue);
            padding-left: 10px;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            background-color: rgba(255,255,255,0.03);
            color: var(--text-muted);
            font-weight: 600;
            font-size: 0.85rem;
        }}
        .cm-cell {{
            text-align: center;
            border: 1px solid rgba(255,255,255,0.05);
        }}
        .row-label {{
            font-weight: bold;
            color: var(--accent-blue);
        }}
        .progress-container {{
            background: #0f172a;
            border-radius: 6px;
            position: relative;
            height: 22px;
            overflow: hidden;
        }}
        .progress-bar {{
            background: linear-gradient(90deg, #0284c7, #38bdf8);
            height: 100%;
            border-radius: 6px;
        }}
        .progress-text {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 0.8rem;
            font-weight: bold;
            color: #fff;
        }}
        .badge {{
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: bold;
        }}
        .badge-true {{ background: rgba(52, 211, 153, 0.2); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); }}
        .badge-pred {{ background: rgba(248, 113, 113, 0.2); color: #f87171; border: 1px solid rgba(248, 113, 113, 0.3); }}
        .filepath {{ font-family: monospace; font-size: 0.85rem; color: var(--text-muted); word-break: break-all; }}

        /* フィルターボタン */
        .filter-container {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .btn-filter {{
            background: #0f172a;
            color: var(--text-muted);
            border: 1px solid var(--border-color);
            padding: 6px 14px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.2s;
        }}
        .btn-filter:hover {{
            color: var(--text-main);
            border-color: var(--accent-blue);
        }}
        .btn-filter.active {{
            background: var(--accent-blue);
            color: #0f172a;
            font-weight: bold;
            border-color: var(--accent-blue);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 {title}</h1>
            <span style="color: var(--text-muted);">インタラクティブ・モデル解析ダッシュボード</span>
        </div>

        <!-- 1. KPI Cards -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-title">全体正解率 (Accuracy)</div>
                <div class="kpi-value">{overall.accuracy * 100:.2f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Macro F1 スコア</div>
                <div class="kpi-value">{overall.macro_f1 * 100:.2f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Weighted F1 スコア</div>
                <div class="kpi-value">{overall.weighted_f1 * 100:.2f}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">評価サンプル総数</div>
                <div class="kpi-value" style="color: var(--accent-blue);">{overall.total_samples} 件</div>
            </div>
        </div>

        <!-- 2. AI Executive Insights -->
        <div class="insights-section">
            <h2 style="border-left-color: #f59e0b;">💡 モデル改善のAI診断 ＆ アクション提案</h2>
            {insights_html}
        </div>

        <!-- 3. Metrics & Chart Grid -->
        <div class="grid-2col">
            <div class="section-card" style="margin-bottom:0;">
                <h2>📈 クラス別精度指標</h2>
                <table>
                    <thead>
                        <tr>
                            <th>クラス</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1-Score</th>
                            <th>件数</th>
                        </tr>
                    </thead>
                    <tbody>
                        {per_class_rows_html}
                    </tbody>
                </table>
            </div>
            <div class="section-card" style="margin-bottom:0;">
                <h2>📊 F1 / Precision / Recall 比較グラフ</h2>
                <div style="height: 250px;">
                    <canvas id="metricsChart"></canvas>
                </div>
            </div>
        </div>

        <!-- 4. Confusion Matrix Heatmap -->
        <div class="section-card">
            <h2>🧩 混同行列ヒートマップ (Confusion Matrix Heatmap)</h2>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: -10px;">※ 各セル内のパーセンテージは、正解ラベル全体に対する予測割合を示しています。</p>
            <table>
                <thead>
                    <tr>
                        <th>正解 ＼ 予測</th>
                        {cm_headers_html}
                    </tr>
                </thead>
                <tbody>
                    {cm_rows_html}
                </tbody>
            </table>
        </div>

        <!-- 5. Misclassified Samples with Audio Player & Filtering -->
        <div class="section-card">
            <h2>🎧 誤識別サンプルの試聴 ＆ 詳細解析 (Misclassified Samples: {len(misclassified)}件)</h2>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: -10px;">ブラウザ上で直接 ▶ ボタンを押すと実際の音声を試聴できます。問題のある文字カテゴリをクリックして絞り込めます。</p>

            <div class="filter-container">
                {filter_buttons_html}
            </div>

            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>正解</th>
                        <th>予測</th>
                        <th>音声試聴 (Audio Player)</th>
                        <th>ファイルパス</th>
                        <th>確信度</th>
                    </tr>
                </thead>
                <tbody id="misclassifiedTbody">
                    {mis_rows_html}
                </tbody>
            </table>
        </div>
    </div>

    <script>
        // Chart.js 描画スクリプト
        const ctx = document.getElementById('metricsChart').getContext('2d');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {chart_labels_js},
                datasets: [
                    {{
                        label: 'F1-Score (%)',
                        data: {chart_f1_js},
                        backgroundColor: '#38bdf8',
                        borderRadius: 4
                    }},
                    {{
                        label: 'Precision (%)',
                        data: {chart_prec_js},
                        backgroundColor: '#34d399',
                        borderRadius: 4
                    }},
                    {{
                        label: 'Recall (%)',
                        data: {chart_rec_js},
                        backgroundColor: '#f87171',
                        borderRadius: 4
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100,
                        grid: {{ color: '#334155' }},
                        ticks: {{ color: '#94a3b8' }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#94a3b8' }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        labels: {{ color: '#f8fafc' }}
                    }}
                }}
            }}
        }});

        // カテゴリ絞り込みフィルター処理
        function filterCategory(cat) {{
            const btns = document.querySelectorAll('.btn-filter');
            btns.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            const rows = document.querySelectorAll('.mis-row');
            rows.forEach(row => {{
                if (cat === 'all') {{
                    row.style.display = '';
                }} else {{
                    if (row.classList.contains('category-' + cat)) {{
                        row.style.display = '';
                    }} else {{
                        row.style.display = 'none';
                    }}
                }}
            }});
        }}
    </script>
</body>
</html>
"""
    return html


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
