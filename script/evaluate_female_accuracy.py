#!/usr/bin/env python3
"""
Evaluate Existing Models on AI Female Voice Dataset.

既存モデル (Wav2Vec2 ONNX / CNN) に対し、生成した女性話者データ
(dataset/ai_female_nanami/ の 312 ファイル) を入力し、
Top-1 正答率、Top-3 正答率、音種別・バリエーション別内訳、
および誤認識ペアの詳細レポートを出力します。
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from voicerecognizer.config_labels import (
    DAKUON_LABELS,
    HANDAKUON_LABELS,
    ROMAJI_TO_HIRAGANA,
    SEION_LABELS,
    YOON_LABELS,
)
from voicerecognizer.recognizers.cnn_recognizer import CNNRecognizer
from voicerecognizer.recognizers.wav2vec2_recognizer import Wav2Vec2Recognizer


@dataclass
class PatternStat:
    name: str
    correct: int = 0
    total: int = 0


def get_label_category(label: str) -> str:
    """音節ラベルのカテゴリ（清音、濁音、半濁音、拗音）を判定。"""
    if label in SEION_LABELS:
        return "清音 (46音)"
    if label in DAKUON_LABELS:
        return "濁音 (20音)"
    if label in HANDAKUON_LABELS:
        return "半濁音 (5音)"
    if label in YOON_LABELS:
        return "拗音 (33音)"
    return "その他"


def evaluate_dataset(
    dataset_dir: Path,
    use_cnn: bool = False,
) -> None:
    """女性話者データセット全体に対する正答率測定と分析レポート出力。"""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # モデルの初期化
    model_name = "CNN" if use_cnn else "Wav2Vec2 (ONNX)"
    print(f"[*] Initializing model: {model_name}...")
    recognizer = CNNRecognizer() if use_cnn else Wav2Vec2Recognizer()

    wav_files = sorted(dataset_dir.rglob("*.wav"))
    total_files = len(wav_files)
    if total_files == 0:
        print("[ERROR] No WAV files found in dataset directory!")
        return

    print(f"[*] Found {total_files} audio files in {dataset_dir}")
    print(f"[*] Starting evaluation with {model_name}...\n")

    correct_top1 = 0
    correct_top3 = 0

    # パターン別 (001: Standard, 002: High, 003: Deep)
    pattern_stats: dict[str, PatternStat] = {
        "001": PatternStat(name="Standard (標準)"),
        "002": PatternStat(name="Bright (高め)"),
        "003": PatternStat(name="Calm (落ち着き)"),
    }

    # カテゴリ別
    category_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    # 誤認識リスト
    mismatches: list[dict[str, str]] = []

    for wav_path in wav_files:
        true_label = wav_path.parent.name
        pattern_key = wav_path.stem[:3]  # "001", "002", "003"
        category = get_label_category(true_label)

        waveform, _ = sf.read(wav_path, dtype="float32", always_2d=False)

        top3_preds: list[str] = []
        score_str = "1.000"
        top3_desc = ""

        if isinstance(recognizer, Wav2Vec2Recognizer):
            candidates = recognizer.recognize_with_candidates(waveform, top_k=3)
            if not candidates:
                continue
            top1_pred = candidates[0][0]
            top3_preds = [c[0] for c in candidates[:3]]
            score_str = f"{candidates[0][1]:.3f}"
            top3_desc = ", ".join([f"{c[0]}({c[1]:.2f})" for c in candidates[:3]])
        else:
            top1_pred = recognizer.recognize(waveform)
            top3_preds = [top1_pred]
            top3_desc = top1_pred

        # 集計
        is_top1 = (top1_pred == true_label)
        is_top3 = (true_label in top3_preds)

        if is_top1:
            correct_top1 += 1
        else:
            char_true = ROMAJI_TO_HIRAGANA.get(true_label, true_label)
            char_pred = ROMAJI_TO_HIRAGANA.get(top1_pred, top1_pred)
            mismatches.append(
                {
                    "file": wav_path.name,
                    "true_label": true_label,
                    "true_char": char_true,
                    "pred_label": top1_pred,
                    "pred_char": char_pred,
                    "score": score_str,
                    "top3": top3_desc,
                }
            )

        if is_top3:
            correct_top3 += 1

        if pattern_key in pattern_stats:
            pattern_stats[pattern_key].total += 1
            if is_top1:
                pattern_stats[pattern_key].correct += 1

        category_stats[category]["total"] += 1
        if is_top1:
            category_stats[category]["correct"] += 1

    # レポート表示
    acc_top1 = (correct_top1 / total_files) * 100.0
    acc_top3 = (correct_top3 / total_files) * 100.0

    print("=" * 65)
    print("     【女性話者データ (ai_female_nanami) 既存モデル認識精度レポート】")
    print(f"      モデル: {model_name}")
    print("=" * 65)
    print("■ 全体正答率 (Overall Accuracy):")
    print(f"  ・Top-1 Accuracy : {correct_top1:>3d} / {total_files:>3d} ({acc_top1:>6.2f}%)")
    print(f"  ・Top-3 Accuracy : {correct_top3:>3d} / {total_files:>3d} ({acc_top3:>6.2f}%)")

    print("\n■ 音色バリエーション別 Top-1 正答率:")
    for p_data in pattern_stats.values():
        p_tot = p_data.total
        p_cor = p_data.correct
        p_acc = (p_cor / p_tot * 100.0) if p_tot > 0 else 0.0
        print(f"  ・{p_data.name:<18}: {p_cor:>3d} / {p_tot:>3d} ({p_acc:>6.2f}%)")

    print("\n■ 音種カテゴリ別 Top-1 正答率:")
    for cat_name in ["清音 (46音)", "濁音 (20音)", "半濁音 (5音)", "拗音 (33音)"]:
        c_data = category_stats[cat_name]
        c_tot = c_data["total"]
        c_cor = c_data["correct"]
        c_acc = (c_cor / c_tot * 100.0) if c_tot > 0 else 0.0
        print(f"  ・{cat_name:<16}: {c_cor:>3d} / {c_tot:>3d} ({c_acc:>6.2f}%)")

    print(f"\n■ 誤認識サンプル内訳 (計 {len(mismatches)} 件):")
    if not mismatches:
        print("  誤認識なし（全問正解！）")
    else:
        for idx, m in enumerate(mismatches[:20], 1):
            print(
                f"  [{idx:2d}] 正解: {m['true_label']:<4} (「{m['true_char']}」) -> "
                f"誤予測: {m['pred_label']:<4} (「{m['pred_char']}」, 確信度={m['score']}) "
                f"| Top3: [{m['top3']}]"
            )
        if len(mismatches) > 20:
            print(f"  ... 他 {len(mismatches) - 20} 件")

    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate existing models on AI Female voice dataset"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset/ai_female_nanami"),
        help="Path to AI female dataset directory",
    )
    parser.add_argument(
        "--cnn",
        action="store_true",
        help="Evaluate with CNN model instead of Wav2Vec2",
    )
    args = parser.parse_args()

    evaluate_dataset(dataset_dir=args.dataset_dir, use_cnn=args.cnn)


if __name__ == "__main__":
    main()
