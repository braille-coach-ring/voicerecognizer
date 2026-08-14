"""
Wav2Vec2 Pre-trained Base Model Benchmark Script

役割:
  事前学習済み Wav2Vec2 ベースモデル群 (ReazonSpeech, XLSR-53, Rinna, Meta等) の音響特徴表現力
  を線形分類 (Linear Probe) で比較・ベンチマークし、最も優秀なベースモデルを選定します。

評価方法:
  1. 各モデルから未学習（Freeze状態）のまま音声特徴ベクトル (Embedding) を抽出
  2. Stratified 80/20 分割データに対して Simple Linear Classifier (Logistic Regression) を学習
  3. テストデータの Accuracy, Macro-F1, Weighted-F1 を算出してモデル間で性能比較

使い方:
  uv run python script/benchmark_base_models.py
  uv run python script/benchmark_base_models.py --models reazon-research/japanese-wav2vec2-base facebook/wav2vec2-large-xlsr-53
"""

import argparse
import contextlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from config import DEFAULT_RECOGNITION_CONFIG, PROJECT_ROOT
from dataset.hiragana_dataset import HiraganaDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# 比較対象のデフォルト Wav2Vec2 ベースモデル候補一覧
DEFAULT_MODEL_CANDIDATES = [
    "reazon-research/japanese-wav2vec2-base",
    "facebook/wav2vec2-large-xlsr-53",
    "vumichien/wav2vec2-large-xlsr-japanese-hiragana",
    "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "rinna/japanese-wav2vec2-base",
    "facebook/wav2vec2-base",  # 比較対照用の英語ベースモデル
]


def load_dataset_audio_and_labels(
    dataset_dir: Path, sample_rate: int = 16000, target_length_seconds: float = 0.6
) -> tuple[list[np.ndarray], list[int], list[str]]:
    """データセットから全音声データとラベルインデックスを読み込む"""
    dataset = HiraganaDataset(root_dir=dataset_dir, sample_rate=sample_rate, cache_in_memory=False)
    target_samples = int(target_length_seconds * sample_rate)

    waveforms = []
    labels = []
    label_names = dataset.labels

    for wav_path, label in dataset.data:
        waveform, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1, dtype=np.float32)
        waveform = waveform.reshape(-1)

        if sr != sample_rate:
            import librosa

            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=sample_rate).astype(
                np.float32
            )

        if target_samples > 0:
            if len(waveform) > target_samples:
                waveform = waveform[:target_samples]
            elif len(waveform) < target_samples:
                waveform = np.pad(waveform, (0, target_samples - len(waveform)))

        waveforms.append(waveform)
        labels.append(label)

    return waveforms, labels, list(label_names)


def extract_features_for_model(
    model_name: str,
    waveforms: list[np.ndarray],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """指定された Wav2Vec2 モデルから音声埋め込みベクトル (Embeddings) を抽出する"""
    from transformers import AutoFeatureExtractor, AutoModel

    logger.info("📦 Loading model and feature extractor: %s ...", model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(waveforms), batch_size):
            batch_waveforms = waveforms[i : i + batch_size]
            inputs = feature_extractor(
                batch_waveforms,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)

            # 最終層の hidden_state を時間軸方向に Mean-pooling して埋め込み表現化
            hidden_states = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_dim)
            embeddings = hidden_states.mean(dim=1).cpu().numpy()
            all_embeddings.append(embeddings)

    del model
    del feature_extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.concatenate(all_embeddings, axis=0)


def evaluate_linear_probe(features: np.ndarray, y: list[int], seed: int = 42) -> dict[str, float]:
    """抽出した特徴量に対して線形分類器 (Logistic Regression) を学習・評価する"""
    features_train, features_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=seed, stratify=y
    )

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    clf.fit(features_train, y_train)
    preds = clf.predict(features_test)

    acc = float(accuracy_score(y_test, preds))
    macro_f1 = float(f1_score(y_test, preds, average="macro"))
    weighted_f1 = float(f1_score(y_test, preds, average="weighted"))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark pre-trained Wav2Vec2 base models for Japanese Hiragana recognition"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODEL_CANDIDATES,
        help="List of Hugging Face Wav2Vec2 model names to benchmark",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_RECOGNITION_CONFIG.processed_dataset_dir
        if DEFAULT_RECOGNITION_CONFIG.processed_dataset_dir.exists()
        else DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "evaluation_results" / "base_model_benchmark.json",
        help="Output path for benchmark results JSON",
    )

    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        with contextlib.suppress(Exception):
            sys.stdout.reconfigure(encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    logger.info("Loading audio files from %s ...", args.dataset_dir)
    waveforms, labels, label_names = load_dataset_audio_and_labels(dataset_dir=args.dataset_dir)
    logger.info("Loaded %d audio samples across %d classes.", len(waveforms), len(label_names))

    results: list[dict[str, Any]] = []

    print("\n" + "=" * 80)
    print(" === Wav2Vec2 Base Model Benchmark Start ===")
    print(" (Zero-shot / Linear Probe classification accuracy for pretrained embeddings)")
    print("=" * 80 + "\n")

    for model_name in args.models:
        logger.info("--------------------------------------------------")
        logger.info("Evaluating: %s", model_name)
        start_time = time.time()
        try:
            embeddings = extract_features_for_model(
                model_name=model_name, waveforms=waveforms, device=device
            )
            metrics = evaluate_linear_probe(features=embeddings, y=labels)
            elapsed = time.time() - start_time

            res = {
                "model_name": model_name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "elapsed_seconds": round(elapsed, 2),
                "embedding_dim": int(embeddings.shape[1]),
                "status": "success",
            }
            results.append(res)

            logger.info(
                "[SUCCESS] [%s] Accuracy: %.4f | Macro F1: %.4f | Time: %.1fs",
                model_name,
                metrics["accuracy"],
                metrics["macro_f1"],
                elapsed,
            )
        except Exception as exc:
            logger.error("[FAILED] Model %s: %s", model_name, exc)
            results.append(
                {
                    "model_name": model_name,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    successful_results = [r for r in results if r.get("status") == "success"]
    successful_results.sort(key=lambda x: x["accuracy"], reverse=True)

    print("\n" + "=" * 85)
    print(" Benchmark Summary (Linear Probe Accuracy)")
    print("=" * 85)
    print(
        f"{'Rank':<5} {'Model Name':<50} {'Accuracy':<10} {'Macro F1':<10} {'Dim':<6} {'Time(s)':<8}"
    )
    print("-" * 85)
    for rank, res in enumerate(successful_results, 1):
        print(
            f"{rank:<5} {res['model_name']:<50} {res['accuracy'] * 100:<9.2f}% {res['macro_f1']:<10.4f} {res['embedding_dim']:<6} {res['elapsed_seconds']:<8.1f}"
        )
    print("=" * 85 + "\n")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Benchmark results saved to: %s", args.output_json)


if __name__ == "__main__":
    main()
