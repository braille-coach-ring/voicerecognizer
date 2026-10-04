#!/usr/bin/env python3
"""
Strategy Benchmark & Leaderboard Generator Script.

役割:
  登録されている音声認識ストラテジー（あるいは指定されたストラテジー）に対し、
  以下の統一評価基準でベンチマークを実行し、リーダーボード (JSON & Markdown) を自動生成します。
  1. General Val Accuracy (data_splits/combined_val.csv または merged_dataset/index.csv)
  2. Speakerphone Test Accuracy (data_splits/speakerphone_test.csv)
  3. Female Voice Test Accuracy (data_splits/female_test.csv)
  4. Average Inference Latency (ms) on CPU
  5. ONNX / Weight Size (MB)

使い方:
  uv run python script/benchmark_strategies.py --strategy wav2vec2_baseline
  uv run python script/benchmark_strategies.py --all
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from voicerecognizer.config import PROJECT_ROOT
from voicerecognizer.core.factory import RecognizerFactory
from voicerecognizer.strategies.registry import (
    DEFINED_STRATEGIES,
    StrategyStatus,
    get_strategy_metadata,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
LEADERBOARD_JSON = BENCHMARKS_DIR / "strategies_leaderboard.json"
LEADERBOARD_MD = BENCHMARKS_DIR / "LEADERBOARD.md"

DATA_SPLITS_DIR = PROJECT_ROOT / "data_splits"


@dataclass
class StrategyBenchmarkResult:
    strategy_name: str
    display_name: str
    category: str
    status: str
    general_val_acc: float | None = None
    speakerphone_test_acc: float | None = None
    female_test_acc: float | None = None
    latency_ms: float | None = None
    model_size_mb: float | None = None
    evaluated_at: str | None = None
    notes: str = ""


def _resolve_csv_audio_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    # Relative to project root
    p_proj = PROJECT_ROOT / p
    if p_proj.exists():
        return p_proj
    # Relative to processed_dataset
    p_proc = PROJECT_ROOT / "processed_dataset" / p
    if p_proc.exists():
        return p_proc
    return p


def evaluate_split(recognizer, split_csv: Path) -> float | None:
    if not split_csv.exists():
        logger.warning("Split file not found: %s", split_csv)
        return None

    correct = 0
    total = 0

    with open(split_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath_str = row.get("filepath") or row.get("source_filepath") or ""
            target_label = row.get("label") or ""
            if not filepath_str or not target_label:
                continue

            audio_path = _resolve_csv_audio_path(filepath_str)
            if not audio_path.exists():
                continue

            try:
                waveform, _ = sf.read(audio_path, dtype="float32", always_2d=False)
                if waveform.ndim > 1:
                    waveform = np.mean(waveform, axis=1)

                pred = recognizer.recognize(waveform)
                if pred == target_label:
                    correct += 1
                total += 1
            except Exception as e:
                logger.debug("Failed recognizing %s: %s", audio_path, e)

    if total == 0:
        return None
    return round((correct / total) * 100, 2)


def measure_latency(recognizer, sample_rate: int = 16000, runs: int = 50) -> float:
    # 0.6s dummy audio
    audio = np.random.randn(int(sample_rate * 0.6)).astype(np.float32)
    # Warmup
    for _ in range(5):
        recognizer.recognize(audio)

    start = time.perf_counter()
    for _ in range(runs):
        recognizer.recognize(audio)
    elapsed = time.perf_counter() - start
    return round((elapsed / runs) * 1000, 2)


def measure_model_size_mb(model_dir: Path) -> float | None:
    if not model_dir.exists():
        return None
    onnx_files = list(model_dir.glob("*.onnx"))
    if onnx_files:
        # Prefer int8 or mel_int8
        for candidate in ("model_mel_int8.onnx", "model_int8.onnx", "model.onnx"):
            p = model_dir / candidate
            if p.exists():
                return round(p.stat().st_size / (1024 * 1024), 1)
        return round(onnx_files[0].stat().st_size / (1024 * 1024), 1)

    safetensors = model_dir / "model.safetensors"
    if safetensors.exists():
        return round(safetensors.stat().st_size / (1024 * 1024), 1)

    return None


def benchmark_strategy(strategy_name: str) -> StrategyBenchmarkResult:
    meta = get_strategy_metadata(strategy_name)
    logger.info("=== Benchmarking Strategy: %s (%s) ===", meta.display_name, strategy_name)

    recognizer = RecognizerFactory.create(strategy_name)

    logger.info("1/4 Evaluating General Validation Accuracy...")
    general_acc = evaluate_split(recognizer, DATA_SPLITS_DIR / "combined_val.csv")

    logger.info("2/4 Evaluating Speakerphone Test Accuracy...")
    speakerphone_acc = evaluate_split(recognizer, DATA_SPLITS_DIR / "speakerphone_test.csv")

    logger.info("3/4 Evaluating Female Test Accuracy...")
    female_acc = evaluate_split(recognizer, DATA_SPLITS_DIR / "female_test.csv")

    logger.info("4/4 Measuring Inference Latency on CPU...")
    latency_ms = measure_latency(recognizer)

    size_mb = measure_model_size_mb(meta.model_dir)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    res = StrategyBenchmarkResult(
        strategy_name=strategy_name,
        display_name=meta.display_name,
        category=meta.category.value,
        status="active" if meta.status == StrategyStatus.ACTIVE else meta.status.value,
        general_val_acc=general_acc,
        speakerphone_test_acc=speakerphone_acc,
        female_test_acc=female_acc,
        latency_ms=latency_ms,
        model_size_mb=size_mb,
        evaluated_at=timestamp,
        notes=meta.description,
    )
    logger.info(
        "Result: General=%s%%, Speakerphone=%s%%, Female=%s%%, Latency=%sms, Size=%sMB",
        general_acc,
        speakerphone_acc,
        female_acc,
        latency_ms,
        size_mb,
    )
    return res


def split_hashes() -> dict[str, str]:
    from script.evaluate_speaker_independent import hash_file
    from voicerecognizer.config import DEFAULT_SPEAKER_SPLIT_DIR

    return {
        name: hash_file(DEFAULT_SPEAKER_SPLIT_DIR / f"{name}.csv")
        for name in ("train", "val", "test")
        if (DEFAULT_SPEAKER_SPLIT_DIR / f"{name}.csv").exists()
    }


def load_leaderboard() -> dict[str, Any]:
    if LEADERBOARD_JSON.exists():
        try:
            data = json.loads(LEADERBOARD_JSON.read_text(encoding="utf-8"))
            if data.get("split_hashes") != split_hashes():
                return {"strategies": {}}
            return data
        except Exception:
            pass
    return {"strategies": {}}


def save_leaderboard(leaderboard: dict[str, Any]) -> None:
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    LEADERBOARD_JSON.write_text(
        json.dumps(leaderboard, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    generate_markdown_leaderboard(leaderboard)


def generate_markdown_leaderboard(leaderboard: dict[str, Any]) -> None:
    rows = []
    strategies = leaderboard.get("strategies", {})

    header = (
        "# Voicerecognizer Strategy Leaderboard\n\n"
        "汎用精度および実環境ロバストネス向上のための各実験ストラテジーの横並びスコアです。\n\n"
        "| Strategy | Category | Status | General Val Acc | Speakerphone Acc | Female Acc | CPU Latency | Model Size |\n"
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    )

    for strat_key, data in strategies.items():
        g_acc = (
            f"{data['general_val_acc']:.1f}%" if data.get("general_val_acc") is not None else "-"
        )
        sp_acc = (
            f"{data['speakerphone_test_acc']:.1f}%"
            if data.get("speakerphone_test_acc") is not None
            else "-"
        )
        fem_acc = (
            f"{data['female_test_acc']:.1f}%" if data.get("female_test_acc") is not None else "-"
        )
        lat = f"{data['latency_ms']:.1f} ms" if data.get("latency_ms") is not None else "-"
        size = f"{data['model_size_mb']:.1f} MB" if data.get("model_size_mb") is not None else "-"
        status = data.get("status", "planned")

        rows.append(
            f"| `{strat_key}` | {data.get('category', '-')} | {status} | {g_acc} | {sp_acc} | {fem_acc} | {lat} | {size} |"
        )

    # Add planned strategies that have no benchmark yet
    for key, meta in DEFINED_STRATEGIES.items():
        if key not in strategies:
            rows.append(
                f"| `{key}` | {meta.category.value} | {meta.status.value} | - | - | - | - | - |"
            )

    content = header + "\n".join(rows) + "\n"
    LEADERBOARD_MD.write_text(content, encoding="utf-8")
    logger.info("Leaderboard updated at %s and %s", LEADERBOARD_JSON, LEADERBOARD_MD)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Voice Recognition Strategies")
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=list(DEFINED_STRATEGIES.keys()),
        help="Specific strategy to benchmark",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Benchmark all implemented strategies",
    )
    args = parser.parse_args()

    targets: list[str] = []
    if args.strategy:
        targets.append(args.strategy)
    elif args.all:
        for name, meta in DEFINED_STRATEGIES.items():
            if meta.status in (StrategyStatus.ACTIVE, StrategyStatus.IMPLEMENTED):
                targets.append(name)
    else:
        # Default to baseline
        targets.append("wav2vec2_baseline")

    leaderboard = load_leaderboard()

    for target in targets:
        try:
            res = benchmark_strategy(target)
            leaderboard["strategies"][target] = asdict(res)
            save_leaderboard(leaderboard)
        except Exception as e:
            logger.error("Failed to benchmark strategy %s: %s", target, e, exc_info=True)


if __name__ == "__main__":
    main()
