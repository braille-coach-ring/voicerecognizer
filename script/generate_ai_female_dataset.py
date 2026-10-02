#!/usr/bin/env python3
"""
AI Female Voice Dataset Generator using Microsoft Neural TTS.

全104音節のひらがなに対して、自然な女性音声（NanamiNeural）の
3つの音色パターン（標準・高め・落ち着き）に微小なランダム性を加えて
WAV音声ファイル（16kHz, PCM_16）を生成し、独立した話者ディレクトリ
(dataset/ai_female_nanami/) に格納します。

安全設計:
- 独立した話者ディレクトリに配置されるため、既存の実録話者データには一切干渉しません。
- --clean オプションにより、ワンクリックで本データセットを安全に完全削除・分離可能です。
"""

import argparse
import asyncio
import io
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import edge_tts
import librosa
import numpy as np
import soundfile as sf

from voicerecognizer.config_labels import ALL_HIRAGANA_LABELS, ROMAJI_TO_HIRAGANA


@dataclass(frozen=True)
class GenerationPattern:
    filename: str
    pitch_base: float
    rate_base: float


async def synthesize_phoneme(
    text: str,
    pitch_offset_hz: str,
    rate_offset_pct: str,
    voice_name: str = "ja-JP-NanamiNeural",
    sample_rate: int = 16000,
) -> np.ndarray:
    """Neural TTS で単音節音声を合成し、トリミングして 16kHz float32 で返す。"""
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice_name,
        pitch=pitch_offset_hz,
        rate=rate_offset_pct,
    )
    mp3_bytes = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_bytes.extend(chunk["data"])

    if not mp3_bytes:
        return np.zeros(sample_rate // 2, dtype=np.float32)

    bio = io.BytesIO(bytes(mp3_bytes))
    y, sr = sf.read(bio, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    if sr != sample_rate:
        y = np.asarray(
            librosa.resample(y, orig_sr=sr, target_sr=sample_rate),
            dtype=np.float32,
        )

    # 発話区間トリミング (前後の無音を除去)
    trimmed, _ = librosa.effects.trim(y, top_db=22)
    # 単音節長として 0.5s (8000サンプル) を下限、1.0s (16000サンプル) を上限とする
    min_len = int(sample_rate * 0.5)
    max_len = int(sample_rate * 1.0)
    if len(trimmed) < min_len:
        trimmed = np.pad(trimmed, (0, min_len - len(trimmed)))
    elif len(trimmed) > max_len:
        trimmed = trimmed[:max_len]

    # ピーク正規化 (0.85〜0.92)
    max_val = float(np.max(np.abs(trimmed)))
    if max_val > 0.0:
        trimmed = (trimmed * (0.88 / max_val)).astype(np.float32)

    return np.ascontiguousarray(trimmed, dtype=np.float32)


async def generate_dataset(
    target_dir: Path,
    labels: Sequence[str],
    seed: int = 42,
) -> None:
    """全音節に対して 3 パターン × ランダム変動で音声ファイルを生成する。"""
    rng = np.random.default_rng(seed)
    target_dir.mkdir(parents=True, exist_ok=True)

    base_patterns: list[GenerationPattern] = [
        GenerationPattern(filename="001.wav", pitch_base=0.0, rate_base=0.0),    # 1. Standard
        GenerationPattern(filename="002.wav", pitch_base=25.0, rate_base=4.0),   # 2. Bright / High
        GenerationPattern(filename="003.wav", pitch_base=-20.0, rate_base=-4.0), # 3. Calm / Deep
    ]

    total_labels = len(labels)
    print(f"[*] Starting dataset generation for {total_labels} phoneme labels...")
    print(f"[*] Target Directory: {target_dir}")

    success_count = 0
    for idx, label in enumerate(labels, 1):
        if label == "other":
            # other は発話ひらがなが存在しないためスキップ
            continue

        char = ROMAJI_TO_HIRAGANA.get(label)
        if not char:
            print(f"[WARN] No hiragana mapping for label '{label}', skipping.")
            continue

        cls_dir = target_dir / label
        cls_dir.mkdir(parents=True, exist_ok=True)

        for pat in base_patterns:
            fname = pat.filename
            out_path = cls_dir / fname

            # 少しランダム性を混ぜる (ピッチ ±5Hz, 速度 ±3%)
            pitch_rand = round(pat.pitch_base + float(rng.uniform(-5.0, 5.0)))
            rate_rand = round(pat.rate_base + float(rng.uniform(-3.0, 3.0)))

            p_sign = "+" if pitch_rand >= 0 else ""
            r_sign = "+" if rate_rand >= 0 else ""
            pitch_arg = f"{p_sign}{pitch_rand}Hz"
            rate_arg = f"{r_sign}{rate_rand}%"

            try:
                wav_data = await synthesize_phoneme(
                    text=char,
                    pitch_offset_hz=pitch_arg,
                    rate_offset_pct=rate_arg,
                )
                sf.write(out_path, wav_data, 16000, format="WAV", subtype="PCM_16")
                success_count += 1
                await asyncio.sleep(0.03)
            except Exception as e:
                print(f"[ERROR] Failed to synthesize {label} ({fname}): {e}")

        if idx % 10 == 0 or idx == total_labels:
            print(f"  [{idx:3d}/{total_labels}] Processed '{label}' (「{char}」)")

    print(f"\n[SUCCESS] Generated {success_count} audio files across {len(labels)} classes in {target_dir}")


def clean_dataset(target_dir: Path) -> None:
    """生成したデータセットディレクトリを安全に完全削除する。"""
    if target_dir.exists():
        shutil.rmtree(target_dir)
        print(f"[CLEANED] Successfully removed dataset directory: {target_dir}")
    else:
        print(f"[INFO] Directory does not exist, nothing to clean: {target_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate AI Female Voice Dataset using Microsoft Neural TTS"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/ai_female_nanami"),
        help="Target speaker directory to place generated dataset",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the generated dataset directory completely",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for pitch and rate perturbation",
    )
    args = parser.parse_args()

    if args.clean:
        clean_dataset(args.output_dir)
        return

    # other を除く全104音節を対象とする
    target_labels = [lbl for lbl in ALL_HIRAGANA_LABELS if lbl != "other"]

    asyncio.run(
        generate_dataset(
            target_dir=args.output_dir,
            labels=target_labels,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
