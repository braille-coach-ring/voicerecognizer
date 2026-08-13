import logging
import shutil
from pathlib import Path

import soundfile as sf

from config import (
    DEFAULT_AUDIO_CONFIG,
    DEFAULT_PREPROCESS_CONFIG,
    DEFAULT_RECOGNITION_CONFIG,
    PROJECT_ROOT,
)
from preprocessing.audio_preprocessor import AudioPreprocessor

logger = logging.getLogger(__name__)


def _to_rel_path(path: Path) -> str:
    """PROJECT_ROOTからの相対パス (POSIX形式) に変換する"""
    try:
        rel = path.resolve().relative_to(PROJECT_ROOT.resolve())
        return rel.as_posix()
    except ValueError:
        return str(path)


"""
DatasetBuilder: 学習用データセット (merged_dataset / processed_dataset) を構築・集約するクラス。

【インデックス (マニフェスト) 方式のデータ構造】
1. 生データ保存場所 (.wav 本体は物理コピーしない):
   - Rawデータ: dataset/<個人名>/<ラベル>/*.wav
   - Collectedデータ: dataset/collected/pc_xxxxxxxx/*.wav (および metadata.csv)

2. merged_dataset/index.csv:
   - 全データソースから有効な正解データ (ground_truth) の場所とラベルを1つのテキストファイルにインデックス化。
   - フォーマット: [filepath, label]
   - 音声ファイル本体のコピーが発生しないため、超高速かつディスク消費ゼロ。

3. processed_dataset/:
   - preprocess_dataset() により、index.csv に記載されたパスから波形を読み込んで前処理し、学習用に保存。
"""


class DatasetBuilder:
    def __init__(
        self,
        labels: tuple[str, ...] = DEFAULT_RECOGNITION_CONFIG.labels,
        sample_rate: int = DEFAULT_AUDIO_CONFIG.sample_rate,
        target_length_seconds: float = DEFAULT_RECOGNITION_CONFIG.target_length_seconds,
        top_db: float = DEFAULT_PREPROCESS_CONFIG.top_db,
    ):
        self.labels = labels
        self.preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            target_length_seconds=target_length_seconds,
            top_db=top_db,
        )
        logger.info("DatasetBuilderの初期化完了 (インデックスマニフェスト方式)")

    def build_index(
        self,
        raw_root: str | Path = DEFAULT_RECOGNITION_CONFIG.raw_dataset_dir,
        collected_dir: str | Path = DEFAULT_RECOGNITION_CONFIG.collected_dataset_dir,
        output_root: str | Path = DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir,
    ) -> Path:
        raw_root = Path(raw_root)
        collected_dir = Path(collected_dir)
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        index_file = output_root / "index.csv"

        entries: list[tuple[str, str, str]] = []

        # 1. Rawデータの収集 (dataset/<person>/<label>/*.wav)
        if raw_root.exists():
            for person in sorted(raw_root.iterdir()):
                if not person.is_dir() or person.name in ("collected", "__pycache__"):
                    continue
                for label in self.labels:
                    input_folder = person / label
                    if input_folder.is_dir():
                        for wav_path in sorted(input_folder.glob("*.wav")):
                            entries.append((_to_rel_path(wav_path), label, ""))

        # 2. Collectedデータの収集 (dataset/collected/pc_xxxxxxxx/metadata.csv)
        if collected_dir.exists():
            for metadata_file in collected_dir.rglob("metadata.csv"):
                folder = metadata_file.parent
                with open(metadata_file, encoding="utf-8") as f:
                    for line in f:
                        parts = [p.strip() for p in line.strip().split(",")]
                        if len(parts) < 3 or not parts[0]:
                            continue
                        filename = parts[1]

                        if len(parts) >= 4:
                            predicted_text = parts[2]
                            ground_truth = parts[3]
                        else:
                            ground_truth = parts[2]
                            predicted_text = ""

                        if ground_truth and (
                            ground_truth in self.labels or ground_truth == "other"
                        ):
                            wav_path = folder / filename
                            if wav_path.exists():
                                entries.append(
                                    (_to_rel_path(wav_path), ground_truth, predicted_text)
                                )

        # 3. index.csv の書き出し
        with open(index_file, "w", encoding="utf-8") as f:
            f.write("filepath,label,predicted_text\n")
            for filepath, label, pred_text in entries:
                f.write(f"{filepath},{label},{pred_text}\n")

        logger.info("インデックスファイルを作成しました: %s (全 %d 件)", index_file, len(entries))
        return index_file

    def merge_by_label(
        self,
        source_root: str | Path = DEFAULT_RECOGNITION_CONFIG.raw_dataset_dir,
        output_root: str | Path = DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir,
    ) -> None:
        self.build_index(raw_root=source_root, output_root=output_root)

    def merge_collected_dataset(
        self,
        collected_dir: str | Path = DEFAULT_RECOGNITION_CONFIG.collected_dataset_dir,
        output_root: str | Path = DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir,
    ) -> None:
        self.build_index(collected_dir=collected_dir, output_root=output_root)

    def preprocess_dataset(
        self,
        input_root: str | Path = DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir,
        output_root: str | Path = DEFAULT_RECOGNITION_CONFIG.processed_dataset_dir,
    ) -> None:
        input_root = Path(input_root)
        output_root = Path(output_root)
        if output_root.exists():
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True)

        index_file = input_root / "index.csv" if input_root.is_dir() else input_root

        if index_file.exists() and index_file.is_file():
            # インデックス CSV ファイルから直接読み込んで前処理
            counts: dict[str, int] = {}
            with open(index_file, encoding="utf-8") as f:
                _ = f.readline()
                for line in f:
                    parts = [p.strip() for p in line.strip().split(",")]
                    if len(parts) < 2 or not parts[0]:
                        continue
                    wav_path = Path(parts[0])
                    if not wav_path.is_absolute():
                        wav_path = PROJECT_ROOT / wav_path
                    label = parts[1]

                    if not wav_path.exists():
                        continue

                    label_dir = output_root / label
                    label_dir.mkdir(parents=True, exist_ok=True)

                    count = counts.get(label, 1)
                    waveform = self.preprocessor.preprocess_waveform(wav_path)
                    sf.write(
                        label_dir / f"{count:03d}.wav",
                        waveform,
                        self.preprocessor.sample_rate,
                    )
                    counts[label] = count + 1

            logger.info(
                "index.csv から全 %d 件の音声データを前処理して %s に出力しました",
                sum(counts.values()),
                output_root,
            )
            return

        # 従来のディレクトリベース処理（後方互換用）
        for label_dir in sorted(input_root.iterdir()):
            if not label_dir.is_dir():
                continue

            output_dir = output_root / label_dir.name
            output_dir.mkdir(exist_ok=True)
            file_number = 1

            for wav_path in sorted(label_dir.glob("*.wav")):
                waveform = self.preprocessor.preprocess_waveform(wav_path)
                sf.write(
                    output_dir / f"{file_number:03d}.wav",
                    waveform,
                    self.preprocessor.sample_rate,
                )
                file_number += 1
            logger.info(f"{label_dir.name}の音声データを{output_dir}にコピーしました")


def ensure_merged_and_preprocessed(skip_prep: bool = False) -> None:
    """
    学習の実行前に自動でデータ統合 (merge_data) および音声前処理 (preprocess) を実行します。
    skip_prep=True (--skip-prep 指定時) の場合は明示的に自動処理をスキップします。
    """
    if skip_prep:
        logger.info(
            "=== [--skip-prep が指定されたため、自動データ統合・前処理をスキップします] ==="
        )
        return

    logger.info("=== [学習前自動処理] データ統合 (index.csv 生成) ＆ 音声前処理を開始します ===")
    builder = DatasetBuilder()
    index_file = builder.build_index()
    logger.info("  [1/2] 統合インデックス作成完了: %s", index_file)
    builder.preprocess_dataset()
    logger.info("  [2/2] 音声前処理完了: processed_dataset/")
