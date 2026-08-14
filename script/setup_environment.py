"""VoiceRecognizer 環境一括初期化セットアップスクリプト

リポジトリをクローンした直後に本スクリプトを実行することで、
推論・開発に必要なすべての準備（モデル取得・ディレクトリ構成・データセット統合・前処理）をワンコマンドで完了します。

使い方:
  uv run python script/setup_environment.py
  uv run python script/setup_environment.py --skip-dataset
  uv run python script/setup_environment.py --skip-models
"""

import argparse
import logging
from pathlib import Path

from voicerecognizer.config import (
    DEFAULT_RECOGNITION_CONFIG,
    PROJECT_ROOT,
    load_env,
)
from voicerecognizer.preprocessing.dataset_builder import DatasetBuilder
from voicerecognizer.utils.model_uploader import download_latest_team_weights_if_needed

load_env()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("setup_env")


def setup_directories() -> None:
    """必要なディレクトリ構成を作成・検証します。"""
    logger.info("--- ステップ 1: ディレクトリ構成の整備 ---")
    dirs_to_create = [
        PROJECT_ROOT / "weights",
        PROJECT_ROOT / "weights" / "wav2vec2_best",
        PROJECT_ROOT / "weights" / "wav2vec2_last",
        PROJECT_ROOT / "dataset",
        PROJECT_ROOT / "dataset" / "collected",
        PROJECT_ROOT / "merged_dataset",
        PROJECT_ROOT / "processed_dataset",
        DEFAULT_RECOGNITION_CONFIG.weights_dir,
        DEFAULT_RECOGNITION_CONFIG.wav2vec2_best_model_dir,
    ]
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
    if not DEFAULT_RECOGNITION_CONFIG.log_path.exists():
        DEFAULT_RECOGNITION_CONFIG.log_path.touch()
    logger.info("ディレクトリ構成の整備が完了しました。")


def setup_models(dest_dir: Path | None = None) -> bool:
    """Hugging Face Hub より公開モデル（Wav2Vec2 ＆ CNN）をダウンロード・同期します。"""
    logger.info("--- ステップ 2: Hugging Face からのモデルダウンロード ---")
    target = dest_dir or DEFAULT_RECOGNITION_CONFIG.weights_dir
    ok_w2v = download_latest_team_weights_if_needed(model_type="wav2vec2", weights_dir=target)
    ok_cnn = download_latest_team_weights_if_needed(model_type="cnn", weights_dir=target)

    # プロジェクト直下の weights/ ディレクトリにも複製・同期（ローカル開発の利便性向上）
    project_weights = PROJECT_ROOT / "weights"
    if target != project_weights and target.exists():
        try:
            for item in target.glob("*"):
                if item.is_file():
                    dest_file = project_weights / item.name
                    if not dest_file.exists():
                        with open(item, "rb") as src_f, open(dest_file, "wb") as dst_f:
                            dst_f.write(src_f.read())
                elif item.is_dir():
                    dest_subdir = project_weights / item.name
                    dest_subdir.mkdir(parents=True, exist_ok=True)
                    for subfile in item.glob("*"):
                        if subfile.is_file():
                            sub_dst = dest_subdir / subfile.name
                            if not sub_dst.exists():
                                with open(subfile, "rb") as src_f, open(sub_dst, "wb") as dst_f:
                                    dst_f.write(src_f.read())
        except Exception as e:
            logger.debug("ローカル weights への同期中に例外 (スキップ): %s", e)

    if ok_w2v and ok_cnn:
        logger.info("モデル重みの取得・同期が完了しました。")
        return True
    logger.warning("一部モデルのダウンロードに失敗しました。")
    return False


def setup_dataset(raw_dir: Path | None = None, force: bool = False) -> None:
    """生データが存在する場合、データセット統合と前処理を実行します。"""
    logger.info("--- ステップ 3: データセットの統合 ＆ 前処理 ---")
    target_raw_dir = raw_dir or DEFAULT_RECOGNITION_CONFIG.raw_dataset_dir
    raw_files: list[Path] = (
        list(target_raw_dir.glob("*/*.wav")) if target_raw_dir.exists() else []
    )

    if not raw_files:
        logger.info(
            "生音声データセット (%s) が存在しないため、データセット統合・前処理をスキップします。",
            target_raw_dir,
        )
        return

    logger.info("生音声データ (%d 件) を検出しました。インデックス作成を開始します...", len(raw_files))
    builder = DatasetBuilder()
    index_file = builder.build_index()
    logger.info("データセット統合インデックスを作成しました: %s", index_file)

    processed_dir = DEFAULT_RECOGNITION_CONFIG.processed_dataset_dir
    processed_count = len(list(processed_dir.glob("*/*.wav"))) if processed_dir.exists() else 0

    if processed_count > 0 and not force:
        logger.info(
            "前処理済みデータセット (%s) が既に存在します (%d 件)。(スキップします。再生成は --force を指定)",
            processed_dir,
            processed_count,
        )
    else:
        logger.info(
            "音声前処理を実行中 (%s -> %s)...",
            DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir,
            processed_dir,
        )
        builder.preprocess_dataset(
            input_root=DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir,
            output_root=processed_dir,
        )
        logger.info("音声前処理が完了しました。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VoiceRecognizer one-command environment setup script"
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip downloading models from Hugging Face Hub",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip dataset indexing and preprocessing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and re-preprocessing",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    logger.info("==================================================")
    logger.info("VoiceRecognizer 初期セットアップを開始します")
    logger.info("==================================================")

    setup_directories()

    if not args.skip_models:
        setup_models()
    else:
        logger.info("モデルダウンロードをスキップしました (--skip-models)")

    if not args.skip_dataset:
        setup_dataset(force=args.force)
    else:
        logger.info("データセット処理をスキップしました (--skip-dataset)")

    logger.info("==================================================")
    logger.info("初期セットアップが正常に完了しました！")
    logger.info("すぐに推論を実行できます: uv run python main.py")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
