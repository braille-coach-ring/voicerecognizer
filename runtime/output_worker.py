"""
認識結果の出力通知および音声データ・認識ログ (metadata.csv) の保存を管理するモジュール。
"""

from collections.abc import Callable
from pathlib import Path
from queue import Empty, Queue
from threading import Event
import numpy as np
import soundfile as sf

from config import DEFAULT_AUDIO_CONFIG, DEFAULT_RECOGNITION_CONFIG
from utils.machine_id import get_machine_id

import logging
logger = logging.getLogger(__name__)


class OutputWorker:
    """
    認識結果の表示および録音波形・メタデータ (metadata.csv) の非同期保存を担当するクラス。

    Attributes:
        output (Callable[[str], None]): 認識結果を出力するハンドラ関数 (デフォルト: print)
        save_dir (Path): 音声およびメタデータを保存するPC固有の匿名ディレクトリ
    """

    def __init__(
        self,
        output: Callable[[str], None] | None = None,
        save_dir: Path | str | None = None,
    ):
        self.output = output or print
        self.save_dir = (
            Path(save_dir)
            if save_dir
            else DEFAULT_RECOGNITION_CONFIG.collected_dataset_dir / get_machine_id()
        )
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def emit(self, text: str) -> None:
        self.output(text)

    def run(self, input_queue: Queue, stop_event: Event) -> None:
        while not stop_event.is_set():
            try:
                text = input_queue.get(timeout=0.1)
            except Empty:
                continue

            self.emit(text)

    def save(
        self,
        audio_data,
        predicted_text: str,
        timestamp,
        ground_truth: str = "",
        sample_rate: int = DEFAULT_AUDIO_CONFIG.sample_rate,
    ) -> None:
        if self.save_dir is None:
            logger.warning("保存先ディレクトリが指定されていません。保存をスキップします。")
            return

        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.error("保存先ディレクトリの作成に失敗しました: %s", self.save_dir, exc_info=True)
            return

        time_str = str(timestamp).replace(".", "_")
        file_name = f"{time_str}.wav"

        # 1. 認識テキストログの保存 (4カラム: timestamp, filename, predicted_text, ground_truth)
        log_file = self.save_dir / "metadata.csv"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{time_str},{file_name},{predicted_text},{ground_truth}\n")
            logger.info("テキストログを保存しました: %s", log_file.name)
        except Exception:
            logger.error("テキストログファイルの書き込みに失敗しました: %s", log_file, exc_info=True)

        # 2. 音声波形（.wav）の保存
        if audio_data is not None and isinstance(audio_data, np.ndarray) and audio_data.size > 0:
            wav_path = self.save_dir / file_name
            try:
                sf.write(wav_path, audio_data, sample_rate)
                logger.info("音声波形ファイルを保存しました: %s", file_name)
            except Exception:
                logger.error("音声波形ファイル(.wav)の保存に失敗しました: %s", wav_path, exc_info=True)
