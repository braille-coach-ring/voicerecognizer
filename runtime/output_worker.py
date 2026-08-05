from collections.abc import Callable
from pathlib import Path
from queue import Empty, Queue
from threading import Event
import soundfile as sf

from config import DEFAULT_AUDIO_CONFIG, DEFAULT_RECOGNITION_CONFIG
from utils.machine_id import get_machine_id

import logging
logger = logging.getLogger(__name__)


class OutputWorker:
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

        # 1. 認識テキストログの保存
        log_file = self.save_dir / "metadata.csv"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{time_str},{file_name},{predicted_text},\n")
            logger.info("テキストログを保存しました: %s", log_file.name)
        except Exception:
            logger.error("テキストログファイルの書き込みに失敗しました: %s", log_file, exc_info=True)

        # 2. 音声波形（.wav）の保存
        if audio_data is not None and len(audio_data) > 0:
            wav_path = self.save_dir / file_name
            try:
                sf.write(wav_path, audio_data, sample_rate)
                logger.info("音声波形ファイルを保存しました: %s", file_name)
            except Exception:
                logger.error("音声波形ファイル(.wav)の保存に失敗しました: %s", wav_path, exc_info=True)
