from queue import Empty, Queue
from threading import Event
import numpy as np

from config import DEFAULT_PREPROCESS_CONFIG, PreprocessConfig

import logging
logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    def __init__(
        self,
        config: PreprocessConfig | None = None,
        silence_threshold: float | None = None,
        rms_threshold: float | None = None,
    ):
        cfg = config or DEFAULT_PREPROCESS_CONFIG
        self.silence_threshold = (
            silence_threshold
            if silence_threshold is not None
            else cfg.vad_silence_threshold
        )
        self.rms_threshold = (
            rms_threshold
            if rms_threshold is not None
            else getattr(cfg, "vad_rms_threshold", 0.002)
        )

    def is_speech(self, audio: np.ndarray) -> bool:
        if audio is None:
            logger.warning("入力された音声データがNoneです")
            return False
        if audio.size == 0:
            logger.warning("入力された音声データが空です")
            return False

        max_vol = float(np.max(np.abs(audio)))
        rms_vol = float(np.sqrt(np.mean(audio**2)))

        if max_vol < self.silence_threshold:
            logger.info(
                f"入力された音声データの最大値({max_vol:.4f})が無音閾値({self.silence_threshold:.4f})未満です"
            )
            return False

        if rms_vol < self.rms_threshold:
            logger.info(
                f"入力された音声データのRMS値({rms_vol:.4f})がRMS閾値({self.rms_threshold:.4f})未満です（ノイズスパイクと判定）"
            )
            return False

        return True

    def run(self, input_queue: Queue, output_queue: Queue, stop_event: Event) -> None:
        while not stop_event.is_set():
            try:
                audio = input_queue.get(timeout=0.1)
            except Empty:
                continue

            if self.is_speech(audio):
                output_queue.put(audio)
