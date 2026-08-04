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
    ):
        cfg = config or DEFAULT_PREPROCESS_CONFIG
        self.silence_threshold = (
            silence_threshold
            if silence_threshold is not None
            else cfg.vad_silence_threshold
        )
        self.min_speech_chunks = max(1, int(cfg.vad_min_speech_chunks))
        self.min_active_ratio = min(1.0, max(0.0, float(cfg.vad_min_active_ratio)))
        self._speech_streak = 0

    def is_speech(self, audio: np.ndarray) -> bool:
        if audio is None:
            logger.warning("入力された音声データがNoneです")
            self._speech_streak = 0
            return False
        if audio.size == 0:
            logger.warning("入力された音声データが空です")
            self._speech_streak = 0
            return False
        abs_audio = np.abs(audio)
        if float(np.max(abs_audio)) < self.silence_threshold:
            logger.info("入力された音声データの最大値が無音閾値未満です")
            self._speech_streak = 0
            return False
        active_ratio = float(np.mean(abs_audio >= self.silence_threshold))
        if active_ratio < self.min_active_ratio:
            logger.info(
                "閾値超えサンプル率が不足しています (%.4f < %.4f)",
                active_ratio,
                self.min_active_ratio,
            )
            self._speech_streak = 0
            return False
        self._speech_streak += 1
        if self._speech_streak < self.min_speech_chunks:
            logger.info(
                "発話候補を検知しましたが確定待ちです (%d/%d)",
                self._speech_streak,
                self.min_speech_chunks,
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
