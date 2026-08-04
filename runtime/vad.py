from queue import Empty, Queue
from threading import Event
import numpy as np

from config import DEFAULT_PREPROCESS_CONFIG, PreprocessConfig


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

    def is_speech(self, audio: np.ndarray) -> bool:
        return bool(
            audio is not None
            and audio.size
            and np.max(np.abs(audio)) >= self.silence_threshold
        )

    def run(self, input_queue: Queue, output_queue: Queue, stop_event: Event) -> None:
        while not stop_event.is_set():
            try:
                audio = input_queue.get(timeout=0.1)
            except Empty:
                continue

            if self.is_speech(audio):
                output_queue.put(audio)
