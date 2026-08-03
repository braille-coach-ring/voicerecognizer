from queue import Queue
from threading import Event

import numpy as np
import sounddevice as sd


class AudioCapture:
    def __init__(self, sample_rate: int = 16000, chunk_seconds: float = 1.0):
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds

    def capture_once(self) -> np.ndarray:
        audio = sd.rec(
            int(self.chunk_seconds * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        return audio.reshape(-1)

    def run(self, output_queue: Queue, stop_event: Event) -> None:
        while not stop_event.is_set():
            output_queue.put(self.capture_once())
