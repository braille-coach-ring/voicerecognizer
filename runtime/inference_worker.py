from queue import Empty, Queue
from threading import Event

from core.services.voice_recognizer import VoiceRecognizer


class InferenceWorker:
    def __init__(self, recognizer: VoiceRecognizer):
        self.recognizer = recognizer

    def recognize(self, audio) -> str:
        return self.recognizer.recognize(audio)

    def run(self, input_queue: Queue, output_queue: Queue, stop_event: Event) -> None:
        while not stop_event.is_set():
            try:
                audio = input_queue.get(timeout=0.1)
            except Empty:
                continue

            output_queue.put(self.recognize(audio))
