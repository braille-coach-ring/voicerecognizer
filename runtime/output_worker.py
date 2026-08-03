from collections.abc import Callable
from queue import Empty, Queue
from threading import Event


class OutputWorker:
    def __init__(self, output: Callable[[str], None] | None = None):
        self.output = output or print

    def emit(self, text: str) -> None:
        self.output(text)

    def run(self, input_queue: Queue, stop_event: Event) -> None:
        while not stop_event.is_set():
            try:
                text = input_queue.get(timeout=0.1)
            except Empty:
                continue

            self.emit(text)
