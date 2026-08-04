from collections.abc import Callable
from pathlib import Path
from queue import Empty, Queue
from threading import Event
import soundfile as sf

from config import DEFAULT_RECOGNITION_CONFIG


class OutputWorker:
    def __init__(
        self,
        output: Callable[[str], None] | None = None,
        save_dir: Path | str | None = None,
    ):
        self.output = output or print
        self.save_dir = Path(save_dir) if save_dir else DEFAULT_RECOGNITION_CONFIG.collected_dataset_dir
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
        sample_rate: int = 16000,
    ) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        time_str = str(timestamp).replace(".", "_")

        # 1. 認識テキストログの保存
        log_file = self.save_dir / "predicted_text.txt"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{time_str},{predicted_text}\n")

        # 2. 音声波形（.wav）の保存
        if audio_data is not None and len(audio_data) > 0:
            wav_file = self.save_dir / f"{time_str}_{predicted_text}.wav"
            sf.write(wav_file, audio_data, sample_rate)
