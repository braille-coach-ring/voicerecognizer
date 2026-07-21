from typing import Any

from core.interfaces import RecognitionStrategy


class WhisperRecognizer(RecognitionStrategy):
    def __init__(self, model: Any = None, decoder: Any = None):
        self.model = model
        self.decoder = decoder

    def recognize(self, audio: Any) -> str:
        raise NotImplementedError("WhisperRecognizer is reserved for future implementation.")
