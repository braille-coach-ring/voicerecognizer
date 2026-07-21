from typing import Any

from core.interfaces import RecognitionStrategy


class VoiceRecognizer:
    def __init__(self, strategy: RecognitionStrategy):
        self._strategy = strategy

    def recognize(self, audio: Any) -> str:
        return self._strategy.recognize(audio)

    def set_strategy(self, strategy: RecognitionStrategy) -> None:
        self._strategy = strategy
