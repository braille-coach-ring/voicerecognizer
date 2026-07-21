from abc import ABC, abstractmethod
from typing import Any


class RecognitionStrategy(ABC):
    @abstractmethod
    def recognize(self, audio: Any) -> str:
        pass
