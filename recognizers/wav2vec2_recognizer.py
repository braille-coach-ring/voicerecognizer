from typing import Any

from core.interfaces import RecognitionStrategy


class Wav2Vec2Recognizer(RecognitionStrategy):
    def __init__(
        self, processor: Any = None, onnx_session: Any = None, decoder: Any = None
    ):
        self.processor = processor
        self.onnx_session = onnx_session
        self.decoder = decoder

    def recognize(self, audio: Any) -> str:
        raise NotImplementedError(
            "Wav2Vec2Recognizer is reserved for future fine-tuning."
        )
