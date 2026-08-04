import logging
from typing import Any

from core.interfaces import RecognitionStrategy

logger = logging.getLogger(__name__)


class Wav2Vec2Recognizer(RecognitionStrategy):
    def __init__(
        self, processor: Any = None, onnx_session: Any = None, decoder: Any = None
    ):
        self.processor = processor
        self.onnx_session = onnx_session
        self.decoder = decoder
        logger.info("Wav2Vec2Recognizer インスタンスを初期化しました (実装予約中)")

    def recognize(self, audio: Any) -> str:
        logger.error("Wav2Vec2Recognizer.recognize が呼び出されましたが、まだ実装されていません")
        raise NotImplementedError(
            "Wav2Vec2Recognizer is reserved for future fine-tuning."
        )

