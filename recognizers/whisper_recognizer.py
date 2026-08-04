import logging
from typing import Any

from core.interfaces import RecognitionStrategy

logger = logging.getLogger(__name__)


class WhisperRecognizer(RecognitionStrategy):
    def __init__(self, model: Any = None, decoder: Any = None):
        self.model = model
        self.decoder = decoder
        logger.info("WhisperRecognizer インスタンスを初期化しました (実装予約中)")

    def recognize(self, audio: Any) -> str:
        logger.error("WhisperRecognizer.recognize が呼び出されましたが、まだ実装されていません")
        raise NotImplementedError(
            "WhisperRecognizer is reserved for future implementation."
        )

