from typing import Any

from core.interfaces import RecognitionStrategy

import time
import logging
logger = logging.getLogger(__name__)


class VoiceRecognizer:
    def __init__(self, strategy: RecognitionStrategy):
        self._strategy = strategy
        logger.info("VoiceRecognizerの初期化が完了: %s", self._strategy)

    def recognize(self, audio: Any) -> str:
        start_time = time.perf_counter()
        logger.info("推論開始: %s", start_time)
        text = self._strategy.recognize(audio)
        end_time = time.perf_counter()
        logger.info("推論終了: %s", end_time)
        logger.info("推論時間: %s", end_time - start_time)
        return text

    def set_strategy(self, strategy: RecognitionStrategy) -> None:
        self._strategy = strategy
