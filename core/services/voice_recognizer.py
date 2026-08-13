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

        stats = getattr(self._strategy, "last_timing_stats", {}).copy()
        if not stats:
            stats = {"total_latency_ms": (end_time - start_time) * 1000.0}
        self.last_timing_stats = stats

        logger.info(
            "推論終了: %s (所要時間: %.2f ms)", end_time, stats.get("total_latency_ms", 0.0)
        )
        return text

    @property
    def strategy(self) -> RecognitionStrategy:
        return self._strategy

    def set_strategy(self, strategy: RecognitionStrategy) -> None:
        self._strategy = strategy
