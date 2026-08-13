"""
Async & Non-blocking Recognition Pipeline Runner (Issue #9)

役割:
  音声ストリーミング入力および文字認識処理を非同期 (async/await) かつ
  メインスレッドをブロックせずに並列処理するための非同期パイプライン機構。
"""

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable
from typing import Any

from core.interfaces import RecognitionStrategy

logger = logging.getLogger(__name__)


class AsyncRecognitionPipeline:
    """非同期・ノンブロッキング音声認識パイプライン"""

    def __init__(self, strategy: RecognitionStrategy):
        self.strategy = strategy
        self._is_running = False

    async def process_stream(
        self,
        audio_stream: AsyncGenerator[Any],
        on_result: Callable[[str], None] | None = None,
    ) -> None:
        """非同期音声ストリームを読み込み、ノンブロッキングで認識結果を順次呼び出す"""
        self._is_running = True
        logger.info("非同期認識パイプラインを開始しました。")

        async for chunk in audio_stream:
            if not self._is_running:
                break
            result = await self.strategy.recognize_async(chunk)
            if on_result and result:
                if asyncio.iscoroutinefunction(on_result):
                    await on_result(result)
                else:
                    on_result(result)

    def stop(self) -> None:
        self._is_running = False
        logger.info("非同期認識パイプラインを停止要求しました。")
