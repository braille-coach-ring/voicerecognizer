from abc import ABC, abstractmethod
from typing import Any


class RecognitionStrategy(ABC):
    @abstractmethod
    def recognize(self, audio: Any) -> str:
        pass

    async def recognize_async(self, audio: Any) -> str:
        """非同期・ノンブロッキングでの認識処理のデフォルト実装"""
        import asyncio

        return await asyncio.to_thread(self.recognize, audio)

