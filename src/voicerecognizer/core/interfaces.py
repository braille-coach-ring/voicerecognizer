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

    def warmup(self, audio_seconds: float = 1.0) -> None:
        """モデルを事前読み込みし、ダミー波形で推論エンジンをウォームアップする。"""
        import contextlib
        import numpy as np

        sample_rate = getattr(self, "sample_rate", 16000)
        dummy_audio = np.zeros(int(sample_rate * audio_seconds), dtype=np.float32)
        with contextlib.suppress(Exception):
            self.recognize(dummy_audio)
