from threading import Lock
import numpy as np
import sounddevice as sd
from config import AudioConfig, DEFAULT_AUDIO_CONFIG

import logging

logger = logging.getLogger(__name__)


class AudioCapture:
    """マイクデバイスを常時開きっぱなしにし、最新の音声波形をリアルタイムにバッファリングするクラス"""

    def __init__(self, config: AudioConfig | None = None):
        cfg = config or DEFAULT_AUDIO_CONFIG
        self.sample_rate: int = cfg.sample_rate
        self.window_seconds: float = float(getattr(cfg, "window_seconds", 1.0))

        self.channels: int = cfg.channels
        self.blocksize_seconds: float = float(getattr(cfg, "callback_blocksize_seconds", 0.05))
        self.warmup_sleep_ms: int = int(getattr(cfg, "warmup_sleep_ms", 500))

        # 直近 N 秒分のサンプル数を保持する NumPy 高速リングバッファ
        self._max_samples: int = int(self.window_seconds * self.sample_rate)
        self._buffer: np.ndarray = np.zeros(self._max_samples, dtype=np.float32)
        self._write_pos: int = 0
        self._lock: Lock = Lock()
        self._stream: sd.InputStream | None = None

        # インスタンス生成時にマイクストリームを開始
        self.start()

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags
    ) -> None:
        """サウンドカードから非同期に音声フレームが届いたときに自動実行されるコールバック"""
        if status:
            pass  # オーバーフロー等の警告ログ処理

        samples = indata[:, 0].astype(np.float32)
        n = len(samples)
        with self._lock:
            if self._write_pos + n <= self._max_samples:
                self._buffer[self._write_pos : self._write_pos + n] = samples
                self._write_pos += n
            else:
                first = self._max_samples - self._write_pos
                self._buffer[self._write_pos :] = samples[:first]
                self._buffer[: n - first] = samples[first:]
                self._write_pos = n - first

    def start(self) -> None:
        """マイクストリームを開き、バックグラウンドで連続録音を開始する"""
        if self._stream is None or not self._stream.active:
            # バッファの初期化（ゼロ埋め）
            with self._lock:
                self._buffer.fill(0.0)
                self._write_pos = 0

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                callback=self._audio_callback,
                blocksize=int(self.sample_rate * self.blocksize_seconds),
            )
            self._stream.start()
            logger.info("マイクストリームを開始しました")

    def stop(self) -> None:
        """マイクストリームを安全に停止・クローズする"""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("マイクストリームを停止しました")

    def capture_once(self, duration: float | None = None) -> np.ndarray:
        """常時録音されているバッファから、直近の音声波形（最新データ）を即座に取得する"""
        if self._stream is None or not self._stream.active:
            self.start()
            sd.sleep(self.warmup_sleep_ms)

        with self._lock:
            arr = np.concatenate((self._buffer[self._write_pos :], self._buffer[: self._write_pos]))

        req_samples = int((duration or self.window_seconds) * self.sample_rate)
        if len(arr) < req_samples:
            arr = np.pad(arr, (req_samples - len(arr), 0))
        else:
            arr = arr[-req_samples:]

        return arr.copy()

    def __del__(self):
        self.stop()
