from collections import deque
from threading import Lock
import numpy as np
import sounddevice as sd
from config import AudioConfig, DEFAULT_AUDIO_CONFIG


class AudioCapture:
    """マイクデバイスを常時開きっぱなしにし、最新の音声波形をリアルタイムにバッファリングするクラス"""

    def __init__(self, config: AudioConfig | None = None):
        cfg = config or DEFAULT_AUDIO_CONFIG
        self.sample_rate: int = cfg.sample_rate
        self.window_seconds: float = float(getattr(cfg, "window_seconds", 1.0))
        
        # 直近 N 秒分のサンプル数を保持するリングバッファ
        self._max_samples: int = int(self.window_seconds * self.sample_rate)
        self._buffer: deque[float] = deque(maxlen=self._max_samples)
        self._lock: Lock = Lock()
        self._stream: sd.InputStream | None = None

        # インスタンス生成時にマイクストリームを開始
        self.start()

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
        """サウンドカードから非同期に音声フレームが届いたときに自動実行されるコールバック"""
        if status:
            pass  # オーバーフロー等の警告ログ処理（必要に応じて）

        # 1次元のfloat32配列に変換してリングバッファに追記
        samples = indata[:, 0].tolist()
        with self._lock:
            self._buffer.extend(samples)

    def start(self) -> None:
        """マイクストリームを開き、バックグラウンドで連続録音を開始する"""
        if self._stream is None or not self._stream.active:
            # バッファの初期化（ゼロ埋め）
            with self._lock:
                self._buffer.clear()
                self._buffer.extend([0.0] * self._max_samples)

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
                blocksize=int(self.sample_rate * 0.05),  # 50msごとにデータ受取
            )
            self._stream.start()

    def stop(self) -> None:
        """マイクストリームを安全に停止・クローズする"""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def capture_once(self, duration: float | None = None) -> np.ndarray:
        """常時録音されているバッファから、直近の音声波形（最新データ）を即座に取得する"""
        # まだ一度もストリームが開始されていなければ開始
        if self._stream is None or not self._stream.active:
            self.start()
            # 初回だけ少し待機
            sd.sleep(500)

        with self._lock:
            arr = np.array(self._buffer, dtype=np.float32)

        req_samples = int((duration or self.window_seconds) * self.sample_rate)
        if len(arr) < req_samples:
            # 不足分はパディング
            arr = np.pad(arr, (req_samples - len(arr), 0))
        else:
            arr = arr[-req_samples:]

        return arr

    def __del__(self):
        self.stop()
