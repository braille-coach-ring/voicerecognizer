"""
Audio Stream Listener (Issue #9)

マイクからのリアルタイム音声ストリームを常時オープンし、
発声が認識されたタイミングでのみ非同期 (async for) で「ひらがな文字」を返却するクラス。

ライフサイクル:
  - start(): 聴取の開始（一時停止状態からの再開も待機時間ゼロで行う）
  - pause(): マイクとモデルは保持したまま、聴取・判定処理のみ一時停止（再開コストゼロ）
  - close(): マイクデバイスを完全にシャットダウンし、全リソースを解放・破棄
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from voicerecognizer.config import DEFAULT_AUDIO_CONFIG, DEFAULT_PREPROCESS_CONFIG, AudioConfig, PreprocessConfig
from voicerecognizer.core.exceptions import DeviceNotFoundError
from voicerecognizer.core.interfaces import RecognitionStrategy
from voicerecognizer.runtime.audio_capture import AudioCapture
from voicerecognizer.runtime.vad import VoiceActivityDetector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecognitionResult:
    """文字認識の詳細結果オブジェクト"""

    text: str
    confidence: float
    top3_candidates: list[tuple[str, float]] = field(default_factory=list)
    timing_stats: dict[str, Any] = field(default_factory=dict)


class AudioStreamListener:
    """リアルタイム音声ストリーミング非同期聴取器"""

    def __init__(
        self,
        recognizer: RecognitionStrategy | None = None,
        min_confidence: float = 0.35,
        poll_interval_seconds: float = 0.08,
        audio_config: AudioConfig | None = None,
        preprocess_config: PreprocessConfig | None = None,
    ):
        from voicerecognizer.recognizers.wav2vec2_recognizer import Wav2Vec2Recognizer

        self.recognizer = recognizer or Wav2Vec2Recognizer()
        self.min_confidence = min_confidence
        self.poll_interval = poll_interval_seconds
        try:
            self.audio_capture = AudioCapture(config=audio_config or DEFAULT_AUDIO_CONFIG)
        except Exception as exc:
            raise DeviceNotFoundError(f"マイクデバイスの初期化に失敗しました: {exc}") from exc

        self.vad = VoiceActivityDetector(config=preprocess_config or DEFAULT_PREPROCESS_CONFIG)
        self._is_listening = False
        self._is_paused = False
        self._last_emitted_text: str | None = None

    def start(self) -> None:
        """聴取を開始（一時停止状態からの即座再開も含む）"""
        self._is_paused = False
        self._is_listening = True
        if self.audio_capture:
            self.audio_capture.start()
        logger.info("🎤 リアルタイム非同期リスニングを開始/再開しました。")

    def pause(self) -> None:
        """マイクとモデルは保持したまま、聴取のみ一時停止（再開コストゼロ）"""
        self._is_paused = True
        logger.info("⏸️ リアルタイム非同期リスニングを一時停止しました（モデル・マイクは保持）。")

    def close(self) -> None:
        """マイクを完全に閉じ、全リソースを解放・終了（Context Manager 対応）"""
        self._is_listening = False
        self._is_paused = True
        if self.audio_capture:
            self.audio_capture.stop()
        logger.info("🛑 リアルタイム非同期リスニングをクローズ・リソース解放しました。")

    def __enter__(self) -> "AudioStreamListener":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    async def __aenter__(self) -> "AudioStreamListener":
        self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    async def listen(self) -> AsyncGenerator[str]:
        """
        マイクからの録音ストリームを非同期監視し、
        文字が認識されたタイミングでのみ「ひらがな文字」を yield 返却します。
        """
        self.start()
        try:
            while self._is_listening:
                await asyncio.sleep(self.poll_interval)

                if self._is_paused:
                    continue

                waveform = self.audio_capture.capture_once()
                if len(waveform) == 0:
                    continue

                # VAD（音声発声検出）
                if not self.vad.is_speech(waveform):
                    # 無音区間に入ったら重複抑止状態をリセット（次の発声を新規として受ける）
                    self._last_emitted_text = None
                    continue

                # 非同期スレッドで推論を実行（メインのイベントループをブロックしない）
                result_text = await self.recognizer.recognize_async(waveform)
                confidence = getattr(self.recognizer, "last_confidence", 1.0)

                # 確信度が閾値を超えている場合のみ emit
                if (
                    result_text
                    and (confidence is None or confidence >= self.min_confidence)
                    and result_text != self._last_emitted_text
                ):
                    self._last_emitted_text = result_text
                    yield result_text
        finally:
            pass

    async def listen_details(self) -> AsyncGenerator[RecognitionResult]:
        """
        確信度スコア、Top-3 候補、タイミング指標を含む
        詳細な RecognitionResult オブジェクトを yield 返却する高度な非同期ジェネレータ
        """
        self.start()
        try:
            while self._is_listening:
                await asyncio.sleep(self.poll_interval)

                if self._is_paused:
                    continue

                waveform = self.audio_capture.capture_once()
                if len(waveform) == 0:
                    continue

                if not self.vad.is_speech(waveform):
                    self._last_emitted_text = None
                    continue

                # 候補リストの取得
                if hasattr(self.recognizer, "recognize_with_candidates"):
                    candidates = await asyncio.to_thread(
                        self.recognizer.recognize_with_candidates, waveform, 3
                    )
                    result_text = candidates[0][0] if candidates else ""
                    confidence = candidates[0][1] if candidates else 0.0
                else:
                    result_text = await self.recognizer.recognize_async(waveform)
                    confidence = getattr(self.recognizer, "last_confidence", 1.0)
                    candidates = [(result_text, confidence)]

                timing_stats = getattr(self.recognizer, "last_timing_stats", {})

                if (
                    result_text
                    and (confidence is None or confidence >= self.min_confidence)
                    and result_text != self._last_emitted_text
                ):
                    self._last_emitted_text = result_text
                    yield RecognitionResult(
                        text=result_text,
                        confidence=float(confidence) if confidence is not None else 0.0,
                        top3_candidates=candidates,
                        timing_stats=timing_stats,
                    )
        finally:
            pass
