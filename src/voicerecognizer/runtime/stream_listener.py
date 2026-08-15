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

from voicerecognizer.config import (
    DEFAULT_AUDIO_CONFIG,
    DEFAULT_PREPROCESS_CONFIG,
    AudioConfig,
    PreprocessConfig,
)
from voicerecognizer.config_labels import to_hiragana, to_katakana, to_romaji
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

    @property
    def hiragana(self) -> str:
        """認識結果をひらがな文字列で取得（例: 'ka' -> 'か'）"""
        return to_hiragana(self.text)

    @property
    def romaji(self) -> str:
        """認識結果をローマ字文字列で取得（例: 'か' -> 'ka'）"""
        return to_romaji(self.text)

    @property
    def katakana(self) -> str:
        """認識結果をカタカナ文字列で取得（例: 'ka' -> 'カ'）"""
        return to_katakana(self.text)


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
        self.speech_settle_seconds = float(
            getattr(audio_config or DEFAULT_AUDIO_CONFIG, "speech_settle_seconds", 0.3)
        )
        self._is_listening = False
        self._is_paused = False
        self._last_emitted_text: str | None = None

    def warmup(self) -> None:
        """認識モデルの事前ロードとウォームアップを実行する。"""
        if hasattr(self.recognizer, "warmup"):
            self.recognizer.warmup()

    def start(self) -> None:
        """聴取を開始（一時停止状態からの即座再開も含む）"""
        self._is_listening = True
        self._is_paused = False
        logger.debug("AudioStreamListener: 聴取を開始しました。")

    def pause(self) -> None:
        """マイク聴取を一時停止する（モデルやマイク接続は保持）"""
        self._is_paused = True
        logger.debug("AudioStreamListener: 聴取を一時停止しました。")

    def close(self) -> None:
        """リスナーを停止し、リソースを解放する。"""
        self._is_listening = False
        self._is_paused = True
        try:
            self.audio_capture.stop()
        except Exception as exc:
            logger.debug("マイクデバイスのクローズ時に例外が発生しました: %s", exc)
        logger.debug("AudioStreamListener: リソースを解放しました。")

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
        マイク音声をリアルタイムに監視し、発話が認識されるたびに
        認識文字列 (str) を yield 返却する非同期ジェネレータ
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

                # 発声開始を検知後、1文字分の発話全体がバッファに入るまで待機
                if self.speech_settle_seconds > 0:
                    await asyncio.sleep(self.speech_settle_seconds)
                    waveform = self.audio_capture.capture_once()

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

                # 発声開始を検知後、1文字分の発話全体がバッファに入るまで待機
                if self.speech_settle_seconds > 0:
                    await asyncio.sleep(self.speech_settle_seconds)
                    waveform = self.audio_capture.capture_once()

                # 候補リストの取得
                if hasattr(self.recognizer, "recognize_with_candidates"):
                    candidates = await asyncio.to_thread(
                        self.recognizer.recognize_with_candidates,
                        waveform,
                        3,
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
