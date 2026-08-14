import logging
import time
from datetime import datetime

import numpy as np

from voicerecognizer.config import DEFAULT_AUDIO_CONFIG
from voicerecognizer.core.services.voice_recognizer import VoiceRecognizer
from voicerecognizer.runtime.audio_capture import AudioCapture
from voicerecognizer.runtime.output_worker import OutputWorker
from voicerecognizer.runtime.vad import VoiceActivityDetector

logger = logging.getLogger(__name__)


class AudioPipeline:
    def __init__(
        self,
        recognizer: VoiceRecognizer,
        audio_capture: AudioCapture | None = None,
        vad: VoiceActivityDetector | None = None,
        output_worker: OutputWorker | None = None,
    ):
        self.recognizer = recognizer
        self.audio_capture = audio_capture or AudioCapture(DEFAULT_AUDIO_CONFIG)
        self.vad = vad or VoiceActivityDetector()
        self.output_worker = output_worker or OutputWorker()
        self.datetime = datetime
        logger.info(
            "AudioPipelineの初期化が完了: %s / %s / %s / %s",
            self.recognizer,
            self.audio_capture,
            self.vad,
            self.output_worker,
        )

    def run(self, audio=None):
        if audio is None:
            logger.info("録音開始: %s", time.perf_counter())
            audio = self.audio_capture.capture_once()
            logger.info("録音終了: %s", time.perf_counter())

            max_vol = float(np.max(np.abs(audio))) if audio is not None and audio.size else 0.0
            rms_vol = float(np.sqrt(np.mean(audio**2))) if audio is not None and audio.size else 0.0
            is_sp = self.vad.is_speech(audio)
            logger.info(
                "VAD判定中 (録音最大音量: %.4f/閾値: %.4f, RMS音量: %.4f/閾値: %.4f) -> 発話検知: %s",
                max_vol,
                self.vad.silence_threshold,
                rms_vol,
                getattr(self.vad, "rms_threshold", 0.0),
                is_sp,
            )

            if not is_sp:
                logger.info("音声ではない")
                return None

        text = self.recognizer.recognize(audio)

        self.output_worker.save(
            audio_data=audio,
            predicted_text=text,
            timestamp=self.datetime.now().timestamp(),
            sample_rate=self.audio_capture.sample_rate,
        )
        return text

    def run_once(self) -> str | None:
        return self.run()

    def run_until_speech(self) -> str | None:
        logger.info("音声を待機中... マイクに向かってお話しください（中断するには Ctrl+C）")
        while True:
            try:
                result = self.run()
                if result is not None:
                    logger.info("認識成功: %s", result)
                    return result
                time.sleep(DEFAULT_AUDIO_CONFIG.chunk_seconds)
            except KeyboardInterrupt:
                logger.info("\n音声待機を停止しました。")
                return None

    def capture_until_speech(self) -> tuple[np.ndarray, str, dict] | None:
        """
        発話が検出されるまでマイク入力をループ監視し、
        生の録音波形 (np.ndarray)、推論予測テキスト (str)、および計測統計情報 (dict) のタプルを返します。
        """
        from datetime import timedelta

        logger.info("音声を待機中... マイクに向かってお話しください（Ctrl+Cで終了）")
        settle_sec = DEFAULT_AUDIO_CONFIG.speech_settle_seconds
        while True:
            try:
                chunk = self.audio_capture.capture_once()
                if chunk is not None and chunk.size > 0 and self.vad.is_speech(chunk):
                    speech_detected_dt = self.datetime.now()
                    # VADが発声を検知した → 発声がバッファに完全に収まるよう少し待つ
                    time.sleep(settle_sec)
                    # 待機後にリングバッファから最新の1秒を再取得
                    raw_audio = self.audio_capture.capture_once()
                    predicted_text = self.recognizer.recognize(raw_audio)

                    stats = getattr(self.recognizer, "last_timing_stats", {}).copy()
                    stats["detected_time"] = speech_detected_dt

                    # raw_audio の長さ (秒)
                    total_audio_sec = len(raw_audio) / self.audio_capture.sample_rate
                    onset_sec = stats.get("onset_ms", 0.0) / 1000.0
                    offset_sec = stats.get("offset_ms", 0.0) / 1000.0

                    # settle_sec の待機込みでの raw_audio の先頭時刻
                    raw_start_dt = speech_detected_dt + timedelta(seconds=settle_sec - total_audio_sec)
                    stats["speech_start_time"] = raw_start_dt + timedelta(seconds=onset_sec)
                    stats["speech_end_time"] = raw_start_dt + timedelta(seconds=offset_sec)

                    return raw_audio, predicted_text, stats
                time.sleep(DEFAULT_AUDIO_CONFIG.chunk_seconds)
            except KeyboardInterrupt:
                logger.info("\n音声待機を停止しました。")
                return None
