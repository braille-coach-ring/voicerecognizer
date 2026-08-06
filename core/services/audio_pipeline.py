from core.services.voice_recognizer import VoiceRecognizer
from runtime.audio_capture import AudioCapture
from runtime.output_worker import OutputWorker
from runtime.vad import VoiceActivityDetector
from config import DEFAULT_AUDIO_CONFIG
from datetime import datetime
import numpy as np
import time

import logging
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
        logger.info(f"AudioPipelineの初期化が完了: {self.recognizer} / {self.audio_capture} / {self.vad} / {self.output_worker}")

    def run(self, audio=None):
        if audio is None:
            logger.info("録音開始: %s", time.perf_counter())
            audio = self.audio_capture.capture_once()
            logger.info("録音終了: %s", time.perf_counter())

            max_vol = (
                float(np.max(np.abs(audio)))
                if audio is not None and audio.size
                else 0.0
            )
            rms_vol = (
                float(np.sqrt(np.mean(audio**2)))
                if audio is not None and audio.size
                else 0.0
            )
            is_sp = self.vad.is_speech(audio)
            logger.info(
                f"VAD判定中 (録音最大音量: {max_vol:.4f}/閾値: {self.vad.silence_threshold:.4f}, RMS音量: {rms_vol:.4f}/閾値: {getattr(self.vad, 'rms_threshold', 0.0):.4f}) -> 発話検知: {is_sp}"
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
                    logger.info(f"認識成功: {result}")
                    return result
                time.sleep(DEFAULT_AUDIO_CONFIG.chunk_seconds)
            except KeyboardInterrupt:
                logger.info("\n音声待機を停止しました。")
                return None

    def capture_until_speech(self) -> tuple[np.ndarray, str] | None:
        """
        発話が検出されるまでマイク入力をループ監視し、
        前処理済みの綺麗な録音波形 (np.ndarray) と推論予測テキスト (str) のペアを返します。
        """
        logger.info("音声を待機中... マイクに向かってお話しください（Ctrl+Cで終了）")
        while True:
            try:
                audio = self.audio_capture.capture_once()
                if audio is not None and audio.size > 0:
                    if self.vad.is_speech(audio):
                        predicted_text = self.recognizer.recognize(audio)
                        strategy = getattr(self.recognizer, "strategy", getattr(self.recognizer, "_strategy", None))
                        if strategy and hasattr(strategy, "audio_preprocessor"):
                            preprocessed_audio = strategy.audio_preprocessor.preprocess_waveform(audio)
                        return preprocessed_audio, predicted_text
                time.sleep(DEFAULT_AUDIO_CONFIG.chunk_seconds)
            except KeyboardInterrupt:
                logger.info("\n音声待機を停止しました。")
                return None
