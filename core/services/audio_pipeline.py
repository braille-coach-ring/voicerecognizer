from core.services.voice_recognizer import VoiceRecognizer
from runtime.audio_capture import AudioCapture
from runtime.output_worker import OutputWorker
from runtime.vad import VoiceActivityDetector
from config import DEFAULT_AUDIO_CONFIG
from datetime import datetime
import numpy as np
import time


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

    def run(self, audio=None):
        if audio is None:
            print("録音開始")
            audio = self.audio_capture.capture_once()
            print("録音終了")

            max_vol = float(np.max(np.abs(audio))) if audio is not None and audio.size else 0.0
            is_sp = self.vad.is_speech(audio)
            print(f"VAD判定中 (録音最大音量: {max_vol:.4f} / 判定閾値: {self.vad.silence_threshold}) -> 発話検知: {is_sp}")

            if not is_sp:
                print("音声ではない")
                return None

        print("認識開始")
        text = self.recognizer.recognize(audio)
        print("認識終了")

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
        print("音声を待機中... マイクに向かってお話しください（中断するには Ctrl+C）")
        while True:
            try:
                result = self.run()
                if result is not None:
                    print(f"認識成功: {result}")
                    return result
                time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n音声待機を停止しました。")
                return None
