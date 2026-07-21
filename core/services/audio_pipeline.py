from core.services.voice_recognizer import VoiceRecognizer
from runtime.audio_capture import AudioCapture
from runtime.output_worker import OutputWorker
from runtime.vad import VoiceActivityDetector


class AudioPipeline:
    def __init__(
        self,
        recognizer: VoiceRecognizer,
        audio_capture: AudioCapture | None = None,
        vad: VoiceActivityDetector | None = None,
        output_worker: OutputWorker | None = None,
    ):
        self.recognizer = recognizer
        self.audio_capture = audio_capture or AudioCapture()
        self.vad = vad or VoiceActivityDetector()
        self.output_worker = output_worker or OutputWorker()

    def run(self, audio=None):
        if audio is None:
            print("録音開始")
            audio = self.audio_capture.capture_once()
            print("録音終了")

            print("VAD判定中")
            print(self.vad.is_speech(audio))

            if not self.vad.is_speech(audio):
                print("音声ではない")
                return None

        print("認識開始")
        text = self.recognizer.recognize(audio)
        print("認識終了")

        self.output_worker.emit(text)
        return text

    def run_once(self) -> str | None:
        return self.run()
