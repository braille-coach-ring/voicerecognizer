from config import DEFAULT_RECOGNITION_CONFIG, RecognitionConfig
from core.interfaces import RecognitionStrategy
from recognizers.cnn_recognizer import CNNRecognizer
from recognizers.wav2vec2_recognizer import Wav2Vec2Recognizer
from recognizers.whisper_recognizer import WhisperRecognizer


class RecognizerFactory:
    @staticmethod
    def available_strategies() -> tuple[str, ...]:
        return ("cnn", "wav2vec2", "whisper")

    @staticmethod
    def create(
        recognizer_type: str,
        config: RecognitionConfig = DEFAULT_RECOGNITION_CONFIG,
    ) -> RecognitionStrategy:
        normalized_type = recognizer_type.lower()

        if normalized_type == "cnn":
            return CNNRecognizer(
                model_path=config.cnn_weight_path,
                labels=config.labels,
                sample_rate=config.sample_rate,
                target_length_seconds=config.target_length_seconds,
                top_db=config.top_db,
                n_mels=config.n_mels,
            )

        if normalized_type == "wav2vec2":
            return Wav2Vec2Recognizer()

        if normalized_type == "whisper":
            return WhisperRecognizer()

        available = ", ".join(RecognizerFactory.available_strategies())
        raise ValueError(
            f"Unknown recognizer type: {recognizer_type}. Available: {available}"
        )
