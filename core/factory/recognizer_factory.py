import logging

from config import DEFAULT_RECOGNITION_CONFIG, RecognitionConfig, RecognizerType
from core.interfaces import RecognitionStrategy
from recognizers.cnn_recognizer import CNNRecognizer
from recognizers.wav2vec2_recognizer import Wav2Vec2Recognizer
from recognizers.whisper_recognizer import WhisperRecognizer

logger = logging.getLogger(__name__)


class RecognizerFactory:
    @staticmethod
    def available_strategies() -> tuple[RecognizerType, ...]:
        return ("cnn", "wav2vec2", "whisper")

    @staticmethod
    def create(
        recognizer_type: RecognizerType,
        config: RecognitionConfig = DEFAULT_RECOGNITION_CONFIG,
    ) -> RecognitionStrategy:
        logger.info("RecognizerFactory.create: %s", recognizer_type)

        if recognizer_type == "cnn":
            return CNNRecognizer(
                model_path=config.cnn_weight_path,
                labels=config.labels,
                sample_rate=config.sample_rate,
                target_length_seconds=config.target_length_seconds,
                top_db=config.top_db,
                n_mels=config.n_mels,
            )

        if recognizer_type == "wav2vec2":
            return Wav2Vec2Recognizer(
                model_path=config.wav2vec2_best_model_dir,
                labels=config.labels,
                sample_rate=config.sample_rate,
                target_length_seconds=config.target_length_seconds,
                top_db=config.top_db,
            )

        if recognizer_type == "whisper":
            return WhisperRecognizer()

        available = ", ".join(RecognizerFactory.available_strategies())
        msg = f"Unknown recognizer type: {recognizer_type}. Available: {available}"
        logger.error(msg)
        raise ValueError(msg)
