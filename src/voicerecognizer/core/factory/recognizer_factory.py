import logging

from voicerecognizer.config import DEFAULT_RECOGNITION_CONFIG, RecognitionConfig, RecognizerType
from voicerecognizer.core.interfaces import RecognitionStrategy
from voicerecognizer.recognizers.cnn_recognizer import CNNRecognizer
from voicerecognizer.recognizers.wav2vec2_recognizer import Wav2Vec2Recognizer

logger = logging.getLogger(__name__)


class RecognizerFactory:
    @staticmethod
    def available_strategies() -> tuple[RecognizerType, ...]:
        return ("cnn", "wav2vec2")

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
                target_length_seconds=config.target_length_seconds,
            )

        if recognizer_type == "wav2vec2":
            return Wav2Vec2Recognizer(
                model_path=config.wav2vec2_best_model_dir,
                labels=config.labels,
                target_length_seconds=config.target_length_seconds,
            )

        available = ", ".join(RecognizerFactory.available_strategies())
        msg = f"Unknown recognizer type: {recognizer_type}. Available: {available}"
        logger.error(msg)
        raise ValueError(msg)
