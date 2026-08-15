import logging
from pathlib import Path

from voicerecognizer.config import DEFAULT_RECOGNITION_CONFIG, RecognitionConfig, RecognizerType
from voicerecognizer.core.interfaces import RecognitionStrategy
from voicerecognizer.recognizers.cnn_recognizer import CNNRecognizer
from voicerecognizer.recognizers.wav2vec2_recognizer import Wav2Vec2Recognizer
from voicerecognizer.recognizers.whisper_recognizer import WhisperRecognizer

logger = logging.getLogger(__name__)


class RecognizerFactory:
    @staticmethod
    def available_strategies() -> tuple[RecognizerType, ...]:
        return ("cnn", "wav2vec2", "whisper")

    @staticmethod
    def create(
        recognizer_type: RecognizerType,
        config: RecognitionConfig = DEFAULT_RECOGNITION_CONFIG,
        use_last: bool = False,
        model_path: str | Path | None = None,
    ) -> RecognitionStrategy:
        logger.info(
            "RecognizerFactory.create: %s (use_last=%s, model_path=%s)",
            recognizer_type,
            use_last,
            model_path,
        )

        if recognizer_type == "cnn":
            target_path = (
                Path(model_path)
                if model_path is not None
                else (config.last_model_path if use_last else config.cnn_weight_path)
            )
            return CNNRecognizer(
                model_path=target_path,
                labels=config.labels,
                sample_rate=config.sample_rate,
                target_length_seconds=config.target_length_seconds,
                top_db=config.top_db,
                n_mels=config.n_mels,
            )

        if recognizer_type == "wav2vec2":
            target_path = (
                Path(model_path)
                if model_path is not None
                else (
                    config.wav2vec2_last_model_dir if use_last else config.wav2vec2_best_model_dir
                )
            )
            return Wav2Vec2Recognizer(
                model_path=target_path,
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
