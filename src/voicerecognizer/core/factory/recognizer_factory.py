import logging
from pathlib import Path

from voicerecognizer.config import DEFAULT_RECOGNITION_CONFIG, RecognitionConfig
from voicerecognizer.core.interfaces import RecognitionStrategy
from voicerecognizer.recognizers.cnn_recognizer import CNNRecognizer
from voicerecognizer.recognizers.wav2vec2_recognizer import Wav2Vec2Recognizer

logger = logging.getLogger(__name__)


class RecognizerFactory:
    @staticmethod
    def available_strategies() -> tuple[str, ...]:
        from voicerecognizer.strategies.registry import DEFINED_STRATEGIES

        base_types: tuple[str, ...] = ("cnn", "wav2vec2")
        strategy_names = tuple(DEFINED_STRATEGIES.keys())
        return base_types + strategy_names

    @staticmethod
    def create(
        recognizer_type: str,
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
                target_length_seconds=config.target_length_seconds,
            )

        if recognizer_type in ("wav2vec2", "wav2vec2_baseline"):
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
                target_length_seconds=config.target_length_seconds,
            )

        from voicerecognizer.strategies.registry import (
            DEFINED_STRATEGIES,
            create_strategy_recognizer,
        )

        if recognizer_type in DEFINED_STRATEGIES:
            return create_strategy_recognizer(
                strategy_name=recognizer_type,
                config=config,
                use_last=use_last,
                model_path=model_path,
            )

        available = ", ".join(RecognizerFactory.available_strategies())
        msg = f"Unknown recognizer type/strategy: {recognizer_type}. Available: {available}"
        logger.error(msg)
        raise ValueError(msg)
