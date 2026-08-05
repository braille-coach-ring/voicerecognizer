import argparse
import logging
from pathlib import Path

from config import DEFAULT_PREPROCESS_CONFIG, DEFAULT_RECOGNITION_CONFIG
from core.factory.recognizer_factory import RecognizerFactory
from core.services.audio_pipeline import AudioPipeline
from core.services.voice_recognizer import VoiceRecognizer
from runtime.vad import VoiceActivityDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(DEFAULT_RECOGNITION_CONFIG.log_path, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run voice recognition.")
    parser.add_argument(
        "audio",
        nargs="?",
        type=Path,
        help="Path to a wav file. If omitted, no recognition is executed.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_RECOGNITION_CONFIG.model_type,
        choices=RecognizerFactory.available_strategies(),
        help="Recognition strategy to use.",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=DEFAULT_PREPROCESS_CONFIG.vad_silence_threshold,
        help="Peak volume threshold for VAD (default: %(default)s).",
    )
    parser.add_argument(
        "--rms-threshold",
        type=float,
        default=DEFAULT_PREPROCESS_CONFIG.vad_rms_threshold,
        help="RMS volume threshold for VAD (default: %(default)s).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    logger.info("Voice Recognizer を起動しました。")
    logger.info("使用モデル: %s", args.model)
    logger.info("VAD設定: Peak閾値=%.4f / RMS閾値=%.4f", args.silence_threshold, args.rms_threshold)

    strategy = RecognizerFactory.create(args.model)
    recognizer = VoiceRecognizer(strategy)
    vad = VoiceActivityDetector(
        silence_threshold=args.silence_threshold,
        rms_threshold=args.rms_threshold,
    )
    pipeline = AudioPipeline(recognizer, vad=vad)
    logger.info("パイプラインの構築が完了しました。")

    if args.audio is None:
        logger.info("音声入力待ちモードに入ります...")
        result = pipeline.run_until_speech()
        logger.info("マイク認識結果: %s", result)
        return

    logger.info("音声ファイルを入力します...")
    result = pipeline.run(args.audio)
    logger.info("ファイル認識結果: %s", result)


if __name__ == "__main__":
    main()
