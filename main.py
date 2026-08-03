import argparse
from pathlib import Path

from config import DEFAULT_CONFIG
from core.factory.recognizer_factory import RecognizerFactory
from core.services.audio_pipeline import AudioPipeline
from core.services.voice_recognizer import VoiceRecognizer


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
        default=DEFAULT_CONFIG.model_type,
        choices=RecognizerFactory.available_strategies(),
        help="Recognition strategy to use.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    strategy = RecognizerFactory.create(args.model)
    recognizer = VoiceRecognizer(strategy)
    pipeline = AudioPipeline(recognizer)

    if args.audio is None:
        pipeline.run()
        return

    pipeline.run(args.audio)


if __name__ == "__main__":
    main()
