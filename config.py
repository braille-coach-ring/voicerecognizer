import os
from dataclasses import dataclass, field
from typing import Literal
from pathlib import Path
from dotenv import load_dotenv

from config_labels import (
    ALL_HIRAGANA_LABELS,
    SEION_LABELS,
    DAKUON_LABELS,
    HANDAKUON_LABELS,
    YOON_LABELS,
    OTHER_LABELS,
    HIRAGANA_TO_ROMAJI,
    ROMAJI_TO_HIRAGANA,
)

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    chunk_seconds: float = 0.1
    window_seconds: float = 1.0
    channels: int = 1
    callback_blocksize_seconds: float = 0.05
    warmup_sleep_ms: int = 500


@dataclass(frozen=True)
class PreprocessConfig:
    n_mels: int = 64
    n_fft: int = 400
    hop_length: int = 160
    top_db: float = 31.6
    vad_silence_threshold: float = 0.015407
    vad_rms_threshold: float = 0.005181
    vad_min_speech_chunks: int = 2
    vad_min_active_ratio: float = 0.02
    dynamic_threshold_enabled: bool = False
    min_top_db: float = 23.6
    max_top_db: float = 39.6
    noise_update_rate: float = 0.005


RecognizerType = Literal["cnn", "wav2vec2", "whisper"]


@dataclass(frozen=True)
class RecognitionConfig:
    sample_rate: int = 16000
    model_type: RecognizerType = "cnn"
    target_length_seconds: float = 0.6
    top_db: float = 31.6
    n_mels: int = 64
    n_fft: int = 400
    hop_length: int = 160
    labels: tuple[str, ...] = field(default_factory=lambda: ALL_HIRAGANA_LABELS)
    weights_dir: Path = PROJECT_ROOT / "weights"
    cnn_weight_path: Path = PROJECT_ROOT / "weights" / "best_model.pth"
    last_model_path: Path = PROJECT_ROOT / "weights" / "last_model.pth"
    torchscript_model_path: Path = PROJECT_ROOT / "weights" / "hiragana_cnn.pt"
    wav2vec2_pretrained_model_name: str = "facebook/wav2vec2-base"
    wav2vec2_best_model_dir: Path = PROJECT_ROOT / "weights" / "wav2vec2_best"
    wav2vec2_last_model_dir: Path = PROJECT_ROOT / "weights" / "wav2vec2_last"
    output_audio_path: Path = PROJECT_ROOT / "predicted_audio.wav"
    raw_dataset_dir: Path = PROJECT_ROOT / "dataset"
    collected_dataset_dir: Path = PROJECT_ROOT / "dataset" / "collected"
    merged_dataset_dir: Path = PROJECT_ROOT / "merged_dataset"
    processed_dataset_dir: Path = PROJECT_ROOT / "processed_dataset"
    log_path: Path = PROJECT_ROOT / "log"


@dataclass(frozen=True)
class HuggingFaceConfig:
    token: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    repo_id: str = field(
        default_factory=lambda: os.getenv(
            "HF_REPO_ID", "braille-mate/braille-mate-hiragana-recognizer"
        )
    )
    auto_upload: bool = field(
        default_factory=lambda: os.getenv("HF_AUTO_UPLOAD", "false").lower() == "true"
    )


DEFAULT_AUDIO_CONFIG = AudioConfig()
DEFAULT_PREPROCESS_CONFIG = PreprocessConfig()
DEFAULT_RECOGNITION_CONFIG = RecognitionConfig()
DEFAULT_HUGGINGFACE_CONFIG = HuggingFaceConfig()
