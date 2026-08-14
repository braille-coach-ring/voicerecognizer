import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from voicerecognizer.config_labels import (
    ALL_HIRAGANA_LABELS,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_env(dotenv_path: str | Path | None = None) -> bool:
    """明示的に .env ファイルをロードするためのヘルパー関数。

    ライブラリインポート時の自動実行は行わず、CLI や呼び出し元から任意で実行可能にします。
    """
    try:
        from dotenv import load_dotenv

        target = Path(dotenv_path) if dotenv_path is not None else (PROJECT_ROOT / ".env")
        if target.exists():
            return load_dotenv(target)
        return False
    except Exception:
        return False


CACHE_DIR = Path(os.getenv("VOICERECOGNIZER_CACHE_DIR", str(Path.home() / ".cache" / "voicerecognizer")))
DEFAULT_WEIGHTS_DIR = CACHE_DIR / "weights"
PUBLIC_DEFAULT_HF_REPO_ID = "braille-mate/braille-mate-hiragana-recognizer"



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
    top_db: float = 19.7
    vad_silence_threshold: float = 0.021067
    vad_rms_threshold: float = 0.007772
    vad_min_speech_chunks: int = 2
    vad_min_active_ratio: float = 0.02
    dynamic_threshold_enabled: bool = False
    min_top_db: float = 11.7
    max_top_db: float = 27.7
    noise_update_rate: float = 0.005


RecognizerType = Literal["cnn", "wav2vec2", "whisper"]


@dataclass(frozen=True)
class RecognitionConfig:
    sample_rate: int = 16000
    model_type: RecognizerType = "cnn"
    target_length_seconds: float = 0.6
    top_db: float = 19.7
    n_mels: int = 64
    n_fft: int = 400
    hop_length: int = 160
    labels: tuple[str, ...] = field(default_factory=lambda: ALL_HIRAGANA_LABELS)
    weights_dir: Path = DEFAULT_WEIGHTS_DIR

    # 共通ファイル名
    labels_filename: str = "labels.json"
    config_filename: str = "config.json"
    preprocessor_config_filename: str = "preprocessor_config.json"

    # CNN モデル関連
    cnn_model_filename: str = "best_model.pth"
    cnn_weight_path: Path = DEFAULT_WEIGHTS_DIR / "best_model.pth"
    last_model_path: Path = DEFAULT_WEIGHTS_DIR / "last_model.pth"
    torchscript_model_path: Path = DEFAULT_WEIGHTS_DIR / "hiragana_cnn.pt"
    cnn_essential_filenames: tuple[str, ...] = ("best_model.pth", "labels.json")

    # Wav2Vec2 モデル関連
    wav2vec2_pretrained_model_name: str = "facebook/wav2vec2-base"
    wav2vec2_best_model_dir: Path = DEFAULT_WEIGHTS_DIR / "wav2vec2_best"
    wav2vec2_last_model_dir: Path = DEFAULT_WEIGHTS_DIR / "wav2vec2_last"

    # 前処理内包型 (Mel/Waveform-prepended) ONNX をデフォルト・最優先とする
    wav2vec2_default_onnx_filename: str = "model_mel_int8.onnx"
    wav2vec2_mel_fp32_onnx_filename: str = "model_mel_fp32.onnx"
    wav2vec2_mel_int8_onnx_filename: str = "model_mel_int8.onnx"
    wav2vec2_int8_onnx_filename: str = "model_int8.onnx"
    wav2vec2_fp32_onnx_filename: str = "model_fp32.onnx"

    # 推論時 ONNX 候補探索順序（前処理内包版 INT8 -> 前処理内包版 FP32 -> 通常 INT8 -> 通常 FP32 -> model.onnx）
    wav2vec2_onnx_candidate_filenames: tuple[str, ...] = (
        "model_mel_int8.onnx",
        "model_mel_fp32.onnx",
        "model_int8.onnx",
        "model_fp32.onnx",
        "model.onnx",
    )
    # Hugging Face 同期対象のファイル一覧
    wav2vec2_essential_filenames: tuple[str, ...] = (
        "model_mel_int8.onnx",
        "model_int8.onnx",
        "labels.json",
        "config.json",
        "preprocessor_config.json",
    )

    output_audio_path: Path = PROJECT_ROOT / "predicted_audio.wav"
    raw_dataset_dir: Path = PROJECT_ROOT / "dataset"
    collected_dataset_dir: Path = PROJECT_ROOT / "dataset" / "collected"
    merged_dataset_dir: Path = PROJECT_ROOT / "merged_dataset"
    processed_dataset_dir: Path = PROJECT_ROOT / "processed_dataset"
    log_path: Path = PROJECT_ROOT / "log"


@dataclass(frozen=True)
class HuggingFaceConfig:
    token: str = field(
        default_factory=lambda: os.getenv("VOICERECOGNIZER_HF_TOKEN", os.getenv("HF_TOKEN", ""))
    )
    repo_id: str = field(
        default_factory=lambda: os.getenv(
            "VOICERECOGNIZER_HF_REPO_ID",
            os.getenv("HF_REPO_ID", PUBLIC_DEFAULT_HF_REPO_ID),
        )
    )
    auto_upload: bool = field(
        default_factory=lambda: os.getenv(
            "VOICERECOGNIZER_HF_AUTO_UPLOAD",
            os.getenv("HF_AUTO_UPLOAD", "false"),
        ).lower()
        == "true"
    )


DEFAULT_AUDIO_CONFIG = AudioConfig()
DEFAULT_PREPROCESS_CONFIG = PreprocessConfig()
DEFAULT_RECOGNITION_CONFIG = RecognitionConfig()
DEFAULT_HUGGINGFACE_CONFIG = HuggingFaceConfig()
