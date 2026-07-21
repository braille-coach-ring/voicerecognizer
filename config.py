from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RecognitionConfig:
    model_type: str = "cnn"
    sample_rate: int = 16000
    target_length_seconds: float = 1.0
    top_db: int = 30
    n_mels: int = 64
    labels: tuple[str, ...] = field(default_factory=lambda: ("a", "e", "i", "o", "u"))
    cnn_weight_path: Path = PROJECT_ROOT / "weights" / "best_model.pth"
    output_audio_path: Path = PROJECT_ROOT / "predicted_audio.wav"


DEFAULT_CONFIG = RecognitionConfig()
