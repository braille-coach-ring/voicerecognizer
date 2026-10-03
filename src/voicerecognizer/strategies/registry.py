"""Strategy definitions and registry for voice recognition experiments."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from voicerecognizer.config import (
    DEFAULT_RECOGNITION_CONFIG,
    PROJECT_ROOT,
    RecognitionConfig,
)
from voicerecognizer.core.interfaces import RecognitionStrategy

logger = logging.getLogger(__name__)


class StrategyCategory(StrEnum):
    BASELINE = "Baseline"
    DISTILLATION = "Knowledge Distillation"
    REPRESENTATION = "Representation & Loss"
    ACOUSTIC_AUGMENTATION = "Acoustic Augmentation"
    BACKBONE_INFERENCE = "Backbone & Inference"


class StrategyStatus(StrEnum):
    ACTIVE = "active"  # Ready and benchmarked
    IMPLEMENTED = "implemented"  # Code ready, awaiting benchmark
    PLANNED = "planned"  # In roadmap


@dataclass(frozen=True)
class StrategyMetadata:
    name: str
    display_name: str
    category: StrategyCategory
    description: str
    status: StrategyStatus
    model_dir: Path
    extra_info: dict[str, Any] = field(default_factory=dict)


STRATEGIES_DIR = PROJECT_ROOT / "weights" / "strategies"


# 10通りの計画ストラテジー（およびベースライン）の定義
DEFINED_STRATEGIES: dict[str, StrategyMetadata] = {
    # 00. Baseline
    "wav2vec2_baseline": StrategyMetadata(
        name="wav2vec2_baseline",
        display_name="Wav2Vec2 Baseline",
        category=StrategyCategory.BASELINE,
        description="Standard facebook/wav2vec2-base fine-tuned on single morae without distillation.",
        status=StrategyStatus.ACTIVE,
        model_dir=DEFAULT_RECOGNITION_CONFIG.wav2vec2_best_model_dir,
    ),
    # 01. Distillation: XLS-R IPA Soft-target
    "wav2vec2_ipa_kd": StrategyMetadata(
        name="wav2vec2_ipa_kd",
        display_name="Wav2Vec2 + XLS-R IPA Distillation",
        category=StrategyCategory.DISTILLATION,
        description="Knowledge distillation using facebook/wav2vec2-xlsr-53-espeak-cv-ft frame phoneme posteriors.",
        status=StrategyStatus.ACTIVE,
        model_dir=STRATEGIES_DIR / "wav2vec2_ipa_kd",
    ),
    # 02. Distillation: Whisper Large-v3 Hidden Feature
    "wav2vec2_whisper_kd": StrategyMetadata(
        name="wav2vec2_whisper_kd",
        display_name="Wav2Vec2 + Whisper-large-v3 Feature KD",
        category=StrategyCategory.DISTILLATION,
        description="Intermediate hidden representation alignment from robust Whisper-large-v3 encoder.",
        status=StrategyStatus.ACTIVE,
        model_dir=STRATEGIES_DIR / "wav2vec2_whisper_kd",
    ),
    # 03. Distillation: Clean Teacher vs Noisy Student (Denoising KD)
    "wav2vec2_denoise_kd": StrategyMetadata(
        name="wav2vec2_denoise_kd",
        display_name="Wav2Vec2 Denoising Consistency KD",
        category=StrategyCategory.DISTILLATION,
        description="Distills clean-audio teacher representations into noise/distortion-corrupted student audio.",
        status=StrategyStatus.PLANNED,
        model_dir=STRATEGIES_DIR / "wav2vec2_denoise_kd",
    ),
    # 04. Representation: ArcFace Angular Margin Loss
    "wav2vec2_arcface": StrategyMetadata(
        name="wav2vec2_arcface",
        display_name="Wav2Vec2 + ArcFace Angular Margin",
        category=StrategyCategory.REPRESENTATION,
        description="Additive Angular Margin Loss to strictly separate acoustically close phoneme pairs.",
        status=StrategyStatus.ACTIVE,
        model_dir=STRATEGIES_DIR / "wav2vec2_arcface",
    ),
    # 05. Representation: Onset-Focused Transient Loss
    "wav2vec2_onset_focused": StrategyMetadata(
        name="wav2vec2_onset_focused",
        display_name="Wav2Vec2 Onset Transient Weighted",
        category=StrategyCategory.REPRESENTATION,
        description="Loss weighting focused on the initial 0-50ms consonant burst/fricative onset.",
        status=StrategyStatus.PLANNED,
        model_dir=STRATEGIES_DIR / "wav2vec2_onset_focused",
    ),
    # 06. Representation: Consonant-Vowel Multi-Task
    "wav2vec2_phoneme_multi": StrategyMetadata(
        name="wav2vec2_phoneme_multi",
        display_name="Wav2Vec2 Consonant/Vowel Multi-Task",
        category=StrategyCategory.REPRESENTATION,
        description="Joint multi-task prediction of Character (105) + Consonant (30) + Vowel (7).",
        status=StrategyStatus.ACTIVE,
        model_dir=STRATEGIES_DIR / "wav2vec2_phoneme_multi",
    ),
    # 07. Augmentation: RIR Room Acoustics & Mic Clipping
    "wav2vec2_rir_simulation": StrategyMetadata(
        name="wav2vec2_rir_simulation",
        display_name="Wav2Vec2 + RIR & Mic Simulation",
        category=StrategyCategory.ACOUSTIC_AUGMENTATION,
        description="Room impulse response convolution (RT60 0.1-0.6s) and dynamic microphone distortion simulation.",
        status=StrategyStatus.ACTIVE,
        model_dir=STRATEGIES_DIR / "wav2vec2_rir_simulation",
    ),
    # 08. Augmentation: Teacher Pseudo-Labeling Scaling
    "wav2vec2_pseudo_label": StrategyMetadata(
        name="wav2vec2_pseudo_label",
        display_name="Wav2Vec2 + Semi-supervised Pseudo-Labels",
        category=StrategyCategory.ACOUSTIC_AUGMENTATION,
        description="Scaling training dataset using high-confidence teacher predictions on unlabelled speech.",
        status=StrategyStatus.PLANNED,
        model_dir=STRATEGIES_DIR / "wav2vec2_pseudo_label",
    ),
    # 09. Backbone: Whisper Encoder ONNX
    "whisper_encoder_onnx": StrategyMetadata(
        name="whisper_encoder_onnx",
        display_name="Whisper Encoder ONNX Classifier",
        category=StrategyCategory.BACKBONE_INFERENCE,
        description="Replacing Wav2Vec2 with OpenAI Whisper-base encoder + classification head in ONNX INT8.",
        status=StrategyStatus.PLANNED,
        model_dir=STRATEGIES_DIR / "whisper_encoder_onnx",
    ),
    # 10. Inference: Test-Time Multi-Window Ensemble
    "wav2vec2_tta_ensemble": StrategyMetadata(
        name="wav2vec2_tta_ensemble",
        display_name="Wav2Vec2 + Multi-Window TTA",
        category=StrategyCategory.BACKBONE_INFERENCE,
        description="Test-time multi-window shift voting [-30ms, 0ms, +30ms] to absorb VAD boundary jitter.",
        status=StrategyStatus.PLANNED,
        model_dir=DEFAULT_RECOGNITION_CONFIG.wav2vec2_best_model_dir,
    ),
    # 11. Hybrid: Consonant/Vowel Multi-Task + XLS-R IPA KD
    "wav2vec2_phoneme_ipa_hybrid": StrategyMetadata(
        name="wav2vec2_phoneme_ipa_hybrid",
        display_name="Wav2Vec2 Phoneme Multi-Task + IPA KD Hybrid",
        category=StrategyCategory.REPRESENTATION,
        description="Joint multi-task consonant/vowel decomposition with XLS-R IPA soft-target distillation.",
        status=StrategyStatus.ACTIVE,
        model_dir=STRATEGIES_DIR / "wav2vec2_phoneme_ipa_hybrid",
    ),
}


def list_strategies() -> list[StrategyMetadata]:
    """Returns list of all registered strategies."""
    return list(DEFINED_STRATEGIES.values())


def get_strategy_metadata(strategy_name: str) -> StrategyMetadata:
    """Retrieve metadata for a given strategy name."""
    if strategy_name not in DEFINED_STRATEGIES:
        available = ", ".join(DEFINED_STRATEGIES.keys())
        raise KeyError(f"Unknown strategy: '{strategy_name}'. Available: {available}")
    return DEFINED_STRATEGIES[strategy_name]


def resolve_strategy_model_path(strategy_name: str) -> Path:
    """Resolve the model directory for a specific strategy."""
    meta = get_strategy_metadata(strategy_name)
    return meta.model_dir


def create_strategy_recognizer(
    strategy_name: str,
    config: RecognitionConfig = DEFAULT_RECOGNITION_CONFIG,
    use_last: bool = False,
    model_path: str | Path | None = None,
    auto_download: bool = True,
) -> RecognitionStrategy:
    """Instantiate a recognizer corresponding to the strategy."""
    from voicerecognizer.recognizers.wav2vec2_recognizer import Wav2Vec2Recognizer
    from voicerecognizer.utils.model_uploader import download_strategy_weights_if_needed

    meta = get_strategy_metadata(strategy_name)

    target_dir = Path(model_path) if model_path is not None else meta.model_dir

    if auto_download and strategy_name != "wav2vec2_baseline":
        onnx_candidates = [
            target_dir / config.wav2vec2_mel_int8_onnx_filename,
            target_dir / config.wav2vec2_mel_fp32_onnx_filename,
            target_dir / config.wav2vec2_int8_onnx_filename,
            target_dir / config.wav2vec2_fp32_onnx_filename,
        ]
        has_onnx = any(f.exists() for f in onnx_candidates)
        has_safetensors = (target_dir / "model.safetensors").exists()

        if not target_dir.exists() or (not has_onnx and not has_safetensors):
            logger.info(
                "ローカルに戦略 '%s' のモデルが見つかりません。Hugging Face Hub より自動ダウンロードを試みます...",
                strategy_name,
            )
            download_strategy_weights_if_needed(strategy_name=strategy_name, target_dir=target_dir)

    # Fallback to wav2vec2_best if target_dir doesn't exist yet
    if not target_dir.exists():
        if strategy_name == "wav2vec2_baseline":
            target_dir = config.wav2vec2_best_model_dir
        else:
            raise FileNotFoundError(
                f"Model directory for strategy '{strategy_name}' does not exist: {target_dir}. "
                f"Please train the strategy model first or benchmark an active strategy."
            )

    return Wav2Vec2Recognizer(
        model_path=target_dir,
        labels=config.labels,
        target_length_seconds=config.target_length_seconds,
        auto_download=auto_download,
    )
