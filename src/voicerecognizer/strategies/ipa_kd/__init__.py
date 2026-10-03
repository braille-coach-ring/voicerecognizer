"""IPA Phoneme Knowledge Distillation strategy package."""

from voicerecognizer.strategies.ipa_kd.model import (
    IPAKDOutput,
    Wav2Vec2ForIPAKDClassification,
)
from voicerecognizer.strategies.ipa_kd.teacher import (
    DEFAULT_IPA_TEACHER_MODEL_ID,
    IPA_VOCAB_SIZE,
    IPATeacher,
    precompute_ipa_posteriors_cache,
)

__all__ = [
    "DEFAULT_IPA_TEACHER_MODEL_ID",
    "IPA_VOCAB_SIZE",
    "IPAKDOutput",
    "IPATeacher",
    "Wav2Vec2ForIPAKDClassification",
    "precompute_ipa_posteriors_cache",
]
