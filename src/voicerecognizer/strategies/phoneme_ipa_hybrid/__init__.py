"""Strategy: Wav2Vec2 Consonant/Vowel Multi-Task + XLS-R IPA Knowledge Distillation Hybrid."""

from voicerecognizer.strategies.phoneme_ipa_hybrid.model import (
    MultiTaskIPAHybridOutput,
    Wav2Vec2ForPhonemeIPAHybridClassification,
)
from voicerecognizer.strategies.phoneme_ipa_hybrid.train import train_phoneme_ipa_hybrid

__all__ = [
    "MultiTaskIPAHybridOutput",
    "Wav2Vec2ForPhonemeIPAHybridClassification",
    "train_phoneme_ipa_hybrid",
]
