"""Strategy 06: Wav2Vec2 Consonant/Vowel Multi-Task Learning."""

from voicerecognizer.strategies.phoneme_multi.model import (
    MultiTaskUncertaintyLoss,
    Wav2Vec2ForPhonemeMultiTaskClassification,
)
from voicerecognizer.strategies.phoneme_multi.phoneme_mapping import (
    CONSONANTS,
    VOWELS,
    build_label_phoneme_tables,
    decompose_label,
)

__all__ = [
    "CONSONANTS",
    "VOWELS",
    "MultiTaskUncertaintyLoss",
    "Wav2Vec2ForPhonemeMultiTaskClassification",
    "build_label_phoneme_tables",
    "decompose_label",
]
