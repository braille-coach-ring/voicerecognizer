"""Phonological decomposition module for Japanese syllable labels.

Provides mappings from Japanese hiragana / romaji syllable labels (e.g. 'ka', 'か')
into constituent acoustic consonant and vowel representations.
"""

from __future__ import annotations

from typing import NamedTuple

from voicerecognizer.config_labels import HIRAGANA_TO_ROMAJI

# Standard Vowel Classifications (7 classes)
VOWELS: tuple[str, ...] = (
    "a",
    "i",
    "u",
    "e",
    "o",
    "none",  # Syllabic nasal 'n' / consonants without clear vowels
    "other",  # Out-of-vocabulary / noise label
)

# Acoustic Consonant Classifications (30 classes)
CONSONANTS: tuple[str, ...] = (
    "none",  # Plain vowels ('a', 'i', 'u', 'e', 'o')
    "k",
    "s",
    "sh",
    "t",
    "ch",
    "ts",
    "n",
    "h",
    "f",
    "m",
    "y",
    "r",
    "w",
    "g",
    "z",
    "j",
    "d",
    "b",
    "p",
    "ky",
    "ny",
    "hy",
    "my",
    "ry",
    "gy",
    "by",
    "py",
    "N",  # Syllabic nasal 'n'
    "other",
)

VOWEL_TO_IDX: dict[str, int] = {v: idx for idx, v in enumerate(VOWELS)}
CONSONANT_TO_IDX: dict[str, int] = {c: idx for idx, c in enumerate(CONSONANTS)}


class PhonemeDecomposition(NamedTuple):
    """Decomposed phoneme representation of a Japanese syllable."""

    consonant: str
    vowel: str
    consonant_idx: int
    vowel_idx: int


def _build_explicit_mapping() -> dict[str, tuple[str, str]]:
    """Builds explicit consonant and vowel mapping for all romaji labels."""
    mapping: dict[str, tuple[str, str]] = {
        # Seion (Vowels only)
        "a": ("none", "a"),
        "i": ("none", "i"),
        "u": ("none", "u"),
        "e": ("none", "e"),
        "o": ("none", "o"),
        # k-row
        "ka": ("k", "a"),
        "ki": ("k", "i"),
        "ku": ("k", "u"),
        "ke": ("k", "e"),
        "ko": ("k", "o"),
        # s-row
        "sa": ("s", "a"),
        "shi": ("sh", "i"),
        "su": ("s", "u"),
        "se": ("s", "e"),
        "so": ("s", "o"),
        # t-row
        "ta": ("t", "a"),
        "chi": ("ch", "i"),
        "tsu": ("ts", "u"),
        "te": ("t", "e"),
        "to": ("t", "o"),
        # n-row
        "na": ("n", "a"),
        "ni": ("n", "i"),
        "nu": ("n", "u"),
        "ne": ("n", "e"),
        "no": ("n", "o"),
        # h-row
        "ha": ("h", "a"),
        "hi": ("h", "i"),
        "fu": ("f", "u"),
        "he": ("h", "e"),
        "ho": ("h", "o"),
        # m-row
        "ma": ("m", "a"),
        "mi": ("m", "i"),
        "mu": ("m", "u"),
        "me": ("m", "e"),
        "mo": ("m", "o"),
        # y-row
        "ya": ("y", "a"),
        "yu": ("y", "u"),
        "yo": ("y", "o"),
        # r-row
        "ra": ("r", "a"),
        "ri": ("r", "i"),
        "ru": ("r", "u"),
        "re": ("r", "e"),
        "ro": ("r", "o"),
        # w-row
        "wa": ("w", "a"),
        "wo": ("w", "o"),
        # Syllabic nasal
        "n": ("N", "none"),
        # Dakuon (g-row)
        "ga": ("g", "a"),
        "gi": ("g", "i"),
        "gu": ("g", "u"),
        "ge": ("g", "e"),
        "go": ("g", "o"),
        # Dakuon (z-row)
        "za": ("z", "a"),
        "ji": ("j", "i"),
        "zu": ("z", "u"),
        "ze": ("z", "e"),
        "zo": ("z", "o"),
        # Dakuon (d-row)
        "da": ("d", "a"),
        "di": ("d", "i"),
        "du": ("d", "u"),
        "de": ("d", "e"),
        "do": ("d", "o"),
        # Dakuon (b-row)
        "ba": ("b", "a"),
        "bi": ("b", "i"),
        "bu": ("b", "u"),
        "be": ("b", "e"),
        "bo": ("b", "o"),
        # Handakuon (p-row)
        "pa": ("p", "a"),
        "pi": ("p", "i"),
        "pu": ("p", "u"),
        "pe": ("p", "e"),
        "po": ("p", "o"),
        # Yoon (Contracted sounds)
        "kya": ("ky", "a"),
        "kyu": ("ky", "u"),
        "kyo": ("ky", "o"),
        "sha": ("sh", "a"),
        "shu": ("sh", "u"),
        "sho": ("sh", "o"),
        "cha": ("ch", "a"),
        "chu": ("ch", "u"),
        "cho": ("ch", "o"),
        "nya": ("ny", "a"),
        "nyu": ("ny", "u"),
        "nyo": ("ny", "o"),
        "hya": ("hy", "a"),
        "hyu": ("hy", "u"),
        "hyo": ("hy", "o"),
        "mya": ("my", "a"),
        "myu": ("my", "u"),
        "myo": ("my", "o"),
        "rya": ("ry", "a"),
        "ryu": ("ry", "u"),
        "ryo": ("ry", "o"),
        "gya": ("gy", "a"),
        "gyu": ("gy", "u"),
        "gyo": ("gy", "o"),
        "ja": ("j", "a"),
        "ju": ("j", "u"),
        "jo": ("j", "o"),
        "bya": ("by", "a"),
        "byu": ("by", "u"),
        "byo": ("by", "o"),
        "pya": ("py", "a"),
        "pyu": ("py", "u"),
        "pyo": ("py", "o"),
        # Other
        "other": ("other", "other"),
    }
    return mapping


_ROMAJI_MAP = _build_explicit_mapping()


def decompose_label(label: str) -> PhonemeDecomposition:
    """Decomposes a given hiragana or romaji syllable label into consonant and vowel components."""
    clean_label = label.strip().lower()
    romaji = HIRAGANA_TO_ROMAJI.get(clean_label, clean_label)

    if romaji in _ROMAJI_MAP:
        cons, vow = _ROMAJI_MAP[romaji]
    else:
        cons, vow = "other", "other"

    c_idx = CONSONANT_TO_IDX.get(cons, CONSONANT_TO_IDX["other"])
    v_idx = VOWEL_TO_IDX.get(vow, VOWEL_TO_IDX["other"])

    return PhonemeDecomposition(
        consonant=cons,
        vowel=vow,
        consonant_idx=c_idx,
        vowel_idx=v_idx,
    )


def build_label_phoneme_tables(
    labels: tuple[str, ...] | list[str],
) -> tuple[list[int], list[int]]:
    """Builds integer lookup tables mapping each class index to its consonant and vowel indices."""
    c_indices: list[int] = []
    v_indices: list[int] = []

    for lbl in labels:
        decomp = decompose_label(lbl)
        c_indices.append(decomp.consonant_idx)
        v_indices.append(decomp.vowel_idx)

    return c_indices, v_indices
