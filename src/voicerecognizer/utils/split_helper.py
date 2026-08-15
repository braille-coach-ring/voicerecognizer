import logging
from collections import Counter
from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from voicerecognizer.utils.speaker import UNKNOWN_SPEAKER_ID

logger = logging.getLogger(__name__)


def safe_stratified_split(
    labels: Sequence[int | str], val_rate: float, seed: int
) -> tuple[list[int], list[int]]:
    """
    データセットのラベル分布に基づき、サンプル数が少ないクラス（1件のみ等）が存在する場合にも
    エラーなく安全に Train / Validation インデックスに分割する関数。
    ・サンプル数 >= 2 のクラス: StratifiedShuffleSplit で均等分割
    ・サンプル数 == 1 のクラス: 学習セット (train_idx) に配置してモデルの学習機会を確保
    """
    if len(labels) == 0:
        return [], []

    counts = Counter(labels)
    singletons = {label for label, count in counts.items() if count < 2}

    if not singletons:
        # 全クラスのサンプル数が 2 以上の場合、通常の StratifiedShuffleSplit を実行
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_rate, random_state=seed)
        train_idx, val_idx = next(splitter.split(np.arange(len(labels)), labels))
        return list(train_idx), list(val_idx)

    logger.warning(
        "以下の %d クラスはデータセット内のサンプル数が1件のため、学習セット(train)へ優先割り当てします: %s",
        len(singletons),
        sorted(singletons),
    )

    multi_indices = [i for i, lbl in enumerate(labels) if lbl not in singletons]
    multi_labels = [labels[i] for i in multi_indices]

    if len(multi_indices) == 0:
        # 万が一全データが1件ずつの場合は全件学習セットへ割り当て
        return list(range(len(labels))), []

    multi_counts = Counter(multi_labels)
    if any(c < 2 for c in multi_counts.values()):
        # 再集計後も 2 未満が存在する場合は単純シャッフル分割
        rng = np.random.RandomState(seed)
        shuffled = rng.permutation(len(labels)).tolist()
        n_val = max(1, int(len(labels) * val_rate))
        return shuffled[n_val:], shuffled[:n_val]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_rate, random_state=seed)
    train_sub, val_sub = next(splitter.split(np.arange(len(multi_labels)), multi_labels))

    train_idx = [multi_indices[i] for i in train_sub] + [
        i for i, lbl in enumerate(labels) if lbl in singletons
    ]
    val_idx = [multi_indices[i] for i in val_sub]

    return train_idx, val_idx


def safe_group_split(
    labels: Sequence[int | str],
    groups: Sequence[str],
    val_rate: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """
    Split data while keeping every group, usually a speaker, fully in either
    train or validation. Falls back to label-stratified splitting when speaker
    metadata is missing or unusable.
    """
    if len(labels) == 0:
        return [], []

    if len(groups) != len(labels):
        logger.warning(
            "Speaker-aware split requested but labels (%d) and groups (%d) differ. "
            "Falling back to stratified split.",
            len(labels),
            len(groups),
        )
        return safe_stratified_split(labels, val_rate=val_rate, seed=seed)

    normalized_groups = [str(group or UNKNOWN_SPEAKER_ID) for group in groups]
    known_groups = {group for group in normalized_groups if group != UNKNOWN_SPEAKER_ID}
    unique_groups = set(normalized_groups)

    if len(unique_groups) < 2 or len(known_groups) < 2:
        logger.warning(
            "Speaker-aware split requested but fewer than two known speakers were found. "
            "Falling back to stratified split."
        )
        return safe_stratified_split(labels, val_rate=val_rate, seed=seed)

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_rate, random_state=seed)
    train_idx_arr, val_idx_arr = next(
        splitter.split(np.arange(len(labels)), labels, normalized_groups)
    )
    train_idx = list(map(int, train_idx_arr))
    val_idx = list(map(int, val_idx_arr))

    train_labels = {labels[index] for index in train_idx}
    missing_train_labels = sorted({str(label) for label in set(labels) - train_labels})
    if missing_train_labels:
        logger.warning(
            "Speaker-aware split left %d label(s) absent from training: %s",
            len(missing_train_labels),
            missing_train_labels,
        )

    train_groups = {normalized_groups[index] for index in train_idx}
    val_groups = {normalized_groups[index] for index in val_idx}
    overlap = train_groups & val_groups
    if overlap:
        raise RuntimeError(f"Speaker split leaked group(s) across train/validation: {overlap}")

    logger.info(
        "Speaker-aware split: train=%d samples/%d speakers, validation=%d samples/%d speakers",
        len(train_idx),
        len(train_groups),
        len(val_idx),
        len(val_groups),
    )

    return train_idx, val_idx
