import logging
from collections import Counter
from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

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


def speaker_aware_stratified_split(
    labels: Sequence[int | str],
    speakers: Sequence[str],
    val_rate: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """話者グループをまたがない validation split を作る。

    ラベル分布をできるだけ保つため StratifiedGroupKFold を使い、指定 val_rate に最も近い fold を
    validation として採用する。speaker 情報が足りない場合は従来の安全な stratified split に戻す。
    """
    if len(labels) != len(speakers):
        raise ValueError("labels と speakers の長さが一致していません")
    if len(labels) == 0:
        return [], []

    normalized_speakers = [str(speaker).strip() for speaker in speakers]
    known_speakers = {speaker for speaker in normalized_speakers if speaker}
    if len(known_speakers) < 2:
        logger.warning("speaker-aware split に必要な話者数が足りないため通常 split に戻します")
        return safe_stratified_split(labels, val_rate=val_rate, seed=seed)

    groups = [
        speaker if speaker else f"unknown_sample_{index}"
        for index, speaker in enumerate(normalized_speakers)
    ]
    unique_group_count = len(set(groups))
    n_splits = min(max(2, round(1.0 / max(val_rate, 1e-6))), unique_group_count)
    if n_splits < 2:
        return safe_stratified_split(labels, val_rate=val_rate, seed=seed)

    try:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        candidates = list(splitter.split(np.arange(len(labels)), labels, groups=groups))
    except ValueError as exc:
        logger.warning("speaker-aware split に失敗したため通常 split に戻します: %s", exc)
        return safe_stratified_split(labels, val_rate=val_rate, seed=seed)

    if not candidates:
        return safe_stratified_split(labels, val_rate=val_rate, seed=seed)

    target_val_size = len(labels) * val_rate
    train_idx, val_idx = min(
        candidates,
        key=lambda item: abs(len(item[1]) - target_val_size),
    )
    return list(train_idx), list(val_idx)
