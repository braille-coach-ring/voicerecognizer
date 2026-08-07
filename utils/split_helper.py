from collections import Counter
import logging
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)


def safe_stratified_split(
    labels: list[int | str], val_rate: float, seed: int
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
        train_idx, val_idx = next(splitter.split(range(len(labels)), labels))
        return list(train_idx), list(val_idx)

    logger.warning(
        "⚠️ 以下の %d クラスはデータセット内のサンプル数が1件のため、学習セット(train)へ優先割り当てします: %s",
        len(singletons),
        sorted(list(singletons)),
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
    train_sub, val_sub = next(splitter.split(range(len(multi_labels)), multi_labels))

    train_idx = [multi_indices[i] for i in train_sub] + [
        i for i, lbl in enumerate(labels) if lbl in singletons
    ]
    val_idx = [multi_indices[i] for i in val_sub]

    return train_idx, val_idx
