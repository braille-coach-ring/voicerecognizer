"""
Split dataset into Train, Val, Test including the new speakerphone dataset.
"""

import csv
from collections import Counter, defaultdict
from operator import itemgetter
from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    splits_dir = PROJECT_ROOT / "data_splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # 1. 新データ（rinry の 021〜030）を抽出
    rinry_files = list((PROJECT_ROOT / "dataset" / "rinry").glob("*/*.wav"))
    new_speakerphone_samples: list[tuple[str, str]] = []
    for f in rinry_files:
        stem = f.stem
        if stem.isdigit() and int(stem) >= 21:
            rel_path = f.relative_to(PROJECT_ROOT).as_posix()
            label = f.parent.name
            new_speakerphone_samples.append((rel_path, label))

    def _sort_sample(item: tuple[str, str]) -> tuple[str, str]:
        return (item[1], item[0])

    new_speakerphone_samples.sort(key=_sort_sample)
    print(f"Total new speakerphone samples: {len(new_speakerphone_samples)}")

    # 2. 既存データ（merged_dataset/index.csv に記載のデータ）を読み込み
    merged_index = PROJECT_ROOT / "merged_dataset" / "index.csv"
    existing_samples = []
    if merged_index.exists():
        with open(merged_index, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filepath = row.get("filepath", "").strip()
                label = row.get("label", "").strip()
                if (
                    filepath
                    and label
                    and not any(filepath == sp[0] for sp in new_speakerphone_samples)
                ):
                    existing_samples.append((filepath, label))
    print(f"Total existing samples: {len(existing_samples)}")

    # 3. 新環境データを Train (70%), Val (15%), Test (15%) に Stratified 分割
    new_labels = [s[1] for s in new_speakerphone_samples]

    # First split: Train (70%) vs Temp (30%)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(sss1.split(new_speakerphone_samples, new_labels))

    temp_samples = [new_speakerphone_samples[i] for i in temp_idx]
    temp_labels = [new_labels[i] for i in temp_idx]

    # Second split: Val (50% of 30% = 15%) vs Test (50% of 30% = 15%)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_sub_idx, test_sub_idx = next(sss2.split(temp_samples, temp_labels))

    sp_train = [new_speakerphone_samples[i] for i in train_idx]
    sp_val = [temp_samples[i] for i in val_sub_idx]
    sp_test = [temp_samples[i] for i in test_sub_idx]

    print(f"Speakerphone split: Train={len(sp_train)}, Val={len(sp_val)}, Test={len(sp_test)}")

    # 4. 女性話者データ (dataset/ai_female_nanami) の抽出と分割
    female_files = list((PROJECT_ROOT / "dataset" / "ai_female_nanami").glob("*/*.wav"))
    female_samples: list[tuple[str, str]] = []
    for f in female_files:
        rel_path = f.relative_to(PROJECT_ROOT).as_posix()
        label = f.parent.name
        female_samples.append((rel_path, label))
    female_samples.sort(key=_sort_sample)
    print(f"Total AI female samples: {len(female_samples)}")

    female_train: list[tuple[str, str]] = []
    female_val: list[tuple[str, str]] = []
    female_test: list[tuple[str, str]] = []

    if female_samples:
        # クラスごとに 001.wav, 002.wav を Train (208件)、003.wav を Val (52件) と Test (52件) に分割
        fem_by_class: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for s in female_samples:
            fem_by_class[s[1]].append(s)

        for c_idx, (_lbl, c_files) in enumerate(sorted(fem_by_class.items())):
            c_files_sorted = sorted(c_files, key=itemgetter(0))
            if len(c_files_sorted) >= 3:
                female_train.extend(c_files_sorted[:2])
                if c_idx % 2 == 0:
                    female_val.append(c_files_sorted[2])
                else:
                    female_test.append(c_files_sorted[2])
            else:
                female_train.extend(c_files_sorted)

        print(
            f"AI female split: Train={len(female_train)}, Val={len(female_val)}, Test={len(female_test)}"
        )

    # 5. 既存データを Train (80%) vs Val (20%) に Stratified 分割
    ex_labels = [s[1] for s in existing_samples]
    # 出現頻度が2未満のクラスを保護
    ex_counts = Counter(ex_labels)
    singletons = {lbl for lbl, c in ex_counts.items() if c < 2}

    valid_ex_indices = [i for i, s in enumerate(existing_samples) if s[1] not in singletons]
    valid_ex_labels = [existing_samples[i][1] for i in valid_ex_indices]

    sss_ex = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    ex_train_sub, ex_val_sub = next(sss_ex.split(valid_ex_indices, valid_ex_labels))

    ex_train = [existing_samples[valid_ex_indices[i]] for i in ex_train_sub] + [
        existing_samples[i]
        for i in range(len(existing_samples))
        if existing_samples[i][1] in singletons
    ]
    ex_val = [existing_samples[valid_ex_indices[i]] for i in ex_val_sub]
    print(f"Existing split: Train={len(ex_train)}, Val={len(ex_val)}")

    # 6. 合計 Train / Val を作成
    combined_train = ex_train + sp_train + female_train
    combined_val = ex_val + sp_val + female_val
    print(
        f"Combined: Train={len(combined_train)}, Val={len(combined_val)}, "
        f"Test (Speakerphone)={len(sp_test)}, Test (Female)={len(female_test)}"
    )

    # CSV 出力ヘルパー
    def save_csv(path: Path, samples: list[tuple[str, str]]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "label", "predicted_text"])
            for fp, lbl in samples:
                writer.writerow([fp, lbl, ""])

    save_csv(splits_dir / "speakerphone_train.csv", sp_train)
    save_csv(splits_dir / "speakerphone_val.csv", sp_val)
    save_csv(splits_dir / "speakerphone_test.csv", sp_test)
    if female_samples:
        save_csv(splits_dir / "female_train.csv", female_train)
        save_csv(splits_dir / "female_val.csv", female_val)
        save_csv(splits_dir / "female_test.csv", female_test)
        fem_test_dir = splits_dir / "female_test_eval"
        save_csv(fem_test_dir / "index.csv", female_test)

    save_csv(splits_dir / "combined_train.csv", combined_train)
    save_csv(splits_dir / "combined_val.csv", combined_val)

    # Evaluator 用にディレクトリ構成で test の index.csv を配置
    sp_test_dir = splits_dir / "speakerphone_test_eval"
    save_csv(sp_test_dir / "index.csv", sp_test)

    print("Splits successfully saved to:", splits_dir)


if __name__ == "__main__":
    main()
