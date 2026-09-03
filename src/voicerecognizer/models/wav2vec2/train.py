"""
Wav2Vec2 Fine-Tuning Script with Layer Freezing and Lazy Disk Loading

役割:
  Wav2Vec2 プリトレイニードモデルの下位層フリーズ + processed_dataset からのlazy loadingにより、
  RAM使用量を抑えてファインチューニングを実行し、best_model ディレクトリおよび labels.json を保存します。

使い方:
  uv run python models/wav2vec2/train.py           # デフォルト設定で Wav2Vec2 を継続ファインチューニング
  uv run python models/wav2vec2/train.py --no-resume # ベースモデル (facebook/wav2vec2-base) から新規学習
"""

import argparse
import csv
import filecmp
import gc
import importlib.util
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import soundfile as sf
import torch
import torch.amp
import torch.cuda
import torch.nn
import torch.optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from tqdm import tqdm

from voicerecognizer.config import (
    DEFAULT_AUDIO_CONFIG,
    DEFAULT_PREPROCESS_CONFIG,
    DEFAULT_RECOGNITION_CONFIG,
    PROJECT_ROOT,
)
from voicerecognizer.dataset.hiragana_dataset import HiraganaDataset
from voicerecognizer.evaluation.evaluator import compute_evaluation_result
from voicerecognizer.models.wav2vec2.export_onnx import export_and_benchmark
from voicerecognizer.preprocessing.audio_augmentor import AudioAugmentor
from voicerecognizer.preprocessing.dataset_builder import ensure_merged_and_preprocessed
from voicerecognizer.utils.plot_saver import save_history_plots
from voicerecognizer.utils.split_helper import (
    safe_stratified_split,
    speaker_aware_stratified_split,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class Wav2Vec2ClassificationDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        sample_rate: int,
        target_length_seconds: float,
        top_db: float,
    ):
        source_dataset = HiraganaDataset(
            root_dir=root_dir,
            sample_rate=sample_rate,
            cache_in_memory=False,
        )
        self.labels = source_dataset.labels
        self.data = source_dataset.data
        self.speakers = collect_speakers_for_data(root_dir, self.data)
        self.sample_rate = sample_rate
        self.target_samples = int(target_length_seconds * sample_rate)

        if not self.data:
            raise ValueError(f"No training samples were found in {root_dir}")

        logger.info(
            "Wav2Vec2 dataset lazy loading enabled: %d samples, %d classes. "
            "Only file paths and labels are kept in RAM.",
            len(self.data),
            len(self.labels),
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        wav_path, label = self.data[index]
        waveform, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1, dtype=np.float32)
        waveform = waveform.reshape(-1)

        if sr != self.sample_rate:
            import librosa

            waveform = librosa.resample(
                waveform,
                orig_sr=sr,
                target_sr=self.sample_rate,
            ).astype(np.float32)

        if self.target_samples > 0:
            if len(waveform) > self.target_samples:
                waveform = waveform[: self.target_samples]
            elif len(waveform) < self.target_samples:
                waveform = np.pad(waveform, (0, self.target_samples - len(waveform)))

        return np.ascontiguousarray(waveform, dtype=np.float32), label


def _resolve_index_audio_path(path_value: str, *, root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    root_relative = root / path
    if root_relative.exists():
        return root_relative
    return PROJECT_ROOT / path


def collect_speakers_for_data(
    root_dir: str | Path,
    data: list[tuple[Path, int]],
) -> list[str]:
    root = Path(root_dir)
    index_file = root / "index.csv" if root.is_dir() else root
    if not index_file.exists():
        return [""] * len(data)

    speaker_by_path: dict[str, str] = {}
    with open(index_file, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = str(row.get("filepath") or row.get("\ufefffilepath") or "").strip()
            speaker = str(row.get("speaker") or "").strip()
            if not filepath:
                continue
            wav_path = _resolve_index_audio_path(filepath, root=index_file.parent)
            speaker_by_path[str(wav_path.resolve())] = speaker

    return [speaker_by_path.get(str(wav_path.resolve()), "") for wav_path, _ in data]


def determine_optimal_num_workers(requested_num_workers: int | None = None) -> int:
    """
    CPUコア数とOS特性に応じて DataLoader の num_workers を自動・動的に計算する。

    ・引数 `requested_num_workers` が 0 以上の数値で指定されていれば手動指定を優先。
    ・未指定 (None または < 0) の場合:
      - os.cpu_count() の半数を基本値とする。
      - Windowsの場合は spawn オーバーヘッドや仮想メモリ消費を抑えるため 1 〜 4 の範囲で動的設定。
      - Linux/macOSの場合は 1 〜 8 の範囲で動的設定。
    """
    if requested_num_workers is not None and requested_num_workers >= 0:
        return requested_num_workers

    cpu_count = os.cpu_count() or 1
    if sys.platform == "win32":
        optimal = min(max(1, cpu_count // 2), 4)
    else:
        optimal = min(max(1, cpu_count // 2), 8)

    return optimal


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(
    dataset: Wav2Vec2ClassificationDataset,
    val_rate: float,
    seed: int,
    speaker_aware: bool = False,
) -> tuple[Subset, Subset]:
    labels = [label for _, label in dataset.data]
    if speaker_aware:
        train_idx, val_idx = speaker_aware_stratified_split(
            labels,
            dataset.speakers,
            val_rate=val_rate,
            seed=seed,
        )
        logger.info(
            "speaker-aware split を有効化しました: train=%d, validation=%d",
            len(train_idx),
            len(val_idx),
        )
    else:
        train_idx, val_idx = safe_stratified_split(labels, val_rate=val_rate, seed=seed)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def collect_augmentation_noise_files(noise_dir: str | Path | None) -> list[Path]:
    if noise_dir is None:
        return []

    path = Path(noise_dir)
    if not path.exists():
        logger.warning("augmentation noise dir が存在しません: %s", path)
        return []
    if path.is_file() and path.suffix.lower() == ".wav":
        return [path]
    if not path.is_dir():
        return []
    return sorted(path.rglob("*.wav"))


class AugmentedSubset(Dataset):
    """
    データ拡張（AudioAugmentor）を訓練データセットにのみ動的適用するための Subset ラッパー。
    """

    def __init__(self, subset: Subset, augmentor: AudioAugmentor | None = None):
        self.subset = subset
        self.augmentor = augmentor

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        item = cast(tuple[np.ndarray, int], self.subset[index])
        waveform, label = item
        if self.augmentor is not None:
            waveform = self.augmentor.augment(waveform)
        return waveform, label


def compute_class_weights(
    labels: list[int] | torch.Tensor | np.ndarray,
    num_classes: int,
    power: float = 0.5,
) -> torch.Tensor:
    """
    クラス出現頻度の逆数根（Smooth Inverse Frequency）に応じたクラス重みを計算する。
    W_c = ( max(N_k) / (N_c + eps) ) ** power
    """
    labels_arr = np.asarray(labels)
    counts = np.bincount(labels_arr, minlength=num_classes)
    counts = np.maximum(counts, 1)
    max_count = np.max(counts)
    weights = (max_count / counts) ** power
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def compute_balanced_sampler_weights(
    labels: list[int] | torch.Tensor | np.ndarray,
    num_classes: int,
    power: float = 0.5,
    confusion_label_multipliers: dict[int, float] | None = None,
) -> list[float]:
    labels_arr = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels_arr, minlength=num_classes)
    counts = np.maximum(counts, 1)
    max_count = np.max(counts)
    class_weights = (max_count / counts) ** power
    multipliers = confusion_label_multipliers or {}
    return [
        float(class_weights[int(label)] * multipliers.get(int(label), 1.0)) for label in labels_arr
    ]


def load_confusion_label_multipliers(
    evaluation_result_path: Path | str,
    labels: tuple[str, ...] | list[str],
    *,
    min_count: int = 3,
    max_pairs: int = 20,
    boost: float = 0.5,
) -> dict[int, float]:
    if boost <= 0 or max_pairs <= 0:
        return {}

    path = Path(evaluation_result_path)
    if not path.exists():
        logger.info("混同ペア重点サンプラー: 評価結果が見つからないためスキップします: %s", path)
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("混同ペア重点サンプラー: 評価結果JSONの読み込みに失敗しました: %s", exc)
        return {}

    matrix = payload.get("confusion_matrix", {}) if isinstance(payload, dict) else {}
    if not isinstance(matrix, dict):
        return {}

    pairs: list[tuple[str, str, int, float]] = []
    for true_label, row in matrix.items():
        if not isinstance(row, dict):
            continue
        row_total = sum(int(value) for value in row.values())
        if row_total <= 0:
            continue
        for predicted_label, raw_count in row.items():
            count = int(raw_count)
            if true_label == predicted_label or count < min_count:
                continue
            pairs.append((str(true_label), str(predicted_label), count, count / row_total))

    if not pairs:
        return {}

    pairs.sort(key=lambda item: item[2], reverse=True)
    label_scores: dict[str, float] = {}
    for true_label, predicted_label, count, rate in pairs[:max_pairs]:
        score = count * (1.0 + rate)
        label_scores[true_label] = label_scores.get(true_label, 0.0) + score
        label_scores[predicted_label] = label_scores.get(predicted_label, 0.0) + score

    if not label_scores:
        return {}

    max_score = max(label_scores.values())
    label_to_idx = {label: index for index, label in enumerate(labels)}
    multipliers: dict[int, float] = {}
    for label, score in label_scores.items():
        if label not in label_to_idx:
            continue
        multipliers[label_to_idx[label]] = 1.0 + boost * (float(score) / float(max_score))

    if multipliers:
        top_labels = ", ".join(
            f"{labels[index]}={multiplier:.2f}"
            for index, multiplier in sorted(
                multipliers.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:10]
        )
        logger.info("混同ペア重点サンプラーを有効化します: %s", top_labels)

    return multipliers


def is_labels_compatible(model_path: Path, current_labels: list[str]) -> bool:
    """
    保存済みチェックポイントの labels.json と現在のデータセットの labels が完全一致するか確認する。
    濁音・半濁音の追加などでラベル構成が変わった場合、MISMATCH 崩壊を防ぐため False を返す。
    """
    labels_file = model_path / "labels.json"
    if not labels_file.exists():
        return False
    try:
        saved_labels = json.loads(labels_file.read_text(encoding="utf-8"))
        return saved_labels == list(current_labels)
    except Exception:
        return False


class Wav2Vec2CollateFn:
    def __init__(self, feature_extractor: Any, sample_rate: int):
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate

    def __call__(self, batch: list[tuple[np.ndarray, int]]) -> dict[str, torch.Tensor]:
        waveforms = [waveform for waveform, _ in batch]
        labels = torch.tensor([label for _, label in batch], dtype=torch.long)
        inputs = self.feature_extractor(
            waveforms,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs["labels"] = labels
        return inputs


def build_collate_fn(feature_extractor: Any, sample_rate: int) -> Wav2Vec2CollateFn:
    return Wav2Vec2CollateFn(feature_extractor, sample_rate)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    epoch: int,
    epochs: int,
    max_grad_norm: float,
    loss_fct: torch.nn.Module | None = None,
    scaler: GradScaler | None = None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    progress = tqdm(loader)

    use_amp = scaler is not None and device.type == "cuda"

    for batch in progress:
        batch = move_batch(batch, device)

        optimizer.zero_grad()
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(**batch)
                if loss_fct is not None:
                    loss = loss_fct(outputs.logits, batch["labels"])
                else:
                    loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(**batch)
            if loss_fct is not None:
                loss = loss_fct(outputs.logits, batch["labels"])
            else:
                loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        scheduler.step()

        logits = outputs.logits.detach()
        pred = logits.argmax(dim=-1)
        labels = batch["labels"]
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()

        progress.set_description(f"Wav2Vec2 Epoch {epoch + 1}/{epochs}")
        progress.set_postfix(loss=f"{loss.item():.3f}")

    return total_loss / max(len(loader), 1), correct / max(total, 1)


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    labels: tuple[str, ...] | list[str],
) -> tuple[float, float, list[str], list[str]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_true: list[str] = []
    all_pred: list[str] = []

    use_amp = device.type == "cuda"

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            if use_amp:
                with autocast():
                    outputs = model(**batch)
            else:
                outputs = model(**batch)

            loss = outputs.loss
            pred = outputs.logits.argmax(dim=-1)
            batch_labels = batch["labels"]

            total_loss += loss.item()
            correct += (pred == batch_labels).sum().item()
            total += batch_labels.size(0)

            for true_index, pred_index in zip(
                batch_labels.cpu().numpy(), pred.cpu().numpy(), strict=False
            ):
                all_true.append(labels[int(true_index)])
                all_pred.append(labels[int(pred_index)])

    return total_loss / max(len(loader), 1), correct / max(total, 1), all_true, all_pred


def save_plots(history: dict[str, list[float]], loss_path: Path, accuracy_path: Path) -> None:
    save_history_plots(
        history=history,
        model_name="wav2vec2",
        legacy_loss_path=loss_path,
        legacy_accuracy_path=accuracy_path,
    )


def save_pretrained_model(
    model: Any,
    feature_extractor: Any,
    output_dir: Path,
    labels: tuple[str, ...] | list[str],
) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        feature_extractor.save_pretrained(output_dir)
        (output_dir / "labels.json").write_text(
            json.dumps(list(labels), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        err_text = str(exc).lower()
        if "not enough space" in err_text or "os error 112" in err_text:
            logger.error(
                "ディスク容量不足のためモデルの保存に失敗しました (%s)。Cドライブの空き容量を確保してください。",
                output_dir,
            )
        else:
            logger.error("モデルの保存中にエラーが発生しました (%s): %s", output_dir, exc)
        raise


def import_transformers() -> tuple[Any, Any, Any, Any]:
    try:
        from transformers import (
            AutoConfig,
            AutoFeatureExtractor,
            Wav2Vec2ForSequenceClassification,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:
        raise ImportError(
            "Wav2Vec2 training requires the 'transformers' package. "
            "Install project dependencies with: uv sync"
        ) from exc

    return (
        AutoFeatureExtractor,
        AutoConfig,
        Wav2Vec2ForSequenceClassification,
        get_linear_schedule_with_warmup,
    )


def is_package_available(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def is_pagefile_or_memory_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        isinstance(exc, MemoryError)
        or "os error 1455" in text
        or "paging file" in text
        or "ページング ファイル" in text
    )


def build_model_load_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {"use_safetensors": True}
    if is_package_available("accelerate"):
        kwargs["low_cpu_mem_usage"] = True
    else:
        logger.info(
            "accelerate is not installed; local Wav2Vec2 checkpoints will use "
            "streaming safetensors fallback if regular loading runs out of memory."
        )
    return kwargs


def get_local_safetensor_files(model_source: str | Path) -> list[Path]:
    model_path = Path(model_source)
    if not model_path.is_dir():
        return []

    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index_data.get("weight_map", {})
        filenames = sorted(set(weight_map.values()))
        return [model_path / filename for filename in filenames]

    safetensors_path = model_path / "model.safetensors"
    if safetensors_path.exists():
        return [safetensors_path]

    return []


def ensure_sharded_safetensors(
    model_source: str | Path,
    max_shard_size_mb: int = 16,
) -> list[Path]:
    model_path = Path(model_source)
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        return get_local_safetensor_files(model_path)

    source_path = model_path / "model.safetensors"
    if not source_path.exists():
        return []

    try:
        from safetensors import safe_open
        from safetensors.numpy import save_file
    except ImportError as exc:
        raise ImportError(
            "Sharding Wav2Vec2 checkpoints requires the 'safetensors' package."
        ) from exc

    max_shard_bytes = max(1, max_shard_size_mb) * 1024 * 1024
    if source_path.stat().st_size <= max_shard_bytes:
        return [source_path]

    for partial_shard in model_path.glob("model-shard-*.safetensors"):
        partial_shard.unlink(missing_ok=True)

    logger.info(
        "Creating sharded safetensors checkpoint for low-memory loading: %s",
        source_path,
    )
    weight_map: dict[str, str] = {}
    shard_files: list[Path] = []
    total_size = 0
    shard_index = 1
    current_tensors: dict[str, np.ndarray] = {}
    current_size = 0

    def flush_shard() -> None:
        nonlocal current_tensors, current_size, shard_index
        if not current_tensors:
            return

        shard_name = f"model-shard-{shard_index:05d}.safetensors"
        shard_path = model_path / shard_name
        save_file(current_tensors, shard_path)
        for tensor_name in current_tensors:
            weight_map[tensor_name] = shard_name
        shard_files.append(shard_path)
        current_tensors = {}
        current_size = 0
        shard_index += 1
        gc.collect()

    with safe_open(str(source_path), framework="numpy") as tensors:
        for key in tensors.keys():
            tensor = tensors.get_tensor(key)
            tensor_size = tensor.nbytes
            if current_tensors and current_size + tensor_size > max_shard_bytes:
                flush_shard()

            current_tensors[key] = tensor
            current_size += tensor_size
            total_size += tensor_size

            # Release each mmap-backed tensor as soon as it has been written.
            # On Windows, grouping many tensors can still trip os error 1455.
            flush_shard()

    index_payload = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    index_path.write_text(
        json.dumps(index_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Sharded safetensors checkpoint created: %d shard(s), index=%s",
        len(shard_files),
        index_path,
    )
    return shard_files


def load_local_safetensors_streaming(
    model: torch.nn.Module,
    model_source: str | Path,
) -> tuple[list[str], list[str], list[str]]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError("Streaming Wav2Vec2 loading requires the 'safetensors' package.") from exc

    tensor_files = get_local_safetensor_files(model_source)
    if len(tensor_files) == 1 and tensor_files[0].name == "model.safetensors":
        tensor_files = ensure_sharded_safetensors(model_source)
    if not tensor_files:
        raise FileNotFoundError(
            f"No local safetensors checkpoint files were found in {model_source}"
        )

    state_dict = model.state_dict()
    loaded: list[str] = []
    unexpected: list[str] = []
    mismatched: list[str] = []

    with torch.no_grad():
        for tensor_file in tensor_files:
            with safe_open(str(tensor_file), framework="pt", device="cpu") as tensors:
                for key in tensors.keys():
                    if key not in state_dict:
                        unexpected.append(key)
                        continue

                    target_tensor = state_dict[key]
                    source_tensor = tensors.get_tensor(key)
                    if tuple(source_tensor.shape) != tuple(target_tensor.shape):
                        mismatched.append(key)
                        continue

                    target_tensor.copy_(source_tensor)
                    loaded.append(key)

    loaded_set = set(loaded)
    missing = [key for key in state_dict if key not in loaded_set]
    logger.info(
        "Streaming safetensors load finished: loaded=%d, missing=%d, mismatched=%d, unexpected=%d",
        len(loaded),
        len(missing),
        len(mismatched),
        len(unexpected),
    )
    if mismatched:
        logger.info(
            "Skipped mismatched tensors, usually classifier heads after label changes: %s",
            ", ".join(mismatched[:10]),
        )
    return missing, mismatched, unexpected


def instantiate_model_for_streaming(model_class: Any, config: Any) -> torch.nn.Module:
    if hasattr(torch.nn.Module, "to_empty"):
        try:
            with torch.device("meta"):
                model = model_class(config)
            model = model.to_empty(device=torch.device("cpu"))
            logger.info("Instantiated Wav2Vec2 model with meta tensors for low-memory loading.")
            return model
        except Exception as exc:
            if is_pagefile_or_memory_error(exc):
                raise
            logger.warning(
                "Meta-device model initialization failed (%s). "
                "Falling back to regular initialization.",
                exc,
            )

    return model_class(config)


def initialize_unloaded_classification_head(
    model: torch.nn.Module,
    unloaded_names: list[str],
) -> None:
    if not unloaded_names or not hasattr(model, "_init_weights"):
        return

    for module_name in ("projector", "classifier"):
        if not any(
            name == module_name or name.startswith(f"{module_name}.") for name in unloaded_names
        ):
            continue

        module = getattr(model, module_name, None)
        init_fn = getattr(model, "_init_weights", None)
        if isinstance(module, torch.nn.Module) and callable(init_fn):
            init_fn(module)
            logger.info("Initialized Wav2Vec2 %s parameters.", module_name)


def load_wav2vec2_classifier(
    model_class: Any,
    config_class: Any,
    model_source: str | Path,
    labels: tuple[str, ...] | list[str],
    label2id: dict[str, int],
    id2label: dict[int, str],
) -> Any:
    common_kwargs = {
        "num_labels": len(labels),
        "label2id": label2id,
        "id2label": id2label,
        "problem_type": "single_label_classification",
        "ignore_mismatched_sizes": True,
    }
    load_kwargs = build_model_load_kwargs()
    local_safetensors = get_local_safetensor_files(model_source)

    if local_safetensors and not is_package_available("accelerate"):
        ensure_sharded_safetensors(model_source)
        config = config_class.from_pretrained(model_source)
        config.num_labels = len(labels)
        config.label2id = label2id
        config.id2label = id2label
        config.problem_type = "single_label_classification"

        model = instantiate_model_for_streaming(model_class, config)
        missing, mismatched, _ = load_local_safetensors_streaming(model, model_source)
        initialize_unloaded_classification_head(model, missing + mismatched)
        return model

    try:
        return model_class.from_pretrained(
            model_source,
            **common_kwargs,
            **load_kwargs,
        )
    except Exception as exc:
        if not is_pagefile_or_memory_error(exc):
            raise

        local_safetensors = get_local_safetensor_files(model_source)
        if not local_safetensors:
            raise

        logger.warning(
            "Regular Wav2Vec2 checkpoint loading ran out of memory (%s). "
            "Retrying with streaming safetensors loader.",
            exc,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ensure_sharded_safetensors(model_source)
        config = config_class.from_pretrained(model_source)
        config.num_labels = len(labels)
        config.label2id = label2id
        config.id2label = id2label
        config.problem_type = "single_label_classification"

        model = instantiate_model_for_streaming(model_class, config)
        missing, mismatched, _ = load_local_safetensors_streaming(model, model_source)
        initialize_unloaded_classification_head(model, missing + mismatched)
        return model


def freeze_wav2vec2_layers(
    model: Any,
    freeze_feature_encoder: bool = True,
    freeze_transformer_layers: int = 10,
) -> None:
    """
    Wav2Vec2 の下位レイヤーをフリーズ（勾配計算対象外化）して学習を高速化・過学習防止する。

    ・Feature Encoder (7層CNN): 常にフリーズ
    ・Transformer Encoder (12層のうち下位 10 層): 勾配計算をオフ (requires_grad = False)
    ・最上位 2 層 + Classifier Head: 勾配計算を継続 (requires_grad = True)
    """
    if freeze_feature_encoder and hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
        logger.info("Wav2Vec2 feature encoder (7-layer CNN) is frozen.")

    if (
        freeze_transformer_layers > 0
        and hasattr(model, "wav2vec2")
        and hasattr(model.wav2vec2, "encoder")
    ):
        layers = model.wav2vec2.encoder.layers
        num_total_layers = len(layers)
        num_frozen = min(freeze_transformer_layers, num_total_layers)
        for i in range(num_frozen):
            for param in layers[i].parameters():
                param.requires_grad = False
        logger.info(
            "Wav2Vec2 Transformer 下位 %d 層 (全 %d 層中) をフリーズしました。上位 %d 層と分類ヘッドのみ学習します。",
            num_frozen,
            num_total_layers,
            num_total_layers - num_frozen,
        )


def train(args: argparse.Namespace) -> None:
    ensure_merged_and_preprocessed(skip_prep=getattr(args, "skip_prep", False))

    resume_from_arg = getattr(args, "resume_from", None)
    best_model_path = getattr(
        args, "best_model_path", DEFAULT_RECOGNITION_CONFIG.wav2vec2_best_model_dir
    )
    last_model_path = getattr(
        args, "last_model_path", DEFAULT_RECOGNITION_CONFIG.wav2vec2_last_model_dir
    )
    resume = getattr(args, "resume", True)

    target_resume_path: Path | None = None

    if resume_from_arg:
        # 1. --resume-from で明示的にパスが指定された場合: HFダウンロードをスキップし、指定パスを直接ロード
        target_resume_path = Path(resume_from_arg)
        if not target_resume_path.exists():
            raise FileNotFoundError(
                f"指定されたチェックポイントが見つかりません: {target_resume_path}"
            )
        logger.info(
            "指定されたローカルチェックポイント (%s) から学習を再開します。(HFダウンロード送信なし)",
            target_resume_path,
        )
    elif resume:
        # 2. 通常の実行: HFからチーム共有最新モデルを自動ダウンロードして同期
        from voicerecognizer.utils.model_uploader import download_latest_team_weights_if_needed

        download_latest_team_weights_if_needed(model_type="wav2vec2")

        if (
            best_model_path
            and best_model_path.exists()
            and (
                (best_model_path / "model.safetensors").exists()
                or (best_model_path / "pytorch_model.bin").exists()
            )
        ):
            target_resume_path = best_model_path
        elif (
            last_model_path
            and last_model_path.exists()
            and (
                (last_model_path / "model.safetensors").exists()
                or (last_model_path / "pytorch_model.bin").exists()
            )
        ):
            target_resume_path = last_model_path

    seed = getattr(args, "seed", 42)
    root_dir = (
        DEFAULT_RECOGNITION_CONFIG.processed_dataset_dir
        if DEFAULT_RECOGNITION_CONFIG.processed_dataset_dir.exists()
        else DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir
    )
    sample_rate = getattr(args, "sample_rate", DEFAULT_AUDIO_CONFIG.sample_rate)
    target_length_seconds = getattr(
        args, "target_length_seconds", DEFAULT_RECOGNITION_CONFIG.target_length_seconds
    )
    top_db = getattr(args, "top_db", DEFAULT_PREPROCESS_CONFIG.top_db)
    val_rate = getattr(args, "val_rate", 0.2)
    target_acc = getattr(args, "target_acc", 0.97)
    num_workers_arg = getattr(args, "num_workers", None)
    num_workers = determine_optimal_num_workers(num_workers_arg)
    logger.info(
        "DataLoader の並列ワーカー数 (num_workers) を自動動的設定しました: %d (CPUコア数: %d)",
        num_workers,
        os.cpu_count() or 1,
    )
    pretrained_model_name = getattr(
        args, "pretrained_model_name", DEFAULT_RECOGNITION_CONFIG.wav2vec2_pretrained_model_name
    )
    weight_decay = getattr(args, "weight_decay", 0.01)
    warmup_ratio = getattr(args, "warmup_ratio", 0.1)
    max_grad_norm = getattr(args, "max_grad_norm", 1.0)
    freeze_feature_encoder = True

    fix_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        auto_feature_extractor_cls,
        auto_config_cls,
        wav2vec2_model_cls,
        get_linear_schedule_with_warmup,
    ) = import_transformers()

    dataset = Wav2Vec2ClassificationDataset(
        root_dir=root_dir,
        sample_rate=sample_rate,
        target_length_seconds=target_length_seconds,
        top_db=top_db,
    )
    train_subset, val_dataset = split_dataset(
        dataset,
        val_rate,
        seed,
        speaker_aware=getattr(args, "speaker_aware_split", False),
    )

    use_class_weights = getattr(args, "use_class_weights", True)
    augment = getattr(args, "augment", True)
    class_weight_power = getattr(args, "class_weight_power", 0.5)

    if augment:
        augmentation_noise_files = collect_augmentation_noise_files(
            getattr(args, "augmentation_noise_dir", None)
        )
        logger.info(
            "Audio Data Augmentation "
            "(ノイズ加算・音量変調・タイムシフト・軽い速度/ピッチ変化・実機ノイズ混合) "
            "を訓練データセットに有効化しました。実機ノイズ=%d件",
            len(augmentation_noise_files),
        )
        train_augmentor = AudioAugmentor(
            sample_rate=sample_rate,
            noise_file_paths=augmentation_noise_files,
        )
        train_dataset = AugmentedSubset(train_subset, train_augmentor)
    else:
        train_dataset = AugmentedSubset(train_subset, augmentor=None)

    train_labels = [dataset.data[i][1] for i in train_subset.indices]

    loss_fct: torch.nn.Module | None = None
    if use_class_weights:
        class_weights = compute_class_weights(
            train_labels, num_classes=len(dataset.labels), power=class_weight_power
        ).to(device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        logger.info(
            "クラス重み付き損失関数 (Weighted CrossEntropyLoss) を有効化しました。(power=%.2f, num_classes=%d)",
            class_weight_power,
            len(dataset.labels),
        )

    has_weights = target_resume_path is not None
    labels_matched = (
        is_labels_compatible(target_resume_path, dataset.labels) if target_resume_path else False
    )
    model_source = pretrained_model_name

    if target_resume_path and has_weights:
        model_source = str(target_resume_path)
        if labels_matched:
            logger.info(
                "チーム共有/ローカルチェックポイント (%s) から学習を開始します。(全 %d クラス)",
                model_source,
                len(dataset.labels),
            )
        else:
            logger.info(
                "旧モデル (%s) の音声特徴表現 (CNN+Transformer) を引き継ぎつつ、分類層を新しいクラス数 (%dクラス) に自動拡張してファインチューニングを開始します。",
                model_source,
                len(dataset.labels),
            )
    elif not resume:
        logger.info(
            "=== [--from-scratch が指定されたため、ベースモデル (%s) から新規ファインチューニングを開始します (全 %d クラス)] ===",
            model_source,
            len(dataset.labels),
        )
    else:
        logger.info(
            "過去のチェックポイントが存在しないため、ベースモデル (%s) から新規学習を開始します (全 %d クラス)。",
            model_source,
            len(dataset.labels),
        )

    feature_extractor = auto_feature_extractor_cls.from_pretrained(model_source)
    feature_sample_rate = getattr(feature_extractor, "sampling_rate", None)
    if feature_sample_rate and feature_sample_rate != sample_rate:
        logger.warning(
            "Feature extractor expects %s Hz but training uses %s Hz",
            feature_sample_rate,
            sample_rate,
        )

    collate_fn = build_collate_fn(feature_extractor, sample_rate)

    use_balanced_sampler = getattr(args, "use_balanced_sampler", True)
    train_sampler = None
    if use_balanced_sampler:
        confusion_label_multipliers: dict[int, float] = {}
        if getattr(args, "use_confusion_pair_sampler", True):
            confusion_label_multipliers = load_confusion_label_multipliers(
                getattr(
                    args,
                    "confusion_pair_evaluation_result",
                    PROJECT_ROOT / "evaluation_results" / "evaluation_result.json",
                ),
                dataset.labels,
                min_count=getattr(args, "confusion_pair_min_count", 3),
                max_pairs=getattr(args, "confusion_pair_max_pairs", 20),
                boost=getattr(args, "confusion_pair_boost", 0.5),
            )
        sample_weights = compute_balanced_sampler_weights(
            train_labels,
            num_classes=len(dataset.labels),
            power=0.5,
            confusion_label_multipliers=confusion_label_multipliers,
        )
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        logger.info(
            "マイルド全クラスサンプラー (WeightedRandomSampler: power=0.5) を有効化しました。"
            "必要に応じて混同ペア重点倍率も上乗せします。"
        )

    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    label2id = {label: index for index, label in enumerate(dataset.labels)}
    id2label = {index: label for label, index in label2id.items()}
    model = load_wav2vec2_classifier(
        wav2vec2_model_cls,
        auto_config_cls,
        model_source,
        dataset.labels,
        label2id,
        id2label,
    )
    freeze_wav2vec2_layers(
        model,
        freeze_feature_encoder=freeze_feature_encoder,
        freeze_transformer_layers=args.freeze_transformer_layers,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=weight_decay,
    )
    training_steps = max(len(train_loader) * args.epochs, 1)
    warmup_steps = int(training_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=training_steps,
    )
    best_macro_f1 = -1.0

    # チーム最高精度 (best_model_path) のベースラインスコアを事前評価してハードルに設定する
    if (
        best_model_path
        and best_model_path.exists()
        and is_labels_compatible(best_model_path, dataset.labels)
    ):
        try:
            baseline_model = load_wav2vec2_classifier(
                wav2vec2_model_cls,
                auto_config_cls,
                best_model_path,
                dataset.labels,
                label2id,
                id2label,
            ).to(device)
            _, val_acc, val_true, val_pred = validate(
                baseline_model, val_loader, device, labels=dataset.labels
            )
            init_result = compute_evaluation_result(val_true, val_pred, labels=dataset.labels)
            best_macro_f1 = init_result.overall.macro_f1
            logger.info(
                "チーム最高精度モデル (%s) のベースラインスコア - Val Acc: %.4f, Val Macro-F1: %.4f",
                best_model_path,
                val_acc,
                best_macro_f1,
            )
            del baseline_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning("既存ベストモデルの事前評価に失敗しました: %s", e)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
    }
    run_best_macro_f1 = -1.0
    is_best_updated = False
    scaler = GradScaler() if device.type == "cuda" else None

    patience = getattr(args, "patience", 5)
    patience_counter = 0

    interrupted = False
    try:
        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                epoch,
                args.epochs,
                max_grad_norm,
                loss_fct=loss_fct,
                scaler=scaler,
            )
            val_loss, val_acc, val_true, val_pred = validate(
                model, val_loader, device, labels=dataset.labels
            )
            eval_result = compute_evaluation_result(val_true, val_pred, labels=dataset.labels)
            macro_f1 = eval_result.overall.macro_f1

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["val_macro_f1"].append(macro_f1)

            logger.info(
                "Epoch %d/%d - Train Loss: %.4f, Train Acc: %.4f | "
                "Val Loss: %.4f, Val Acc: %.4f, Val Macro-F1: %.4f",
                epoch + 1,
                args.epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                macro_f1,
            )

            # 今回の学習ランにおける最高精度 (Last Training Best) を更新した場合
            if macro_f1 > run_best_macro_f1:
                run_best_macro_f1 = macro_f1
                save_pretrained_model(
                    model,
                    feature_extractor,
                    last_model_path,
                    dataset.labels,
                )
                logger.info(
                    "今回の学習ランの最高精度モデル (Last) を更新・保存しました: %s (Val Macro-F1: %.4f)",
                    last_model_path,
                    run_best_macro_f1,
                )

            # チーム共有・歴代最高精度 (Global Best) を更新した場合
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                is_best_updated = True
                patience_counter = 0
                save_pretrained_model(
                    model,
                    feature_extractor,
                    best_model_path,
                    dataset.labels,
                )
                logger.info(
                    "歴代最高精度モデル (Best) を更新・保存しました: %s (Val Macro-F1: %.4f, Val Acc: %.4f)",
                    best_model_path,
                    best_macro_f1,
                    val_acc,
                )
            else:
                patience_counter += 1
                if patience > 0 and patience_counter >= patience:
                    logger.info(
                        "Early stopping: Validation Macro-F1 が %d エポック連続で向上しなかったため、頭打ちと判断して学習を自動終了します (Run Best: %.4f, Global Best: %.4f)",
                        patience,
                        run_best_macro_f1,
                        best_macro_f1,
                    )
                    break

            if val_acc >= target_acc:
                logger.info(
                    "Target validation accuracy reached: %.2f%%",
                    target_acc * 100,
                )
                break

    except KeyboardInterrupt:
        interrupted = True
        logger.warning("ユーザー操作 (Ctrl+C / SIGINT) により学習が途中で中断されました。")
        if run_best_macro_f1 < 0:
            # 1エポックも完了前に中断された場合は現在モデルを保存
            save_pretrained_model(
                model,
                feature_extractor,
                last_model_path,
                dataset.labels,
            )

    if history["train_loss"]:
        save_history_plots(
            history=history,
            model_name="wav2vec2",
            num_classes=len(dataset.labels),
            num_samples=len(dataset),
        )

    if interrupted:
        logger.info(
            "中断処理が正常に完了しました。保存されたチェックポイント (%s) から `--resume-from %s` で学習を再開可能です。",
            last_model_path,
            last_model_path,
        )
        return

    logger.info(
        "Training finished. Last Wav2Vec2 model saved to %s",
        last_model_path,
    )

    # ONNX 自動エクスポート ＆ 量子化
    no_onnx_export = getattr(args, "no_onnx_export", False)
    if not no_onnx_export:
        # 1. ベストモデルのエクスポート ＆ ベンチマーク実行（1回のみ）
        if best_model_path and best_model_path.exists():
            try:
                logger.info(
                    "学習完了後の Wav2Vec2 ONNX (best) エクスポート ＆ INT8 量子化を開始します..."
                )
                export_and_benchmark(model_dir=best_model_path)
            except Exception as e:
                logger.error("ONNX 自動エクスポート (best) 中にエラーが発生しました: %s", e)

        # 2. 最終チェックポイント (last) のスマート同期
        if last_model_path and last_model_path.exists() and last_model_path != best_model_path:
            best_safetensors = best_model_path / "model.safetensors"
            last_safetensors = last_model_path / "model.safetensors"

            is_identical = (
                best_safetensors.exists()
                and last_safetensors.exists()
                and filecmp.cmp(best_safetensors, last_safetensors, shallow=False)
            )

            if is_identical:
                logger.info(
                    "wav2vec2_best と wav2vec2_last の重みが同一のため、ONNX 成果物をコピーして即時同期します..."
                )
                for onnx_file in best_model_path.glob("*.onnx"):
                    shutil.copy2(onnx_file, last_model_path / onnx_file.name)
                logger.info("wav2vec2_last への ONNX ファイルの高速同期が完了しました。")
            else:
                try:
                    logger.info(
                        "wav2vec2_last の重みが best と異なるため、個別に ONNX エクスポートを実行します..."
                    )
                    export_and_benchmark(model_dir=last_model_path, skip_benchmark=True)
                except Exception as e:
                    logger.error("ONNX 自動エクスポート (last) 中にエラーが発生しました: %s", e)

    # Hugging Face 自動アップロード判定 (チーム最高精度を更新し、アップロードが許可されている場合のみ)
    hf_upload = getattr(args, "hf_upload", True)
    if is_best_updated and hf_upload:
        from voicerecognizer.utils.model_uploader import upload_weights_to_hf

        upload_weights_to_hf(model_type="wav2vec2")
    elif is_best_updated and not hf_upload:
        logger.info(
            "--no-hf-upload が指定されたため、Hugging Face へのアップロードをスキップします。"
        )
    else:
        logger.info(
            "チーム最高精度が更新されなかったため、Hugging Face へのアップロードをスキップします。"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune Wav2Vec2 for hiragana label classification."
    )
    parser.add_argument(
        "--epochs",
        "--epoch",
        type=int,
        default=30,
        help="Number of training epochs (default: 30)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
        help="Learning rate (default: 3e-5)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Number of epochs with no validation improvement after which training stops early (default: 5, set 0 to disable)",
    )
    parser.add_argument(
        "--freeze-transformer-layers",
        type=int,
        default=10,
        help="Number of bottom Transformer layers to freeze (default: 10 out of 12)",
    )
    parser.add_argument(
        "--from-scratch",
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Train from base pretrained model without reusing existing checkpoint",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to specific checkpoint directory to resume from (e.g. weights/wav2vec2_last). Skips HF download when specified.",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_false",
        dest="use_class_weights",
        default=True,
        help="Disable class weighting for loss calculation (enabled by default)",
    )
    parser.add_argument(
        "--no-augment",
        action="store_false",
        dest="augment",
        default=True,
        help="Disable training audio data augmentation (enabled by default)",
    )
    parser.add_argument(
        "--augmentation-noise-dir",
        type=Path,
        default=None,
        help="Optional wav file or directory of recorded device/background noise for augmentation",
    )
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=0.5,
        help="Power exponent for smooth inverse class weighting (default: 0.5)",
    )
    parser.add_argument(
        "--skip-prep",
        action="store_true",
        help="Skip automatic data merging and preprocessing before training",
    )
    parser.add_argument(
        "--no-onnx-export",
        action="store_true",
        help="Skip automatic ONNX export and INT8 quantization after training",
    )
    parser.add_argument(
        "--no-hf-upload",
        action="store_false",
        dest="hf_upload",
        default=True,
        help="Skip Hugging Face upload even if the best model is updated",
    )
    parser.add_argument(
        "--no-balanced-sampler",
        action="store_false",
        dest="use_balanced_sampler",
        help="Disable WeightedRandomSampler for class-balanced training",
    )
    parser.add_argument(
        "--speaker-aware-split",
        action="store_true",
        default=False,
        help="Split validation by speaker groups using processed_dataset/index.csv speaker metadata",
    )
    parser.add_argument(
        "--no-confusion-pair-sampler",
        action="store_false",
        dest="use_confusion_pair_sampler",
        default=True,
        help="Disable extra sampling weight for labels involved in past confusion pairs",
    )
    parser.add_argument(
        "--confusion-pair-evaluation-result",
        type=Path,
        default=PROJECT_ROOT / "evaluation_results" / "evaluation_result.json",
        help="Evaluation JSON containing the confusion matrix used by the sampler",
    )
    parser.add_argument(
        "--confusion-pair-min-count",
        type=int,
        default=3,
        help="Minimum off-diagonal confusion count to include in the sampler",
    )
    parser.add_argument(
        "--confusion-pair-max-pairs",
        type=int,
        default=20,
        help="Maximum number of confusion pairs to boost",
    )
    parser.add_argument(
        "--confusion-pair-boost",
        type=float,
        default=0.5,
        help="Maximum extra sampling multiplier for the hardest confusion label",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of DataLoader background worker processes (default: auto-detected based on CPU cores)",
    )
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
