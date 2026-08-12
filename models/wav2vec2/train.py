"""
Wav2Vec2 Fine-Tuning Script with Layer Freezing and Lazy Disk Loading

役割:
  Wav2Vec2 プリトレイニードモデルの下位層フリーズ ＋ processed_dataset からのlazy loadingにより、
  RAM使用量を抑えてファインチューニングを実行し、best_model ディレクトリおよび labels.json を保存します。

使い方:
  uv run python models/wav2vec2/train.py           # デフォルト設定で Wav2Vec2 を継続ファインチューニング
  uv run python models/wav2vec2/train.py --no-resume # ベースモデル (facebook/wav2vec2-base) から新規学習
"""

import argparse
from collections import Counter
import gc
import importlib.util
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from tqdm import tqdm

from config import DEFAULT_RECOGNITION_CONFIG
from dataset.hiragana_dataset import HiraganaDataset
from evaluation.evaluator import compute_evaluation_result
from preprocessing.audio_augmentor import AudioAugmentor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
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


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


from utils.split_helper import safe_stratified_split


def split_dataset(
    dataset: Wav2Vec2ClassificationDataset, val_rate: float, seed: int
) -> tuple[Subset, Subset]:
    labels = [label for _, label in dataset.data]
    train_idx, val_idx = safe_stratified_split(labels, val_rate=val_rate, seed=seed)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


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
        waveform, label = self.subset[index]
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


def build_collate_fn(feature_extractor: Any, sample_rate: int) -> Callable:
    def collate(batch: list[tuple[np.ndarray, int]]) -> dict[str, torch.Tensor]:
        waveforms = [waveform for waveform, _ in batch]
        labels = torch.tensor([label for _, label in batch], dtype=torch.long)
        inputs = feature_extractor(
            waveforms,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs["labels"] = labels
        return inputs

    return collate


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
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
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    progress = tqdm(loader)

    for batch in progress:
        batch = move_batch(batch, device)

        optimizer.zero_grad()
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

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            pred = outputs.logits.argmax(dim=-1)
            batch_labels = batch["labels"]

            total_loss += loss.item()
            correct += (pred == batch_labels).sum().item()
            total += batch_labels.size(0)

            for true_index, pred_index in zip(
                batch_labels.cpu().numpy(), pred.cpu().numpy()
            ):
                all_true.append(labels[int(true_index)])
                all_pred.append(labels[int(pred_index)])

    return total_loss / max(len(loader), 1), correct / max(total, 1), all_true, all_pred


from utils.plot_saver import save_history_plots


def save_plots(
    history: dict[str, list[float]], loss_path: Path, accuracy_path: Path
) -> None:
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
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    feature_extractor.save_pretrained(output_dir)
    (output_dir / "labels.json").write_text(
        json.dumps(list(labels), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def import_transformers() -> tuple[Any, Any, Any, Any]:
    try:
        from transformers import (
            AutoFeatureExtractor,
            AutoConfig,
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

    max_shard_bytes = max(1, int(max_shard_size_mb)) * 1024 * 1024
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
        raise ImportError(
            "Streaming Wav2Vec2 loading requires the 'safetensors' package."
        ) from exc

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
        "Streaming safetensors load finished: loaded=%d, missing=%d, "
        "mismatched=%d, unexpected=%d",
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
            logger.info(
                "Instantiated Wav2Vec2 model with meta tensors for low-memory loading."
            )
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
            name == module_name or name.startswith(f"{module_name}.")
            for name in unloaded_names
        ):
            continue

        module = getattr(model, module_name, None)
        if isinstance(module, torch.nn.Module):
            model._init_weights(module)
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

    if freeze_transformer_layers > 0 and hasattr(model, "wav2vec2") and hasattr(model.wav2vec2, "encoder"):
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


from preprocessing.dataset_builder import ensure_merged_and_preprocessed


def train(args: argparse.Namespace) -> None:
    ensure_merged_and_preprocessed(skip_prep=getattr(args, "skip_prep", False))

    resume_from_arg = getattr(args, "resume_from", None)
    best_model_path = getattr(args, "best_model_path", DEFAULT_RECOGNITION_CONFIG.wav2vec2_best_model_dir)
    last_model_path = getattr(args, "last_model_path", DEFAULT_RECOGNITION_CONFIG.wav2vec2_last_model_dir)
    resume = getattr(args, "resume", True)

    target_resume_path: Path | None = None

    if resume_from_arg:
        # 1. --resume-from で明示的にパスが指定された場合: HFダウンロードをスキップし、指定パスを直接ロード
        target_resume_path = Path(resume_from_arg)
        if not target_resume_path.exists():
            raise FileNotFoundError(f"指定されたチェックポイントが見つかりません: {target_resume_path}")
        logger.info("📌 指定されたローカルチェックポイント (%s) から学習を再開します。(HFダウンロード送信なし)", target_resume_path)
    elif resume:
        # 2. 通常の実行: HFからチーム共有最新モデルを自動ダウンロードして同期
        from utils.model_uploader import download_latest_team_weights_if_needed
        download_latest_team_weights_if_needed(model_type="wav2vec2")

        if best_model_path.exists() and ((best_model_path / "model.safetensors").exists() or (best_model_path / "pytorch_model.bin").exists()):
            target_resume_path = best_model_path
        elif last_model_path.exists() and ((last_model_path / "model.safetensors").exists() or (last_model_path / "pytorch_model.bin").exists()):
            target_resume_path = last_model_path

    seed = getattr(args, "seed", 42)
    root_dir = (
        DEFAULT_RECOGNITION_CONFIG.processed_dataset_dir
        if DEFAULT_RECOGNITION_CONFIG.processed_dataset_dir.exists()
        else DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir
    )
    sample_rate = getattr(args, "sample_rate", DEFAULT_RECOGNITION_CONFIG.sample_rate)
    target_length_seconds = getattr(args, "target_length_seconds", DEFAULT_RECOGNITION_CONFIG.target_length_seconds)
    top_db = getattr(args, "top_db", DEFAULT_RECOGNITION_CONFIG.top_db)
    val_rate = getattr(args, "val_rate", 0.2)
    target_acc = getattr(args, "target_acc", 0.97)
    num_workers = getattr(args, "num_workers", 0)
    pretrained_model_name = getattr(args, "pretrained_model_name", DEFAULT_RECOGNITION_CONFIG.wav2vec2_pretrained_model_name)
    weight_decay = getattr(args, "weight_decay", 0.01)
    warmup_ratio = getattr(args, "warmup_ratio", 0.1)
    max_grad_norm = getattr(args, "max_grad_norm", 1.0)
    freeze_feature_encoder = True
    loss_plot_path = None
    accuracy_plot_path = None

    fix_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        AutoFeatureExtractor,
        AutoConfig,
        Wav2Vec2ForSequenceClassification,
        get_linear_schedule_with_warmup,
    ) = import_transformers()

    dataset = Wav2Vec2ClassificationDataset(
        root_dir=root_dir,
        sample_rate=sample_rate,
        target_length_seconds=target_length_seconds,
        top_db=top_db,
    )
    train_dataset, val_dataset = split_dataset(dataset, val_rate, seed)

    use_class_weights = getattr(args, "use_class_weights", True)
    augment = getattr(args, "augment", True)
    class_weight_power = getattr(args, "class_weight_power", 0.5)

    if augment:
        logger.info("Audio Data Augmentation (ノイズ加算・音量変調・タイムシフト) を訓練データセットに有効化しました。")
        train_augmentor = AudioAugmentor()
        train_dataset = AugmentedSubset(train_dataset, train_augmentor)
    else:
        train_dataset = AugmentedSubset(train_dataset, augmentor=None)

    raw_train_subset = train_dataset.subset if isinstance(train_dataset, AugmentedSubset) else train_dataset
    train_labels = [dataset.data[i][1] for i in raw_train_subset.indices]

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
    labels_matched = is_labels_compatible(target_resume_path, dataset.labels) if target_resume_path else False
    model_source = pretrained_model_name

    if target_resume_path and has_weights:
        model_source = str(target_resume_path)
        if labels_matched:
            logger.info("チーム共有/ローカルチェックポイント (%s) から学習を開始します。(全 %d クラス)", model_source, len(dataset.labels))
        else:
            logger.info(
                "💡 旧モデル (%s) の音声特徴表現 (CNN+Transformer) を引き継ぎつつ、分類層を新しいクラス数 (%dクラス) に自動拡張してファインチューニングを開始します。",
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

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_source)
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
        label_counts = Counter(train_labels)
        class_weights_dict = {lbl: 1.0 / (count ** 0.5) for lbl, count in label_counts.items()}
        sample_weights = [class_weights_dict[lbl] for lbl in train_labels]
        sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights_tensor,
            num_samples=len(sample_weights_tensor),
            replacement=True,
        )
        logger.info("⚖️ マイルド全クラスサンプラー (WeightedRandomSampler: power=0.5) を有効化しました。aiueo の正解率を維持しつつマイナー音もバランスよく学習します。")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    label2id = {label: index for index, label in enumerate(dataset.labels)}
    id2label = {index: label for label, index in label2id.items()}
    model = load_wav2vec2_classifier(
        Wav2Vec2ForSequenceClassification,
        AutoConfig,
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

    # ラベル構成が完全一致する場合のみ、旧モデルのベースラインスコアを事前評価する。
    # ラベルが変わった場合、旧分類器で評価しても無意味なのでスキップする。
    if has_weights and labels_matched and target_resume_path:
        try:
            _, val_acc, val_true, val_pred = validate(
                model, val_loader, device, labels=dataset.labels
            )
            init_result = compute_evaluation_result(
                val_true, val_pred, labels=dataset.labels
            )
            best_macro_f1 = init_result.overall.macro_f1
            logger.info(
                "保存済み Wav2Vec2 チェックポイント (%s) の評価スコア - Val Acc: %.4f, Val Macro-F1: %.4f",
                target_resume_path,
                val_acc,
                best_macro_f1,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning("既存 Wav2Vec2 モデルの評価に失敗しました: %s", e)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
    }
    is_best_updated = False

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
        )
        val_loss, val_acc, val_true, val_pred = validate(
            model, val_loader, device, labels=dataset.labels
        )
        eval_result = compute_evaluation_result(
            val_true, val_pred, labels=dataset.labels
        )
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

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            is_best_updated = True
            save_pretrained_model(
                model,
                feature_extractor,
                best_model_path,
                dataset.labels,
            )
            logger.info(
                "Best Wav2Vec2 model saved: %s (Val Macro-F1: %.4f, Val Acc: %.4f)",
                best_model_path,
                best_macro_f1,
                val_acc,
            )

        if val_acc >= target_acc:
            logger.info(
                "Target validation accuracy reached: %.2f%%",
                target_acc * 100,
            )
            break

    save_pretrained_model(
        model,
        feature_extractor,
        last_model_path,
        dataset.labels,
    )
    save_history_plots(
        history=history,
        model_name="wav2vec2",
        num_classes=len(dataset.labels),
        num_samples=len(dataset),
    )
    logger.info(
        "Training finished. Last Wav2Vec2 model saved to %s",
        last_model_path,
    )

    # ONNX 自動エクスポート ＆ 量子化
    no_onnx_export = getattr(args, "no_onnx_export", False)
    if not no_onnx_export and best_model_path.exists():
        try:
            logger.info("学習完了後の Wav2Vec2 ONNX エクスポート ＆ INT8 量子化を開始します...")
            from models.wav2vec2.export_onnx import export_and_benchmark
            export_and_benchmark(model_dir=best_model_path)
        except Exception as e:
            logger.error("ONNX 自動エクスポート中にエラーが発生しました: %s", e)

    # Hugging Face 自動アップロード判定 (チーム最高精度を更新した場合のみ)
    if is_best_updated:
        from utils.model_uploader import upload_weights_to_hf
        upload_weights_to_hf(model_type="wav2vec2")
    else:
        logger.info("チーム最高精度が更新されなかったため、Hugging Face へのアップロードをスキップします。")


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
        "--no-balanced-sampler",
        action="store_false",
        dest="use_balanced_sampler",
        help="Disable WeightedRandomSampler for class-balanced training",
    )
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
