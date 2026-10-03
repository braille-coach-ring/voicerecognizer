"""Teacher feature extractor using Whisper encoder for Knowledge Distillation."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperModel

from voicerecognizer.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_WHISPER_TEACHER_MODEL_ID = "openai/whisper-large-v3-turbo"


class WhisperTeacher:
    """Extracts rich acoustic embeddings from a pretrained Whisper Encoder."""

    def __init__(
        self,
        model_id: str = DEFAULT_WHISPER_TEACHER_MODEL_ID,
        device: torch.device | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            "Loading Whisper teacher encoder from '%s' on %s...",
            self.model_id,
            self.device,
        )

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.model_id)
        full_whisper = WhisperModel.from_pretrained(self.model_id)
        self.encoder = full_whisper.encoder.to(self.device)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.hidden_dim = self.encoder.config.d_model
        logger.info("Whisper teacher encoder ready. Hidden dimension: %d", self.hidden_dim)

    @torch.no_grad()
    def extract_embedding(self, waveform: np.ndarray, sample_rate: int = 16000) -> torch.Tensor:
        """Extract a 1D L2-normalized representation vector for a single waveform."""
        embeddings = self.extract_batch_embeddings([waveform], sample_rate=sample_rate)
        return embeddings[0]

    @torch.no_grad()
    def extract_batch_embeddings(
        self, waveforms: list[np.ndarray], sample_rate: int = 16000
    ) -> torch.Tensor:
        """Extract L2-normalized representation vectors for a batch of waveforms."""
        processed = [w.astype(np.float32) for w in waveforms]
        inputs = self.feature_extractor(
            processed,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device)
        encoder_outputs = self.encoder(input_features)
        hidden_states = encoder_outputs.last_hidden_state  # (B, T, D)

        # Average pool over time
        pooled = hidden_states.mean(dim=1)  # (B, D)
        normalized = F.normalize(pooled, p=2, dim=-1)
        return normalized.cpu()


def precompute_whisper_embeddings_cache(
    index_csv_path: Path,
    output_cache_path: Path,
    teacher: WhisperTeacher,
    sample_rate: int = 16000,
    target_length_seconds: float = 0.6,
    batch_size: int = 32,
    max_samples_per_class: int | None = None,
) -> Path:
    """Precomputes Whisper teacher embeddings for all audio samples in index_csv_path."""
    logger.info("Precomputing Whisper embeddings cache for: %s", index_csv_path)
    output_cache_path.parent.mkdir(parents=True, exist_ok=True)

    target_samples = int(sample_rate * target_length_seconds)
    audio_paths: list[Path] = []
    counts: dict[str, int] = {}

    with open(index_csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath_str = row.get("filepath") or row.get("source_filepath") or ""
            label_str = row.get("label") or ""
            if not filepath_str:
                continue

            if max_samples_per_class is not None and max_samples_per_class > 0:
                if counts.get(label_str, 0) >= max_samples_per_class:
                    continue
                counts[label_str] = counts.get(label_str, 0) + 1

            p = Path(filepath_str)
            if not (p.is_absolute() and p.exists()):
                if (index_csv_path.parent / p).exists():
                    p = index_csv_path.parent / p
                elif (PROJECT_ROOT / p).exists():
                    p = PROJECT_ROOT / p
                elif (PROJECT_ROOT / "processed_dataset" / p).exists():
                    p = PROJECT_ROOT / "processed_dataset" / p
                else:
                    continue
            audio_paths.append(p)

    logger.info("Found %d audio files to extract Whisper representations for.", len(audio_paths))

    cache: dict[str, torch.Tensor] = {}
    for i in tqdm(range(0, len(audio_paths), batch_size), desc="Precomputing Whisper Embeddings"):
        batch_paths = audio_paths[i : i + batch_size]
        batch_waveforms: list[np.ndarray] = []

        for p in batch_paths:
            waveform, _ = sf.read(p, dtype="float32", always_2d=False)
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)

            if len(waveform) > target_samples:
                waveform = waveform[:target_samples]
            elif len(waveform) < target_samples:
                waveform = np.pad(waveform, (0, target_samples - len(waveform)))
            batch_waveforms.append(waveform)

        embeddings = teacher.extract_batch_embeddings(batch_waveforms, sample_rate=sample_rate)

        for p, emb in zip(batch_paths, embeddings, strict=True):
            resolved_key = str(p.resolve())
            cache[resolved_key] = emb.clone()

    torch.save(cache, output_cache_path)
    logger.info("Saved Whisper embeddings cache with %d items to: %s", len(cache), output_cache_path)
    return output_cache_path
