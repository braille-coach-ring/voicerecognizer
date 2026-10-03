"""Teacher module for IPA Phoneme Distillation using facebook/wav2vec2-xlsr-53-espeak-cv-ft."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

logger = logging.getLogger(__name__)

DEFAULT_IPA_TEACHER_MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
IPA_VOCAB_SIZE = 392


class IPATeacher(nn.Module):
    """Heavyweight Teacher model providing dense IPA phoneme posterior distributions."""

    def __init__(
        self,
        model_id: str = DEFAULT_IPA_TEACHER_MODEL_ID,
        device: torch.device | str = "cpu",
        temperature: float = 2.0,
    ):
        super().__init__()
        self.model_id = model_id
        self.device = torch.device(device)
        self.temperature = temperature

        logger.info("Initializing IPATeacher model: %s on %s...", model_id, self.device)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(self.device)
        self.model.eval()

        # Freeze all teacher parameters
        for p in self.model.parameters():
            p.requires_grad = False

        self.vocab_size = getattr(self.model.config, "vocab_size", IPA_VOCAB_SIZE)
        logger.info("IPATeacher loaded successfully (vocab_size=%d).", self.vocab_size)

    @torch.no_grad()
    def extract_pooled_posteriors(
        self,
        waveforms: torch.Tensor | np.ndarray,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """Extract temporally pooled IPA phoneme soft probability distribution for each audio.

        Args:
            waveforms: Tensor or ndarray of shape (batch, time) or (time,)
            sample_rate: sampling rate (default 16000)

        Returns:
            posteriors: Tensor of shape (batch, vocab_size) representing soft phoneme distribution.
        """
        if isinstance(waveforms, np.ndarray):
            waveforms = torch.from_numpy(waveforms).float()
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)

        waveforms = waveforms.to(self.device)
        outputs = self.model(waveforms)
        logits = outputs.logits  # shape: (batch, time_frames, vocab_size)

        # Temperature scaling
        scaled_logits = logits / self.temperature

        # Compute frame posteriors then pool temporally, capturing full mora articulation
        frame_probs = softmax(scaled_logits, dim=-1)
        pooled_posteriors = torch.mean(frame_probs, dim=1)  # shape: (batch, vocab_size)

        return pooled_posteriors.cpu()


def precompute_ipa_posteriors_cache(
    index_csv_path: Path,
    output_cache_path: Path,
    teacher: IPATeacher,
    batch_size: int = 16,
    max_samples_per_class: int | None = None,
) -> Path:
    """Precompute and cache teacher posteriors for all dataset files to accelerate Student training."""
    import csv

    import soundfile as sf
    from tqdm import tqdm

    from voicerecognizer.config import PROJECT_ROOT

    logger.info(
        "Precomputing IPA teacher posteriors from %s -> %s", index_csv_path, output_cache_path
    )
    output_cache_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with open(index_csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if max_samples_per_class is not None and max_samples_per_class > 0:
        counts: dict[str, int] = {}
        filtered_rows = []
        for r in rows:
            lb = r.get("label", "")
            if counts.get(lb, 0) < max_samples_per_class:
                counts[lb] = counts.get(lb, 0) + 1
                filtered_rows.append(r)
        rows = filtered_rows
        logger.info("Filtered to %d samples (max %d per class).", len(rows), max_samples_per_class)

    posteriors_by_path: dict[str, torch.Tensor] = {}

    batch_wavs: list[torch.Tensor] = []
    batch_keys: list[str] = []

    target_samples = int(16000 * 0.6)  # 0.6s

    for row in tqdm(rows, desc="Teacher Posteriors Extraction"):
        filepath_str = row.get("filepath") or row.get("source_filepath") or ""
        if not filepath_str:
            continue
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

        try:
            waveform, _ = sf.read(p, dtype="float32", always_2d=False)
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)
            # Pad or trim to target_samples
            if len(waveform) > target_samples:
                waveform = waveform[:target_samples]
            elif len(waveform) < target_samples:
                waveform = np.pad(waveform, (0, target_samples - len(waveform)))

            batch_wavs.append(torch.from_numpy(waveform).float())
            batch_keys.append(str(p.resolve()))

            if len(batch_wavs) >= batch_size:
                batch_tensor = torch.stack(batch_wavs, dim=0)
                posts = teacher.extract_pooled_posteriors(batch_tensor)
                for k, post in zip(batch_keys, posts, strict=True):
                    posteriors_by_path[k] = post.half()  # Save in FP16 to minimize disk space
                batch_wavs = []
                batch_keys = []
        except Exception as e:
            logger.warning("Error reading %s: %s", p, e)

    if batch_wavs:
        batch_tensor = torch.stack(batch_wavs, dim=0)
        posts = teacher.extract_pooled_posteriors(batch_tensor)
        for k, post in zip(batch_keys, posts, strict=True):
            posteriors_by_path[k] = post.half()

    torch.save(posteriors_by_path, output_cache_path)
    logger.info(
        "Successfully cached %d posteriors into %s", len(posteriors_by_path), output_cache_path
    )
    return output_cache_path
