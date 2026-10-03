"""Unit tests for Strategy 07: Wav2Vec2 + RIR Simulation & Mic Distortion Augmentation."""

import numpy as np
import torch
from transformers import Wav2Vec2Config

from voicerecognizer.preprocessing.audio_augmentor import AudioAugmentor
from voicerecognizer.strategies.phoneme_multi.model import (
    MultiTaskUncertaintyLoss,
    Wav2Vec2ForPhonemeMultiTaskClassification,
)
from voicerecognizer.strategies.phoneme_multi.phoneme_mapping import (
    CONSONANTS,
    VOWELS,
    build_label_phoneme_tables,
)


def test_audio_augmentor_mic_saturation():
    augmentor = AudioAugmentor(p=1.0, seed=42)
    waveform = np.sin(np.linspace(0, 100 * np.pi, 16000, dtype=np.float32))

    saturated = augmentor.apply_mic_saturation(waveform)
    assert saturated.shape == waveform.shape
    assert saturated.dtype == np.float32
    assert not np.allclose(saturated, waveform)
    # Check bounded output
    assert np.max(np.abs(saturated)) <= 1.05


def test_audio_augmentor_full_pipeline():
    augmentor = AudioAugmentor(p=1.0, seed=42)
    waveform = np.sin(np.linspace(0, 100 * np.pi, 9600, dtype=np.float32)) * 0.5

    augmented = augmentor.augment(waveform)
    assert augmented.shape == waveform.shape
    assert augmented.dtype == np.float32
    assert not np.isnan(augmented).any()
    assert not np.isinf(augmented).any()


def test_rir_simulation_phoneme_multi_step():
    config = Wav2Vec2Config(
        vocab_size=10,
        num_labels=10,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        conv_dim=(32, 32),
        conv_stride=(5, 2),
        conv_kernel=(10, 3),
        conv_bias=False,
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
        classifier_proj_size=64,
        final_dropout=0.0,
    )
    config.num_consonants = len(CONSONANTS)
    config.num_vowels = len(VOWELS)

    model = Wav2Vec2ForPhonemeMultiTaskClassification(config)
    uncertainty_fn = MultiTaskUncertaintyLoss()

    labels = ["a", "ka", "sa", "ta", "na", "ha", "ma", "ya", "ra", "wa"]
    c_indices, v_indices = build_label_phoneme_tables(labels)
    c_table = torch.tensor(c_indices, dtype=torch.long)
    v_table = torch.tensor(v_indices, dtype=torch.long)

    batch_size = 2
    seq_len = 9600
    input_values = torch.randn(batch_size, seq_len)
    batch_labels = torch.tensor([0, 1], dtype=torch.long)

    output = model(
        input_values=input_values,
        labels=batch_labels,
        c_table=c_table,
        v_table=v_table,
        uncertainty_loss_fn=uncertainty_fn,
    )

    assert output.loss is not None
    assert output.loss.ndim == 0
    assert not torch.isnan(output.loss)
    assert output.logits.shape == (batch_size, 10)
    assert output.cons_logits.shape == (batch_size, len(CONSONANTS))
    assert output.vowel_logits.shape == (batch_size, len(VOWELS))

    # Inference model extraction
    inf_model = model.extract_inference_model()
    inf_model.eval()
    with torch.no_grad():
        inf_out = inf_model(input_values)
    assert inf_out.logits.shape == (batch_size, 10)
