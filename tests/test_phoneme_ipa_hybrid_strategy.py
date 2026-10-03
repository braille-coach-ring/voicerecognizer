"""Unit tests for Strategy: Wav2Vec2 Consonant/Vowel Multi-Task + XLS-R IPA KD Hybrid."""

import pytest
import torch
from transformers import AutoConfig, Wav2Vec2ForSequenceClassification

from voicerecognizer.strategies.ipa_kd.teacher import IPA_VOCAB_SIZE
from voicerecognizer.strategies.phoneme_ipa_hybrid.model import (
    MultiTaskIPAHybridOutput,
    MultiTaskUncertaintyLoss,
    Wav2Vec2ForPhonemeIPAHybridClassification,
)
from voicerecognizer.strategies.phoneme_multi.phoneme_mapping import (
    CONSONANTS,
    VOWELS,
    build_label_phoneme_tables,
)


@pytest.fixture
def dummy_labels() -> list[str]:
    return ["a", "ka", "sa", "ta", "na", "ha", "ma", "ya", "ra", "wa"]


@pytest.fixture
def dummy_config(dummy_labels: list[str]) -> AutoConfig:
    config = AutoConfig.from_pretrained(
        "facebook/wav2vec2-base",
        num_labels=len(dummy_labels),
        finetuning_task="audio-classification",
    )
    config.classifier_proj_size = 64
    config.num_consonants = len(CONSONANTS)
    config.num_vowels = len(VOWELS)
    return config


def test_hybrid_forward_and_loss(dummy_config: AutoConfig, dummy_labels: list[str]) -> None:
    model = Wav2Vec2ForPhonemeIPAHybridClassification(
        config=dummy_config,
        ipa_vocab_size=IPA_VOCAB_SIZE,
        temperature=2.0,
    )
    model.eval()

    c_indices, v_indices = build_label_phoneme_tables(dummy_labels)
    c_table = torch.tensor(c_indices, dtype=torch.long)
    v_table = torch.tensor(v_indices, dtype=torch.long)
    uncertainty_loss_fn = MultiTaskUncertaintyLoss(init_log_var=0.0)

    batch_size = 2
    seq_len = 9600  # 0.6s at 16kHz
    input_values = torch.randn(batch_size, seq_len)
    labels = torch.tensor([0, 1], dtype=torch.long)
    teacher_ipa = torch.softmax(torch.randn(batch_size, IPA_VOCAB_SIZE), dim=-1)

    output = model(
        input_values=input_values,
        labels=labels,
        c_table=c_table,
        v_table=v_table,
        teacher_ipa_posteriors=teacher_ipa,
        alpha_kd=0.20,
        uncertainty_loss_fn=uncertainty_loss_fn,
    )

    assert isinstance(output, MultiTaskIPAHybridOutput)
    assert output.loss is not None
    assert output.loss.ndim == 0
    assert not torch.isnan(output.loss)
    assert output.logits.shape == (batch_size, len(dummy_labels))
    assert output.cons_logits.shape == (batch_size, len(CONSONANTS))
    assert output.vowel_logits.shape == (batch_size, len(VOWELS))
    assert output.ipa_logits.shape == (batch_size, IPA_VOCAB_SIZE)


def test_hybrid_extract_inference_model(dummy_config: AutoConfig) -> None:
    hybrid_model = Wav2Vec2ForPhonemeIPAHybridClassification(
        config=dummy_config,
        ipa_vocab_size=IPA_VOCAB_SIZE,
    )
    inference_model = hybrid_model.extract_inference_model()
    inference_model.eval()

    assert isinstance(inference_model, Wav2Vec2ForSequenceClassification)
    assert not hasattr(inference_model, "consonant_classifier")
    assert not hasattr(inference_model, "vowel_classifier")
    assert not hasattr(inference_model, "ipa_head")

    # Forward pass through extracted inference model
    input_values = torch.randn(2, 9600)
    with torch.no_grad():
        out = inference_model(input_values)
    assert out.logits.shape == (2, dummy_config.num_labels)
