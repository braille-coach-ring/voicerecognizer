"""Unit tests for Strategy 06: Wav2Vec2 Consonant/Vowel Multi-Task Learning."""

import torch
from transformers import Wav2Vec2Config, Wav2Vec2ForSequenceClassification

from voicerecognizer.strategies.phoneme_multi.model import (
    MultiTaskUncertaintyLoss,
    Wav2Vec2ForPhonemeMultiTaskClassification,
)
from voicerecognizer.strategies.phoneme_multi.phoneme_mapping import (
    CONSONANTS,
    VOWELS,
    build_label_phoneme_tables,
    decompose_label,
)


def test_phoneme_mapping_decomposition():
    ka = decompose_label("ka")
    assert ka.consonant == "k"
    assert ka.vowel == "a"

    shi = decompose_label("shi")
    assert shi.consonant == "sh"
    assert shi.vowel == "i"

    a_sound = decompose_label("a")
    assert a_sound.consonant == "none"
    assert a_sound.vowel == "a"

    labels = ["a", "ka", "shi"]
    c_indices, v_indices = build_label_phoneme_tables(labels)
    assert len(c_indices) == 3
    assert len(v_indices) == 3
    assert c_indices[1] == CONSONANTS.index("k")
    assert v_indices[1] == VOWELS.index("a")


def test_wav2vec2_phoneme_multi_forward_and_loss():
    config = Wav2Vec2Config(
        vocab_size=105,
        num_labels=105,
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

    batch_size = 2
    seq_len = 1600
    input_values = torch.randn(batch_size, seq_len)
    labels = torch.tensor([1, 2], dtype=torch.long)
    c_table = torch.tensor([0, 1, 3], dtype=torch.long)
    v_table = torch.tensor([0, 0, 1], dtype=torch.long)

    output = model(
        input_values=input_values,
        labels=labels,
        c_table=c_table,
        v_table=v_table,
        uncertainty_loss_fn=uncertainty_fn,
    )

    assert output.loss is not None
    assert output.char_logits.shape == (batch_size, 105)
    assert output.cons_logits.shape == (batch_size, len(CONSONANTS))
    assert output.vowel_logits.shape == (batch_size, len(VOWELS))
    assert output.loss.item() > 0.0

    output.loss.backward()
    assert model.classifier.weight.grad is not None
    assert model.consonant_classifier.weight.grad is not None
    assert model.vowel_classifier.weight.grad is not None
    assert uncertainty_fn.log_vars.grad is not None


def test_extract_inference_model():
    config = Wav2Vec2Config(
        vocab_size=105,
        num_labels=105,
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
    inference_model = model.extract_inference_model()

    assert isinstance(inference_model, Wav2Vec2ForSequenceClassification)
    assert not hasattr(inference_model, "consonant_classifier")
    assert not hasattr(inference_model, "vowel_classifier")

    input_values = torch.randn(1, 1600)
    out = inference_model(input_values)
    assert out.logits.shape == (1, 105)
