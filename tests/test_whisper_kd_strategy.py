"""Unit tests for Strategy 02: Wav2Vec2 + Whisper Hidden Feature Distillation."""

import torch
from transformers import Wav2Vec2Config, Wav2Vec2ForSequenceClassification

from voicerecognizer.strategies.whisper_kd.model import Wav2Vec2ForWhisperKDClassification


def test_wav2vec2_whisper_kd_forward_and_loss():
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

    whisper_dim = 128
    model = Wav2Vec2ForWhisperKDClassification(config, whisper_dim=whisper_dim)

    batch_size = 2
    seq_len = 1600  # 0.1s at 16kHz
    input_values = torch.randn(batch_size, seq_len)
    labels = torch.tensor([5, 42], dtype=torch.long)
    teacher_embs = torch.randn(batch_size, whisper_dim)

    output = model(
        input_values=input_values,
        labels=labels,
        teacher_whisper_embeddings=teacher_embs,
        alpha_kd=0.5,
    )

    assert output.loss is not None
    assert output.logits is not None
    assert output.logits.shape == (batch_size, 105)
    assert output.student_emb.shape == (batch_size, whisper_dim)
    assert output.loss.item() > 0.0

    output.loss.backward()
    assert model.classifier.weight.grad is not None
    assert model.whisper_projector[0].weight.grad is not None


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

    model = Wav2Vec2ForWhisperKDClassification(config, whisper_dim=128)
    inference_model = model.extract_inference_model()

    assert isinstance(inference_model, Wav2Vec2ForSequenceClassification)
    assert not hasattr(inference_model, "whisper_projector")

    input_values = torch.randn(1, 1600)
    out = inference_model(input_values)
    assert out.logits.shape == (1, 105)
