"""Unit tests for Strategy 04: Wav2Vec2 + ArcFace Angular Margin Loss."""

import torch
from transformers import Wav2Vec2Config, Wav2Vec2ForSequenceClassification

from voicerecognizer.strategies.arcface.loss import ArcMarginProduct
from voicerecognizer.strategies.arcface.model import Wav2Vec2ForArcFaceClassification


def test_arc_margin_product_forward():
    in_features = 64
    out_features = 10
    s = 24.0
    m = 0.30

    layer = ArcMarginProduct(in_features=in_features, out_features=out_features, s=s, m=m)

    batch_size = 4
    x = torch.randn(batch_size, in_features)
    labels = torch.tensor([0, 2, 5, 9], dtype=torch.long)

    penalized_logits, raw_logits = layer(x, labels)

    assert penalized_logits.shape == (batch_size, out_features)
    assert raw_logits.shape == (batch_size, out_features)

    # For the target class, penalized logit should be <= raw logit (angular penalty)
    for b in range(batch_size):
        target = labels[b].item()
        assert penalized_logits[b, target].item() <= raw_logits[b, target].item() + 1e-4

    # Gradient check
    loss = penalized_logits.sum()
    loss.backward()
    assert layer.weight.grad is not None


def test_wav2vec2_arcface_forward_and_loss():
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

    model = Wav2Vec2ForArcFaceClassification(config, s=24.0, m=0.3)

    batch_size = 2
    seq_len = 1600
    input_values = torch.randn(batch_size, seq_len)
    labels = torch.tensor([12, 85], dtype=torch.long)

    output = model(input_values=input_values, labels=labels)

    assert output.loss is not None
    assert output.logits.shape == (batch_size, 105)
    assert output.penalized_logits.shape == (batch_size, 105)
    assert output.loss.item() > 0.0

    output.loss.backward()
    assert model.arcface_head.weight.grad is not None
    assert model.projector.weight.grad is not None


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

    model = Wav2Vec2ForArcFaceClassification(config, s=24.0, m=0.3)
    inference_model = model.extract_inference_model()

    assert isinstance(inference_model, Wav2Vec2ForSequenceClassification)
    assert not hasattr(inference_model, "arcface_head")

    input_values = torch.randn(1, 1600)
    out = inference_model(input_values)
    assert out.logits.shape == (1, 105)
