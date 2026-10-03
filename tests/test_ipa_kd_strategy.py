"""Unit test for Strategy 01: Wav2Vec2 + XLS-R IPA Knowledge Distillation."""

import torch
from transformers import AutoConfig, Wav2Vec2ForSequenceClassification

from voicerecognizer.strategies.ipa_kd.model import Wav2Vec2ForIPAKDClassification
from voicerecognizer.strategies.ipa_kd.teacher import IPA_VOCAB_SIZE


def test_wav2vec2_ipa_kd_forward_and_loss():
    # Use small dummy config
    config = AutoConfig.from_pretrained("facebook/wav2vec2-base")
    config.num_labels = 105
    config.classifier_proj_size = 256
    config.num_hidden_layers = 2  # speed up test

    model = Wav2Vec2ForIPAKDClassification(
        config=config,
        ipa_vocab_size=IPA_VOCAB_SIZE,
        temperature=2.0,
    )
    model.eval()

    batch_size = 2
    samples = 16000
    dummy_wav = torch.randn(batch_size, samples)
    dummy_labels = torch.tensor([0, 1], dtype=torch.long)
    dummy_teacher_posteriors = torch.softmax(torch.randn(batch_size, IPA_VOCAB_SIZE), dim=-1)

    output = model(
        input_values=dummy_wav,
        labels=dummy_labels,
        teacher_ipa_posteriors=dummy_teacher_posteriors,
        alpha_kd=0.5,
    )

    assert output.loss is not None
    assert output.logits.shape == (batch_size, 105)
    assert output.ipa_logits.shape == (batch_size, IPA_VOCAB_SIZE)
    assert output.loss.item() > 0.0


def test_extract_inference_model():
    config = AutoConfig.from_pretrained("facebook/wav2vec2-base")
    config.num_labels = 105
    config.classifier_proj_size = 256
    config.num_hidden_layers = 1

    model = Wav2Vec2ForIPAKDClassification(config=config, ipa_vocab_size=IPA_VOCAB_SIZE)
    clean_model = model.extract_inference_model()

    assert isinstance(clean_model, Wav2Vec2ForSequenceClassification)
    assert not hasattr(clean_model, "ipa_head")
    assert clean_model.classifier.out_features == 105
