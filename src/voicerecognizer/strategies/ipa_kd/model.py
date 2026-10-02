"""Student model architecture with auxiliary IPA phoneme distillation head."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from transformers import Wav2Vec2ForSequenceClassification

from voicerecognizer.strategies.ipa_kd.teacher import IPA_VOCAB_SIZE


class IPAKDOutput(NamedTuple):
    loss: torch.Tensor | None
    logits: torch.Tensor
    char_logits: torch.Tensor
    ipa_logits: torch.Tensor


class Wav2Vec2ForIPAKDClassification(Wav2Vec2ForSequenceClassification):
    """Wav2Vec2 Sequence Classification Model with an auxiliary IPA Phoneme Distillation Head.

    The primary head classifies 105 hiragana syllable classes.
    The auxiliary head predicts the dense 392-dim IPA phoneme posterior distribution
    distilled from the teacher model (XLS-R 53 espeak).
    """

    def __init__(self, config, ipa_vocab_size: int = IPA_VOCAB_SIZE, temperature: float = 2.0):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.ipa_vocab_size = ipa_vocab_size
        self.temperature = temperature

        # Identify hidden dimension feeding into classifier
        # Wav2Vec2ForSequenceClassification has project (Linear) and classifier (Linear)
        # config.classifier_proj_size or config.hidden_size
        proj_size = getattr(config, "classifier_proj_size", config.hidden_size)

        # Auxiliary IPA distillation head
        self.ipa_head = nn.Sequential(
            nn.Dropout(config.final_dropout if hasattr(config, "final_dropout") else 0.1),
            nn.Linear(proj_size, ipa_vocab_size),
        )

        self.post_init()

    def forward(
        self,
        input_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        labels: torch.Tensor | None = None,
        teacher_ipa_posteriors: torch.Tensor | None = None,
        alpha_kd: float = 0.5,
        **kwargs,
    ) -> IPAKDOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward through Wav2Vec2 base encoder
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]  # (batch, time, hidden_size)

        # Project features
        hidden_states = self.projector(hidden_states)

        # Pool over time sequence
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(
                hidden_states.shape[1], attention_mask
            )
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        # Primary character classification (105 classes)
        char_logits = self.classifier(pooled_output)

        # Auxiliary IPA phoneme prediction
        ipa_logits = self.ipa_head(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_char = loss_fct(char_logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss_char

            if teacher_ipa_posteriors is not None:
                # Soft-target distillation loss via KL-divergence
                # student log-probabilities with temperature scaling
                student_log_probs = F.log_softmax(ipa_logits / self.temperature, dim=-1)
                # target is already a soft probability distribution from teacher
                target_probs = teacher_ipa_posteriors.to(student_log_probs.device).float()

                # KL divergence: KL(teacher || student)
                loss_kd = F.kl_div(student_log_probs, target_probs, reduction="batchmean")
                loss_kd = loss_kd * (self.temperature**2)

                loss = (1.0 - alpha_kd) * loss_char + alpha_kd * loss_kd

        return IPAKDOutput(
            loss=loss,
            logits=char_logits,
            char_logits=char_logits,
            ipa_logits=ipa_logits,
        )

    def extract_inference_model(self) -> Wav2Vec2ForSequenceClassification:
        """Extract a clean, standard Wav2Vec2ForSequenceClassification instance without auxiliary heads for ONNX export."""
        clean_model = Wav2Vec2ForSequenceClassification(self.config)
        clean_state = {k: v for k, v in self.state_dict().items() if not k.startswith("ipa_head")}
        clean_model.load_state_dict(clean_state, strict=True)
        return clean_model
