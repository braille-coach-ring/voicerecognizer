"""Student model architecture with auxiliary Whisper feature distillation head."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from transformers import Wav2Vec2ForSequenceClassification


class WhisperKDOutput(NamedTuple):
    loss: torch.Tensor | None
    logits: torch.Tensor
    char_logits: torch.Tensor
    student_emb: torch.Tensor
    loss_ce: torch.Tensor | None
    loss_kd: torch.Tensor | None


class Wav2Vec2ForWhisperKDClassification(Wav2Vec2ForSequenceClassification):
    """Wav2Vec2 Sequence Classification Model with auxiliary Whisper Feature Distillation Head.

    The primary head classifies 105 hiragana syllable classes.
    The auxiliary projection head maps the pooled representation into Whisper's embedding space,
    aligning student acoustic representations with Whisper's robust features.
    """

    def __init__(self, config, whisper_dim: int = 1280):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.whisper_dim = whisper_dim

        proj_size = getattr(config, "classifier_proj_size", config.hidden_size)

        # Auxiliary head to project Wav2Vec2 pooled features into Whisper embedding space
        self.whisper_projector = nn.Sequential(
            nn.Linear(proj_size, whisper_dim),
            nn.GELU(),
            nn.Linear(whisper_dim, whisper_dim),
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
        teacher_whisper_embeddings: torch.Tensor | None = None,
        alpha_kd: float = 0.5,
        **kwargs,
    ) -> WhisperKDOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward through base Wav2Vec2 encoder
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.projector(hidden_states)

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

        # Auxiliary projection to Whisper embedding space
        student_emb = F.normalize(self.whisper_projector(pooled_output), p=2, dim=-1)

        loss = None
        loss_ce = None
        loss_kd = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_ce = loss_fct(char_logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss_ce

            if teacher_whisper_embeddings is not None:
                # Cosine distance loss: 1.0 - cos_sim
                target_emb = teacher_whisper_embeddings.to(student_emb.device).float()
                cos_sim = F.cosine_similarity(student_emb, target_emb, dim=-1)
                loss_kd = (1.0 - cos_sim).mean()

                loss = (1.0 - alpha_kd) * loss_ce + alpha_kd * loss_kd

        return WhisperKDOutput(
            loss=loss,
            logits=char_logits,
            char_logits=char_logits,
            student_emb=student_emb,
            loss_ce=loss_ce,
            loss_kd=loss_kd,
        )

    def extract_inference_model(self) -> Wav2Vec2ForSequenceClassification:
        """Extract a standard Wav2Vec2ForSequenceClassification instance without auxiliary heads."""
        clean_model = Wav2Vec2ForSequenceClassification(self.config)
        clean_state = {
            k: v for k, v in self.state_dict().items() if not k.startswith("whisper_projector")
        }
        clean_model.load_state_dict(clean_state, strict=True)
        return clean_model
