"""Wav2Vec2 model architecture with ArcFace Angular Margin Head."""

from __future__ import annotations

from typing import Any, NamedTuple

import torch
from torch.nn.functional import cross_entropy, normalize
from transformers import Wav2Vec2ForSequenceClassification

from voicerecognizer.strategies.arcface.loss import ArcMarginProduct


class ArcFaceOutput(NamedTuple):
    loss: torch.Tensor | None
    logits: torch.Tensor
    penalized_logits: torch.Tensor


class Wav2Vec2ForArcFaceClassification(Wav2Vec2ForSequenceClassification):
    r"""Wav2Vec2 classification model trained with Additive Angular Margin Loss (ArcFace)."""

    def __init__(
        self,
        config: Any,
        s: float = 24.0,
        m: float = 0.30,
        easy_margin: bool = False,
    ) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.s = s
        self.m = m

        proj_size = getattr(config, "classifier_proj_size", config.hidden_size)

        # ArcFace margin head
        self.arcface_head = ArcMarginProduct(
            in_features=proj_size,
            out_features=self.num_labels,
            s=s,
            m=m,
            easy_margin=easy_margin,
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
        **kwargs,
    ) -> ArcFaceOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward through Wav2Vec2 base encoder
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.projector(hidden_states)

        # Time pooling
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(
                hidden_states.shape[1], attention_mask
            )
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        # ArcFace loss computation
        loss: torch.Tensor | None = None
        if labels is not None:
            penalized_logits, raw_logits = self.arcface_head(pooled_output, labels)
            loss = cross_entropy(penalized_logits, labels)
        else:
            penalized_logits, raw_logits = self.arcface_head(pooled_output, labels=None)

        return ArcFaceOutput(
            loss=loss,
            logits=raw_logits,
            penalized_logits=penalized_logits,
        )

    def extract_inference_model(self) -> Wav2Vec2ForSequenceClassification:
        """Extracts standard Wav2Vec2ForSequenceClassification instance without auxiliary overhead."""
        standard_model = Wav2Vec2ForSequenceClassification(self.config)
        standard_model.wav2vec2.load_state_dict(self.wav2vec2.state_dict())
        standard_model.projector.load_state_dict(self.projector.state_dict())

        # Set normalized classifier weights scaled by s
        with torch.no_grad():
            norm_weight = self.s * normalize(self.arcface_head.weight, p=2, dim=1)
            standard_model.classifier.weight.copy_(norm_weight)

            if standard_model.classifier.bias is not None:
                standard_model.classifier.bias.zero_()

        return standard_model
