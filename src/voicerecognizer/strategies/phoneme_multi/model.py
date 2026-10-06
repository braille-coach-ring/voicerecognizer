"""Model architecture for Strategy 06: Wav2Vec2 Consonant/Vowel Multi-Task Learning."""

from __future__ import annotations

from typing import Any, NamedTuple, cast, override

import torch
import torch.nn as nn
from transformers import Wav2Vec2ForSequenceClassification

from voicerecognizer.strategies.phoneme_multi.phoneme_mapping import (
    CONSONANTS,
    VOWELS,
)


class MultiTaskWav2Vec2Output(NamedTuple):
    loss: torch.Tensor | None
    logits: torch.Tensor
    char_logits: torch.Tensor
    cons_logits: torch.Tensor
    vowel_logits: torch.Tensor


class MultiTaskUncertaintyLoss(nn.Module):
    r"""Homoscedastic uncertainty weighting for multi-task loss (Kendall et al., CVPR 2018).

    Learns task-dependent noise variances $s_i = \log \sigma_i^2$ dynamically:
    $$
    \mathcal{L}_{\text{total}} = \frac{1}{2}\sum_{i} \left( e^{-s_i} \mathcal{L}_i + s_i \right)
    $$
    """

    def __init__(self, init_log_var: float = 0.0) -> None:
        super().__init__()
        # Indices: 0: char, 1: consonant, 2: vowel
        self.log_vars = nn.Parameter(torch.full((3,), init_log_var, dtype=torch.float32))

    @override
    def forward(
        self,
        loss_char: torch.Tensor,
        loss_cons: torch.Tensor,
        loss_vowel: torch.Tensor,
    ) -> torch.Tensor:
        precision_c = torch.exp(-self.log_vars[0])
        precision_cons = torch.exp(-self.log_vars[1])
        precision_vow = torch.exp(-self.log_vars[2])

        loss = (
            0.5 * precision_c * loss_char
            + 0.5 * precision_cons * loss_cons
            + 0.5 * precision_vow * loss_vowel
            + 0.5 * (self.log_vars[0] + self.log_vars[1] + self.log_vars[2])
        )
        return loss

    def get_effective_weights(self) -> dict[str, float]:
        """Returns current effective weights 0.5 * exp(-s) for observability."""
        with torch.no_grad():
            precisions = torch.exp(-self.log_vars) * 0.5
            return {
                "weight_char": float(precisions[0].item()),
                "weight_cons": float(precisions[1].item()),
                "weight_vow": float(precisions[2].item()),
                "sigma_char": float(torch.exp(0.5 * self.log_vars[0]).item()),
                "sigma_cons": float(torch.exp(0.5 * self.log_vars[1]).item()),
                "sigma_vow": float(torch.exp(0.5 * self.log_vars[2]).item()),
            }


class Wav2Vec2ForPhonemeMultiTaskClassification(Wav2Vec2ForSequenceClassification):
    r"""Wav2Vec2 architecture equipped with auxiliary consonant and vowel classification heads."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.num_consonants = getattr(config, "num_consonants", len(CONSONANTS))
        self.num_vowels = getattr(config, "num_vowels", len(VOWELS))

        proj_size = config.classifier_proj_size
        if not isinstance(proj_size, int):
            raise TypeError("classifier_proj_size must be an integer")
        self.consonant_classifier = nn.Linear(proj_size, self.num_consonants)
        self.vowel_classifier = nn.Linear(proj_size, self.num_vowels)

        self.post_init()

    @override
    def forward(
        self,
        input_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        labels: torch.Tensor | None = None,
        cons_labels: torch.Tensor | None = None,
        vowel_labels: torch.Tensor | None = None,
        c_table: torch.Tensor | None = None,
        v_table: torch.Tensor | None = None,
        lambda_cons: float = 0.5,
        lambda_vowel: float = 0.5,
        uncertainty_loss_fn: MultiTaskUncertaintyLoss | None = None,
        return_phonemes: bool = False,
        **kwargs: Any,
    ) -> Any:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Feature extractor and transformer backbone
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden_states = outputs[0]
        hidden_states = self.projector(hidden_states)

        # Time pooling
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
            frame_cons_logits = self.consonant_classifier(hidden_states)
            cons_logits = frame_cons_logits.max(dim=1)[0]
        else:
            padding_mask = self._get_feature_vector_attention_mask(
                hidden_states.shape[1], cast(torch.LongTensor, attention_mask)
            )
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)
            frame_cons_logits = self.consonant_classifier(hidden_states)
            masked_cons_logits = frame_cons_logits.masked_fill(~padding_mask.unsqueeze(-1), -1e9)
            cons_logits = masked_cons_logits.max(dim=1)[0]

        # Task heads
        char_logits = self.classifier(pooled_output)
        vowel_logits = self.vowel_classifier(pooled_output)

        loss: torch.Tensor | None = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_char = loss_fct(char_logits, labels)

            # Determine consonant and vowel ground truths from lookup tables
            if cons_labels is None and c_table is not None:
                cons_labels = c_table[labels]
            if vowel_labels is None and v_table is not None:
                vowel_labels = v_table[labels]

            if cons_labels is not None and vowel_labels is not None:
                loss_cons = loss_fct(cons_logits, cons_labels)
                loss_vow = loss_fct(vowel_logits, vowel_labels)
                if uncertainty_loss_fn is not None:
                    loss = uncertainty_loss_fn(loss_char, loss_cons, loss_vow)
                else:
                    loss = loss_char + lambda_cons * loss_cons + lambda_vowel * loss_vow
            else:
                loss = loss_char

        if not return_phonemes:
            return MultiTaskWav2Vec2Output(
                loss=loss,
                logits=char_logits,
                char_logits=char_logits,
                cons_logits=cons_logits,
                vowel_logits=vowel_logits,
            )

        return char_logits, cons_logits, vowel_logits

    def extract_inference_model(self) -> Wav2Vec2ForSequenceClassification:
        """Extracts standard Wav2Vec2ForSequenceClassification instance without auxiliary heads."""
        standard_model = Wav2Vec2ForSequenceClassification(self.config)
        clean_state = {
            k: v
            for k, v in self.state_dict().items()
            if not k.startswith("consonant_classifier") and not k.startswith("vowel_classifier")
        }
        standard_model.load_state_dict(clean_state, strict=True)
        return standard_model
