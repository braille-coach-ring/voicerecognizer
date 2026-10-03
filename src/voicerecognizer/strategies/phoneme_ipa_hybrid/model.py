"""Model architecture for Strategy: Wav2Vec2 Consonant/Vowel Multi-Task + XLS-R IPA KD Hybrid."""

from __future__ import annotations

from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from transformers import Wav2Vec2ForSequenceClassification

from voicerecognizer.strategies.ipa_kd.teacher import IPA_VOCAB_SIZE
from voicerecognizer.strategies.phoneme_multi.model import MultiTaskUncertaintyLoss
from voicerecognizer.strategies.phoneme_multi.phoneme_mapping import (
    CONSONANTS,
    VOWELS,
)


class MultiTaskIPAHybridOutput(NamedTuple):
    loss: torch.Tensor | None
    logits: torch.Tensor
    char_logits: torch.Tensor
    cons_logits: torch.Tensor
    vowel_logits: torch.Tensor
    ipa_logits: torch.Tensor


class Wav2Vec2ForPhonemeIPAHybridClassification(Wav2Vec2ForSequenceClassification):
    r"""Wav2Vec2 architecture equipped with auxiliary consonant, vowel, and IPA distillation heads.

    Combines:
    1. 105-class character prediction (primary objective).
    2. 30-class consonant and 7-class vowel multi-task prediction (Kendall uncertainty loss).
    3. 392-class XLS-R IPA soft-target distribution alignment (temperature-scaled KL divergence).
    """

    def __init__(
        self,
        config: Any,
        ipa_vocab_size: int = IPA_VOCAB_SIZE,
        temperature: float = 2.0,
    ) -> None:
        super().__init__(config)
        self.num_consonants = getattr(config, "num_consonants", len(CONSONANTS))
        self.num_vowels = getattr(config, "num_vowels", len(VOWELS))
        self.ipa_vocab_size = ipa_vocab_size
        self.temperature = temperature

        proj_size = getattr(config, "classifier_proj_size", config.hidden_size)
        self.consonant_classifier = nn.Linear(proj_size, self.num_consonants)
        self.vowel_classifier = nn.Linear(proj_size, self.num_vowels)
        self.ipa_head = nn.Linear(proj_size, self.ipa_vocab_size)

        self.post_init()

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
        teacher_ipa_posteriors: torch.Tensor | None = None,
        alpha_kd: float = 0.20,
        uncertainty_loss_fn: MultiTaskUncertaintyLoss | None = None,
        return_phonemes: bool = False,
        **kwargs,
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
        else:
            padding_mask = self._get_feature_vector_attention_mask(
                hidden_states.shape[1], attention_mask
            )
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        # Task heads
        char_logits = self.classifier(pooled_output)
        cons_logits = self.consonant_classifier(pooled_output)
        vowel_logits = self.vowel_classifier(pooled_output)
        ipa_logits = self.ipa_head(pooled_output)

        loss: torch.Tensor | None = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_char = loss_fct(char_logits, labels)

            # Consonant and vowel ground truths
            if cons_labels is None and c_table is not None:
                if not isinstance(c_table, torch.Tensor):
                    c_table = torch.tensor(c_table, dtype=torch.long, device=labels.device)
                cons_labels = c_table[labels]
            if vowel_labels is None and v_table is not None:
                if not isinstance(v_table, torch.Tensor):
                    v_table = torch.tensor(v_table, dtype=torch.long, device=labels.device)
                vowel_labels = v_table[labels]

            if cons_labels is not None and vowel_labels is not None:
                loss_cons = loss_fct(cons_logits, cons_labels)
                loss_vow = loss_fct(vowel_logits, vowel_labels)
                if uncertainty_loss_fn is not None:
                    loss_multi = uncertainty_loss_fn(loss_char, loss_cons, loss_vow)
                else:
                    loss_multi = loss_char + 0.5 * loss_cons + 0.5 * loss_vow
            else:
                loss_multi = loss_char

            # Distillation loss from XLS-R IPA teacher
            if teacher_ipa_posteriors is not None and alpha_kd > 0.0:
                student_log_probs = F.log_softmax(ipa_logits / self.temperature, dim=-1)
                target_probs = teacher_ipa_posteriors.to(student_log_probs.device).float()
                loss_kd = F.kl_div(student_log_probs, target_probs, reduction="batchmean")
                loss_kd = loss_kd * (self.temperature**2)
                loss = loss_multi + alpha_kd * loss_kd
            else:
                loss = loss_multi

        if not return_phonemes:
            return MultiTaskIPAHybridOutput(
                loss=loss,
                logits=char_logits,
                char_logits=char_logits,
                cons_logits=cons_logits,
                vowel_logits=vowel_logits,
                ipa_logits=ipa_logits,
            )

        return char_logits, cons_logits, vowel_logits, ipa_logits

    def extract_inference_model(self) -> Wav2Vec2ForSequenceClassification:
        """Extracts standard Wav2Vec2ForSequenceClassification instance without auxiliary heads."""
        standard_model = Wav2Vec2ForSequenceClassification(self.config)
        clean_state = {
            k: v
            for k, v in self.state_dict().items()
            if not k.startswith("consonant_classifier")
            and not k.startswith("vowel_classifier")
            and not k.startswith("ipa_head")
        }
        standard_model.load_state_dict(clean_state, strict=True)
        return standard_model
