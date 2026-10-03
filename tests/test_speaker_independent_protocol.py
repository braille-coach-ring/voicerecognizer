import csv
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from voicerecognizer.evaluation.evaluator import score_predictions
from voicerecognizer.models.wav2vec2.train import training_loss
from voicerecognizer.utils.split_helper import fixed_manifest_indices


def test_homophone_scores_keep_wrong_predictions_and_missing_classes():
    report = score_predictions(
        ["di", "du", "wo", "a"],
        ["ji", "zu", "o", "ka"],
        ["female"] * 3 + ["male"],
        ["di", "ji", "du", "zu", "wo", "o", "a", "ka"],
    )
    assert report["raw"]["overall"]["accuracy"] == 0
    assert report["normalized"]["overall"]["accuracy"] == 0.75
    assert report["normalized"]["overall"]["macro_f1"] == 0.75
    assert report["normalized"]["overall"]["inventory_macro_f1"] == 0.6
    assert report["normalized"]["speakers"]["female"]["accuracy"] == 1
    assert report["normalized"]["speaker_balanced_accuracy"] == 0.5
    with pytest.raises(ValueError):
        score_predictions(["a"], ["missing"], ["female"], ["a"])


def test_fixed_manifests_fail_on_omission_and_overlap():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        def write(name: str, rows: list[dict[str, str]]) -> Path:
            path = root / name
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["filepath", "source_filepath", "speaker"]
                )
                writer.writeheader()
                writer.writerows(rows)
            return path

        train = write("train.csv", [{"filepath": "raw_train.wav", "speaker": "train"}])
        val = write("val.csv", [{"filepath": "raw_val.wav", "speaker": "val"}])
        index = write(
            "index.csv",
            [
                {"filepath": "p0.wav", "source_filepath": "raw_val.wav"},
                {"filepath": "p1.wav", "source_filepath": "raw_train.wav"},
            ],
        )
        data = [(root / "p0.wav", 0), (root / "p1.wav", 0)]
        assert fixed_manifest_indices(data, index, train, val, root) == ([1], [0])
        with pytest.raises(ValueError):
            fixed_manifest_indices(data[:1], index, train, val, root)
        with pytest.raises(ValueError):
            fixed_manifest_indices(data, index, train, train, root)


def test_auxiliary_loss_reaches_auxiliary_heads_with_weighted_character_loss():
    outputs = SimpleNamespace(
        logits=torch.zeros(2, 2, requires_grad=True),
        cons_logits=torch.zeros(2, 2, requires_grad=True),
        vowel_logits=torch.zeros(2, 2, requires_grad=True),
    )
    model = SimpleNamespace(c_table=torch.tensor([0, 1]), v_table=torch.tensor([1, 0]))
    loss = training_loss(model, outputs, torch.tensor([0, 1]), torch.nn.CrossEntropyLoss())
    loss.backward()
    assert outputs.cons_logits.grad.abs().sum() > 0
    assert outputs.vowel_logits.grad.abs().sum() > 0


def test_preprocessing_preserves_explicit_tts_profile():
    import numpy as np
    import soundfile as sf

    from voicerecognizer.preprocessing.dataset_builder import DatasetBuilder

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        audio = root / "tts_augmented" / "nanami" / "a" / "001.wav"
        audio.parent.mkdir(parents=True)
        sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)
        index = root / "source" / "index.csv"
        index.parent.mkdir()
        with index.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("filepath", "label", "speaker"))
            writer.writeheader()
            writer.writerow(
                {"filepath": str(audio), "label": "a", "speaker": "tts_female_nanami_std"}
            )
        output = root / "processed"
        DatasetBuilder(labels=("a",)).preprocess_dataset(index.parent, output)
        with (output / "index.csv").open() as stream:
            assert next(csv.DictReader(stream))["speaker"] == "tts_female_nanami_std"


def test_reused_multitask_model_forward_and_standard_checkpoint(tmp_path: Path):
    from transformers import Wav2Vec2Config, Wav2Vec2ForSequenceClassification

    from voicerecognizer.strategies.phoneme_multi.model import (
        Wav2Vec2ForPhonemeMultiTaskClassification,
    )

    config = Wav2Vec2Config(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        conv_dim=(8, 8, 8),
        conv_stride=(2, 2, 2),
        conv_kernel=(3, 3, 3),
        classifier_proj_size=8,
        num_conv_pos_embedding_groups=2,
        num_conv_pos_embeddings=8,
        mask_time_prob=0.0,
    )
    config.num_labels = 2
    model = Wav2Vec2ForPhonemeMultiTaskClassification(config)
    model.register_buffer("c_table", torch.tensor([0, 1]), persistent=False)
    model.register_buffer("v_table", torch.tensor([1, 0]), persistent=False)
    labels = torch.tensor([0, 1])
    outputs = model(input_values=torch.randn(2, 128), labels=labels)
    loss = training_loss(model, outputs, labels, torch.nn.CrossEntropyLoss())
    loss.backward()
    assert model.consonant_classifier.weight.grad is not None
    assert model.vowel_classifier.weight.grad is not None
    assert model.consonant_classifier.weight.grad.abs().sum() > 0
    assert model.vowel_classifier.weight.grad.abs().sum() > 0
    model.save_pretrained(tmp_path)
    standard = Wav2Vec2ForSequenceClassification.from_pretrained(tmp_path)
    torch.testing.assert_close(standard.classifier.weight, model.classifier.weight)


def test_colab_output_redaction_and_windows_console():
    import io
    import subprocess
    from contextlib import redirect_stdout
    from unittest.mock import patch

    from script.run_speaker_independent_colab import colab

    stream = io.BytesIO()
    console = io.TextIOWrapper(stream, encoding="cp932")
    result = subprocess.CompletedProcess(
        [], 0, stdout="╭ ready colab-runtime-proxy-token=private-token&x=1\n", stderr=""
    )
    with (
        patch("script.run_speaker_independent_colab.subprocess.run", return_value=result),
        redirect_stdout(console),
    ):
        colab("test", ["status"])
    console.flush()
    output = stream.getvalue().decode("cp932")
    assert "ready" in output and "[redacted]" in output
    assert "private-token" not in output
    missing = subprocess.CompletedProcess(
        [], 0, stdout="[colab] Session 'test' not found.", stderr=""
    )
    with (
        patch("script.run_speaker_independent_colab.subprocess.run", return_value=missing),
        pytest.raises(subprocess.CalledProcessError),
    ):
        colab("test", ["status"])


def test_registered_multitask_recognizer_uses_explicit_checkpoint(tmp_path: Path):
    from unittest.mock import patch

    from voicerecognizer.core.factory.recognizer_factory import RecognizerFactory
    from voicerecognizer.strategies.registry import create_strategy_recognizer

    assert "wav2vec2_phoneme_multi" in RecognizerFactory.available_strategies()
    checkpoint = tmp_path / "measured-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").touch()
    with (
        patch("voicerecognizer.recognizers.wav2vec2_recognizer.Wav2Vec2Recognizer") as recognizer,
        patch("voicerecognizer.utils.model_uploader.download_strategy_weights_if_needed") as download,
    ):
        result = RecognizerFactory.create("wav2vec2_phoneme_multi", model_path=checkpoint)
        assert result is recognizer.return_value
        assert recognizer.call_args.kwargs["model_path"] == checkpoint
        download.assert_not_called()
    with pytest.raises(FileNotFoundError):
        create_strategy_recognizer(
            "wav2vec2_phoneme_multi", model_path=tmp_path / "missing", auto_download=False
        )
