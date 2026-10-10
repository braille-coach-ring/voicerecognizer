from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import script.collect_with_label_sound as collector


def test_default_mode_records_while_space_is_held() -> None:
    args = collector.build_parser().parse_args(["rinry", "--no-upload"])

    assert args.mode == "hold"


def test_collect_hold_records_one_take_only_during_space_press(monkeypatch, tmp_path) -> None:
    chunk_samples = int(collector.SAMPLE_RATE * 0.05)
    saved_audio: list[np.ndarray] = []

    class FakeInputStream:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, samples):
            assert samples == chunk_samples
            return np.full((samples, 1), 0.2, dtype=np.float32), False

    key_states = iter([False, True, True, False, False])

    def fake_space_pressed() -> bool:
        return next(key_states, False)

    def fake_save_audio(folder, number, audio):
        saved_audio.append(audio.copy())
        return folder / f"{number:03d}.wav"

    monkeypatch.setattr(collector, "is_space_pressed", fake_space_pressed)
    monkeypatch.setattr(collector.sd, "InputStream", FakeInputStream)
    monkeypatch.setattr(collector, "save_audio", fake_save_audio)
    monkeypatch.setattr(collector.time, "sleep", lambda _seconds: None)

    args = SimpleNamespace(
        chunk_seconds=0.05,
        min_utterance_seconds=0.01,
        max_utterance_seconds=1.0,
        silence_threshold=0.01,
        rms_threshold=0.01,
        interval=0.0,
    )

    next_number = collector.collect_hold("a", tmp_path, 1, 7, args)

    assert next_number == 8
    assert len(saved_audio) == 1
    assert saved_audio[0].shape == (chunk_samples,)
