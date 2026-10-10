from __future__ import annotations

import argparse
from pathlib import Path
from types import TracebackType

import numpy as np
import pytest

import script.collect_with_label_sound as collector


def test_default_mode_records_while_space_is_held() -> None:
    args = collector.build_parser().parse_args(["rinry", "--no-upload"])

    assert args.mode == "hold"


def test_collect_hold_records_one_take_only_during_space_press(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    chunk_samples = int(collector.SAMPLE_RATE * 0.05)
    saved_audio: list[np.ndarray] = []

    class FakeInputStream:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def __enter__(self) -> FakeInputStream:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: TracebackType | None,
        ) -> bool:
            return False

        def read(self, samples: int) -> tuple[np.ndarray, bool]:
            assert samples == chunk_samples
            return np.full((samples, 1), 0.2, dtype=np.float32), False

    key_states = iter([False, True, True, False, False])

    def fake_space_pressed() -> bool:
        return next(key_states, False)

    def fake_save_audio(folder: Path, number: int, audio: np.ndarray) -> Path:
        saved_audio.append(audio.copy())
        return folder / f"{number:03d}.wav"

    def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(collector, "is_space_pressed", fake_space_pressed)
    monkeypatch.setattr(collector.sd, "InputStream", FakeInputStream)
    monkeypatch.setattr(collector, "save_audio", fake_save_audio)
    monkeypatch.setattr(collector.time, "sleep", fake_sleep)

    args = argparse.Namespace(
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
