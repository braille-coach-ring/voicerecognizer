from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


from config import DEFAULT_AUDIO_CONFIG


def load_audio(path: str | Path, sample_rate: int = DEFAULT_AUDIO_CONFIG.sample_rate) -> np.ndarray:
    waveform, _ = librosa.load(Path(path), sr=sample_rate, mono=True)
    return waveform.astype(np.float32)


def save_audio(path: str | Path, audio: np.ndarray, sample_rate: int = DEFAULT_AUDIO_CONFIG.sample_rate) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sample_rate)
