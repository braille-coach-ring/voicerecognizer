import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from voicerecognizer.config import DEFAULT_AUDIO_CONFIG

logger = logging.getLogger(__name__)


def load_audio(path: str | Path, sample_rate: int = DEFAULT_AUDIO_CONFIG.sample_rate) -> np.ndarray:
    try:
        waveform, _ = librosa.load(Path(path), sr=sample_rate, mono=True)
        return waveform.astype(np.float32)
    except Exception as e:
        logger.error("音声ファイルの読み込みに失敗しました: %s (%s)", path, e, exc_info=True)
        raise e


def save_audio(
    path: str | Path,
    audio: np.ndarray,
    sample_rate: int = DEFAULT_AUDIO_CONFIG.sample_rate,
) -> None:
    output_path = Path(path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, sample_rate)
        logger.debug("音声ファイルを書き込み保存しました: %s", output_path)
    except Exception as e:
        logger.error(
            "音声ファイルの書き込み保存に失敗しました: %s (%s)", output_path, e, exc_info=True
        )
        raise e
