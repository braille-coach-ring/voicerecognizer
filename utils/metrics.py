import numpy as np


def accuracy(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def audio_level_summary(audio: np.ndarray) -> dict[str, float]:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
    db = float(20 * np.log10(rms + 1e-10))
    return {"peak": peak, "rms": rms, "db": db}
