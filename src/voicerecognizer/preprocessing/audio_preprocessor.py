import logging
import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from voicerecognizer.config import (
    DEFAULT_AUDIO_CONFIG,
    DEFAULT_PREPROCESS_CONFIG,
    DEFAULT_RECOGNITION_CONFIG,
)
from voicerecognizer.preprocessing.threshold_calculator import (
    AbstractSilenceThresholdCalculator,
    FixedSilenceThresholdCalculator,
)

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    音声前処理クラス (Issue #17 対応)
    ・音量実効値（RMS）ベースのダイナミックレンジ補正 ＆ tanh ソフトクリッピング（「お(o)」の歪み誤認・ブツ切れ防止）
    ・適正 split_top_db (最大 40dB) による無音境界分離
    ・「お」の低音域フォルマント（100Hz〜200Hz）減衰音を保持する 80ms/120ms 安全余白マージン
    ・波形不連続ノイズ（クリック音）を排除する 20ms Raised-Cosine (Hanning) フェード処理
    ・ターゲット長固定（末尾カット時にもフェードアウト適用）
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_AUDIO_CONFIG.sample_rate,
        target_length_seconds: float = DEFAULT_RECOGNITION_CONFIG.target_length_seconds,
        top_db: float = DEFAULT_PREPROCESS_CONFIG.top_db,
        target_rms: float = DEFAULT_PREPROCESS_CONFIG.target_rms,
        threshold_calculator: AbstractSilenceThresholdCalculator | None = None,
    ):
        self.sample_rate = sample_rate
        self.target_length_seconds = target_length_seconds
        self.top_db = top_db
        self.target_rms = target_rms
        self.threshold_calculator = threshold_calculator or FixedSilenceThresholdCalculator(
            top_db=float(top_db)
        )
        logger.info(
            "AudioPreprocessorの初期化完了 (RMSダイナミックレンジ補正 + お(o)ブツ切れ防止適用)"
        )

    def load(self, audio: str | Path | np.ndarray) -> np.ndarray:
        if isinstance(audio, np.ndarray):
            return audio.astype(np.float32).reshape(-1)

        waveform, _ = librosa.load(Path(audio), sr=self.sample_rate, mono=True)
        return waveform.astype(np.float32)

    def preprocess_waveform(
        self,
        audio: Any,
        pad_to_target: bool = True,
        min_length_seconds: float = 0.2,
    ) -> np.ndarray:
        t_prep_start = time.perf_counter()
        waveform = self.load(audio)
        self.threshold_calculator.update(waveform)
        current_top_db = self.threshold_calculator.get_silence_threshold()

        # 1. 音声セグメント分離用の適正 top_db 算出 (最大40dBにクランプし雑音や過大ピークによる語尾切りを防止)
        split_top_db = (
            min(float(current_top_db), 40.0) if current_top_db > 40.0 else float(current_top_db)
        )

        # 無音境界の検索 (frame_length=1024, hop_length=256 で基本周波数の低周波成分を精密補足)
        intervals = librosa.effects.split(
            waveform,
            top_db=split_top_db,
            frame_length=1024,
            hop_length=256,
        )
        onset_ms = 0.0
        offset_ms = 0.0
        speech_duration_ms = 0.0

        if len(intervals) > 0:
            start_idx = intervals[0][0]
            end_idx = intervals[-1][1]
            onset_ms = float(start_idx / self.sample_rate * 1000.0)
            offset_ms = float(end_idx / self.sample_rate * 1000.0)
            speech_duration_ms = float((end_idx - start_idx) / self.sample_rate * 1000.0)

            # 2. 「頭切れ・語尾切れ」絶対防止マージン (先頭120ms / 末尾150ms の安全余白)
            start_margin = int(self.sample_rate * 0.12)  # 120ms
            end_margin = int(self.sample_rate * 0.15)  # 150ms
            start_idx = max(0, start_idx - start_margin)
            end_idx = min(len(waveform), end_idx + end_margin)
            waveform = waveform[start_idx:end_idx]

        # 3. 低周波音(100Hz/周期10ms)の波形不連続ノイズ（ブツッ音）を抑える 20ms Raised-Cosine フェード処理
        fade_samples = int(self.sample_rate * 0.020)  # 20ms (100Hz波形の2周期分を完全にカバー)
        if len(waveform) > fade_samples * 2:
            fade_in = 0.5 * (
                1.0 - np.cos(np.pi * np.linspace(0, 1, fade_samples, dtype=np.float32))
            )
            fade_out = 0.5 * (
                1.0 + np.cos(np.pi * np.linspace(0, 1, fade_samples, dtype=np.float32))
            )
            waveform[:fade_samples] *= fade_in
            waveform[-fade_samples:] *= fade_out

        # 4. RMSベースのダイナミックレンジ補正 ＆ tanh ソフトクリッピング
        waveform = self._normalize_volume(waveform)
        result_waveform = self._fit_length(
            waveform,
            pad_to_target=pad_to_target,
            min_length_seconds=min_length_seconds,
        )

        t_prep_end = time.perf_counter()
        self.last_stats = {
            "onset_ms": onset_ms,
            "offset_ms": offset_ms,
            "speech_duration_ms": speech_duration_ms,
            "preprocess_latency_ms": (t_prep_end - t_prep_start) * 1000.0,
        }

        return result_waveform

    def _normalize_volume(self, waveform: np.ndarray) -> np.ndarray:
        """
        音量実効値（RMS）ベースのダイナミックレンジ補正 ＆ tanh ソフトクリッピング。
        「お(o)」等の低周波フォルマント・減衰音の音量を適正化し、
        アタック音の過大ピーク歪みや追従不良による「ブツ切れ・歪み誤認」を物理排除します。
        """
        if waveform.size == 0:
            return waveform

        rms = np.sqrt(np.mean(waveform**2) + 1e-8)
        target_rms = self.target_rms

        if rms > 1e-5:
            gain = target_rms / rms
            gain = min(gain, 8.0)  # 過大増幅防止ガード
            scaled = waveform * gain
            # ソフトクリッピング・リミッター (tanh) によりアタック音の過大ピーク歪みを滑らかに圧縮
            compressed = np.tanh(scaled) * 0.95
            return compressed.astype(np.float32)

        return waveform

    def _fit_length(
        self,
        waveform: np.ndarray,
        pad_to_target: bool = True,
        min_length_seconds: float = 0.2,
    ) -> np.ndarray:
        target_samples = int(self.target_length_seconds * self.sample_rate)
        if len(waveform) > target_samples:
            # ターゲット長でカットする際にも末尾20msにフェードアウトを施しブツ切れを防止
            truncated = waveform[:target_samples].copy()
            fade_samples = int(self.sample_rate * 0.020)
            if len(truncated) > fade_samples:
                fade_out = 0.5 * (
                    1.0 + np.cos(np.pi * np.linspace(0, 1, fade_samples, dtype=np.float32))
                )
                truncated[-fade_samples:] *= fade_out
            return truncated

        if not pad_to_target:
            min_samples = int(min_length_seconds * self.sample_rate)
            if len(waveform) < min_samples:
                return np.pad(waveform, (0, min_samples - len(waveform)))
            return waveform

        return np.pad(waveform, (0, target_samples - len(waveform)))
