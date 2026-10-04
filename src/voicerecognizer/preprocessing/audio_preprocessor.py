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
        enable_trimming: bool = False,
        onset_pre_roll_seconds: float | None = DEFAULT_PREPROCESS_CONFIG.onset_pre_roll_seconds,
    ):
        self.sample_rate = sample_rate
        self.target_length_seconds = target_length_seconds
        self.top_db = top_db
        self.target_rms = target_rms
        self.enable_trimming = enable_trimming
        # None で発話開始位置への揃えを無効化
        self.onset_pre_roll_seconds = onset_pre_roll_seconds
        logger.info(
            "AudioPreprocessorの初期化完了 (トリミング=%s, RMSダイナミックレンジ補正適用)",
            self.enable_trimming,
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

        onset_ms = 0.0
        offset_ms = float(len(waveform) / self.sample_rate * 1000.0)
        speech_duration_ms = offset_ms

        if self.enable_trimming:
            intervals = librosa.effects.split(
                waveform,
                top_db=self.top_db,
                frame_length=1024,
                hop_length=256,
            )
            if len(intervals) > 0:
                start_idx = intervals[0][0]
                end_idx = intervals[-1][1]
                onset_ms = float(start_idx / self.sample_rate * 1000.0)
                offset_ms = float(end_idx / self.sample_rate * 1000.0)
                speech_duration_ms = float((end_idx - start_idx) / self.sample_rate * 1000.0)

                # 「頭切れ・語尾切れ」防止マージン (先頭120ms / 末尾150ms)
                start_margin = int(self.sample_rate * 0.12)
                end_margin = int(self.sample_rate * 0.15)
                start_idx = max(0, start_idx - start_margin)
                end_idx = min(len(waveform), end_idx + end_margin)
                waveform = waveform[start_idx:end_idx]
        elif self.onset_pre_roll_seconds is not None:
            # マイク推論時の波形は 1 秒窓の後半 (中央値 440ms) に発話があり、先頭 0.6s 固定長
            # 切り出しで子音の位置が学習データ (発話開始 30〜70ms) とずれる / 発話自体が欠落する。
            # 発話開始の直前から切り出し、学習時と同じ位置に子音が来るようにする。
            speech_start = self.find_speech_onset(waveform)
            onset_ms = float(speech_start / self.sample_rate * 1000.0)
            speech_duration_ms = offset_ms - onset_ms
            start_idx = max(0, speech_start - int(self.sample_rate * self.onset_pre_roll_seconds))
            waveform = waveform[start_idx:]

        # ※ 子音のアタックエネルギー（5〜15ms）を保持するため、先頭フェードインは適用しない
        # 末尾の急激な切断ノイズのみ最小限(5ms)ケア
        fade_samples = int(self.sample_rate * 0.005)
        if len(waveform) > fade_samples * 2:
            fade_out = 0.5 * (
                1.0 + np.cos(np.pi * np.linspace(0, 1, fade_samples, dtype=np.float32))
            )
            waveform[-fade_samples:] *= fade_out

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

    def find_speech_onset(self, waveform: np.ndarray) -> int:
        """10ms フレームの RMS から発話開始のサンプル位置を推定する (見つからなければ 0)。

        しきい値は「暗騒音 (下位 10% フレーム) の 3 倍」と「ピークの -26dB」の大きい方。
        母音ピーク基準だけでは弱い摩擦音 (s/sh/h) の立ち上がりを取りこぼすため、
        呼び出し側で onset_pre_roll_seconds 分の余白を付ける。
        """
        frame = int(self.sample_rate * 0.01)
        n_frames = len(waveform) // frame
        if n_frames < 2:
            return 0
        energy = np.sqrt(
            np.mean(waveform[: n_frames * frame].reshape(n_frames, frame) ** 2, axis=1)
        )
        if float(energy.max()) < 1e-4:
            return 0
        threshold = max(float(np.percentile(energy, 10)) * 3.0, float(energy.max()) * 0.05)
        active = energy >= threshold
        # 単発のクリック音で誤検出しないよう 2 フレーム連続を要求
        starts = np.flatnonzero(active[:-1] & active[1:])
        return int(starts[0] * frame) if starts.size else 0

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
