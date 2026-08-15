import logging
from queue import Empty, Queue
from threading import Event

import numpy as np

from voicerecognizer.config import DEFAULT_PREPROCESS_CONFIG, PreprocessConfig

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    def __init__(
        self,
        config: PreprocessConfig | None = None,
        silence_threshold: float | None = None,
        rms_threshold: float | None = None,
        adaptive: bool = True,
    ):
        cfg = config or DEFAULT_PREPROCESS_CONFIG
        self.silence_threshold = (
            silence_threshold if silence_threshold is not None else cfg.vad_silence_threshold
        )
        self.rms_threshold = (
            rms_threshold if rms_threshold is not None else getattr(cfg, "vad_rms_threshold", 0.008)
        )
        self.min_speech_chunks = max(1, int(cfg.vad_min_speech_chunks))
        self.min_active_ratio = min(1.0, max(0.0, float(cfg.vad_min_active_ratio)))
        self.adaptive = adaptive
        self._speech_streak = 0
        self.noise_rms_floor: float | None = None
        self.noise_peak_floor: float | None = None

    def is_speech(self, audio: np.ndarray | None) -> bool:
        if audio is None:
            logger.warning("入力された音声データがNoneです")
            self._speech_streak = 0
            return False
        if audio.size == 0:
            logger.warning("入力された音声データが空です")
            self._speech_streak = 0
            return False

        abs_audio = np.abs(audio)
        max_vol = float(np.max(abs_audio))
        rms_vol = float(np.sqrt(np.mean(audio**2)))

        # 動的適応閾値の算出 (暗騒音に基づくが、無音時の誤爆や大声時の検知不能を防ぐため厳格に狭い範囲でクランプ)
        if (
            self.adaptive
            and self.noise_rms_floor is not None
            and self.noise_peak_floor is not None
        ):
            raw_silence_th = self.noise_peak_floor * 1.5
            raw_rms_th = self.noise_rms_floor * 1.8
            eff_silence_th = float(
                np.clip(
                    raw_silence_th,
                    self.silence_threshold * 0.80,
                    self.silence_threshold * 1.25,
                )
            )
            eff_rms_th = float(
                np.clip(
                    raw_rms_th,
                    self.rms_threshold * 0.80,
                    self.rms_threshold * 1.25,
                )
            )
        else:
            eff_silence_th = self.silence_threshold
            eff_rms_th = self.rms_threshold

        if max_vol < eff_silence_th:
            self._update_noise_floor(rms_vol, max_vol)
            self._speech_streak = 0
            return False

        active_ratio = float(np.mean(abs_audio >= eff_silence_th))
        if active_ratio < self.min_active_ratio:
            self._update_noise_floor(rms_vol, max_vol)
            self._speech_streak = 0
            return False

        if rms_vol < eff_rms_th:
            self._update_noise_floor(rms_vol, max_vol)
            self._speech_streak = 0
            return False

        self._speech_streak += 1
        return self._speech_streak >= self.min_speech_chunks

    def _update_noise_floor(self, rms: float, peak: float) -> None:
        """非発話フレーム時の暗騒音フロアを EMA で更新"""
        if not self.adaptive:
            return
        alpha = 0.05
        if self.noise_rms_floor is None or self.noise_peak_floor is None:
            self.noise_rms_floor = rms
            self.noise_peak_floor = peak
        else:
            self.noise_rms_floor = (1 - alpha) * self.noise_rms_floor + alpha * rms
            self.noise_peak_floor = (1 - alpha) * self.noise_peak_floor + alpha * peak

    def run(
        self,
        input_queue: Queue[np.ndarray | None],
        output_queue: Queue[np.ndarray],
        stop_event: Event,
    ) -> None:
        while not stop_event.is_set():
            try:
                audio = input_queue.get(timeout=0.1)
            except Empty:
                continue

            if audio is not None and self.is_speech(audio):
                output_queue.put(audio)
