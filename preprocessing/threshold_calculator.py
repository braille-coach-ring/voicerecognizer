from abc import ABC, abstractmethod
import numpy as np
import logging
logger = logging.getLogger(__name__)

from config import DEFAULT_PREPROCESS_CONFIG, PreprocessConfig


class AbstractSilenceThresholdCalculator(ABC):
    """無音検出判定の閾値（top_db相当）を動的または固定で決定する計算器の抽象基底クラス"""

    @abstractmethod
    def update(self, audio_chunk: np.ndarray) -> None:
        """新しい音声フレーム（波形データ）を受け取り、内部状態を更新する"""
        pass

    @abstractmethod
    def get_silence_threshold(self) -> float:
        """現在の最適判定閾値（top_db相当）を返す"""
        pass


class FixedSilenceThresholdCalculator(AbstractSilenceThresholdCalculator):
    """常に一定の top_db を返す計算器（固定値モード）"""

    def __init__(self, top_db: float = 30.0):
        self._top_db = float(top_db)
        logger.info(f"固定値モード: top_db = {self._top_db}")

    def update(self, audio_chunk: np.ndarray) -> None:
        # 固定値モードのため更新は不要
        pass

    def get_silence_threshold(self) -> float:
        return self._top_db


class AdaptiveSilenceThresholdCalculator(AbstractSilenceThresholdCalculator):
    """音声のバックグラウンドノイズを追跡し、適応的に top_db を動的計算する計算器（移動平均モード）"""

    def __init__(self, config: PreprocessConfig | None = None):
        cfg = config or DEFAULT_PREPROCESS_CONFIG
        self._current_top_db: float = float(cfg.top_db)
        self._estimated_noise_db: float = -60.0  # ノイズ床の初期推定値 (dB)
        self._alpha: float = cfg.noise_update_rate
        self._min_top_db: float = float(cfg.min_top_db)
        self._max_top_db: float = float(cfg.max_top_db)
        logger.info(f"適応値モード: ノイズ床の初期値: {self._estimated_noise_db} dB, ノイズアップデートレート: {self._alpha} , デシベル下限値上限値: {self._min_top_db}-{self._max_top_db}")

    def update(self, audio_chunk: np.ndarray) -> None:
        if audio_chunk is None or audio_chunk.size == 0:
            return

        # 音声波形のRMS（実効値）からdB値を計算
        rms = np.sqrt(np.mean(audio_chunk**2) + 1e-10)
        current_db = float(20 * np.log10(rms + 1e-10))

        # ノイズ推定の更新（背景ノイズレベル以下と思われる範囲で移動平均）
        if current_db < self._estimated_noise_db + 15.0:
            self._estimated_noise_db = (
                1.0 - self._alpha
            ) * self._estimated_noise_db + self._alpha * current_db

        # 推定されたノイズレベルを元に最適 top_db を算出
        calculated_top_db = 30.0 + (self._estimated_noise_db + 50.0) * 0.5

        # クランプ処理（指定された最小・最大範囲内に収める）
        self._current_top_db = float(
            np.clip(calculated_top_db, self._min_top_db, self._max_top_db)
        )

        logger.info(f"現在のノイズ床: {self._estimated_noise_db} dB, 現在の適応閾値 (top_db): {self._current_top_db:.2f} dB")

    def get_silence_threshold(self) -> float:
        return self._current_top_db


def create_threshold_calculator(
    config: PreprocessConfig | None = None,
) -> AbstractSilenceThresholdCalculator:
    """PreprocessConfig に基づいて適切な閾値計算器を生成するファクトリ関数"""
    cfg = config or DEFAULT_PREPROCESS_CONFIG
    if cfg.dynamic_threshold_enabled:
        return AdaptiveSilenceThresholdCalculator(config=cfg)
    return FixedSilenceThresholdCalculator(top_db=float(cfg.top_db))
