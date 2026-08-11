"""
Audio Data Augmentation Module

音声波形 (1D np.ndarray float32) に対するデータ拡張（Data Augmentation）機能を提供します。
・ノイズ加算 (Additive Gaussian Noise)
・音量変更 (Gain / Amplitude Scaling)
・タイムシフト (Time Shift)
"""

import numpy as np


class AudioAugmentor:
    """
    音声データに対するデータ拡張を実行するパイプラインクラス。
    """

    def __init__(
        self,
        noise_level: float = 0.005,
        gain_range: tuple[float, float] = (0.8, 1.2),
        shift_max_ratio: float = 0.1,
        p: float = 0.5,
        seed: int | None = None,
    ):
        """
        Args:
            noise_level: 追加するガウスノイズの標準偏差 scale
            gain_range: 音量乗算の最小値と最大値のタプル (min, max)
            shift_max_ratio: 波形長に対する最大時間シフト率 (0.1 = 最大10%シフト)
            p: 各拡張が適用される確率 (0.0 ～ 1.0)
            seed: 乱数シード (テスト用)
        """
        self.noise_level = noise_level
        self.gain_range = gain_range
        self.shift_max_ratio = shift_max_ratio
        self.p = p
        if seed is not None:
            np.random.seed(seed)

    def add_noise(self, waveform: np.ndarray) -> np.ndarray:
        """ガウスノイズを加算"""
        if np.random.rand() > self.p:
            return waveform
        noise = np.random.randn(*waveform.shape).astype(np.float32) * self.noise_level
        return waveform + noise

    def change_gain(self, waveform: np.ndarray) -> np.ndarray:
        """音量をランダムに拡大・縮小"""
        if np.random.rand() > self.p:
            return waveform
        gain = np.random.uniform(self.gain_range[0], self.gain_range[1])
        return (waveform * gain).astype(np.float32)

    def shift_time(self, waveform: np.ndarray) -> np.ndarray:
        """時間軸上で波形を前後シフト（端は0パディング）"""
        if np.random.rand() > self.p:
            return waveform
        max_shift = int(len(waveform) * self.shift_max_ratio)
        if max_shift <= 0:
            return waveform
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift == 0:
            return waveform

        result = np.zeros_like(waveform)
        if shift > 0:
            result[shift:] = waveform[:-shift]
        else:
            result[:shift] = waveform[-shift:]
        return result

    def augment(self, waveform: np.ndarray) -> np.ndarray:
        """全拡張を順次適用した新しい波形配列を返す"""
        aug_waveform = np.ascontiguousarray(waveform, dtype=np.float32).copy()
        aug_waveform = self.add_noise(aug_waveform)
        aug_waveform = self.change_gain(aug_waveform)
        aug_waveform = self.shift_time(aug_waveform)
        return aug_waveform
