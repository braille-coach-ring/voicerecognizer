"""
Audio Data Augmentation Module

音声波形 (1D np.ndarray float32) に対するデータ拡張（Data Augmentation）機能を提供します。
・ノイズ加算 (Additive Gaussian Noise)
・音量変更 (Gain / Amplitude Scaling)
・タイムシフト (Time Shift)
・軽い速度変化 (Mild Speed Perturbation)
・軽いピッチ変化 (Mild Pitch Shift)
・実機ノイズ混合 (Optional Recorded Device Noise Mix)
"""

from pathlib import Path

import numpy as np
import soundfile as sf


class AudioAugmentor:
    """
    音声データに対するデータ拡張を実行するパイプラインクラス。
    """

    def __init__(
        self,
        noise_level: float = 0.005,
        gain_range: tuple[float, float] = (0.8, 1.2),
        shift_max_ratio: float = 0.1,
        speed_range: tuple[float, float] = (0.95, 1.05),
        pitch_shift_steps: tuple[float, float] = (-0.5, 0.5),
        noise_mix_snr_db_range: tuple[float, float] = (18.0, 30.0),
        noise_file_paths: list[str | Path] | None = None,
        sample_rate: int = 16000,
        p: float = 0.5,
        seed: int | None = None,
    ):
        """
        Args:
            noise_level: 追加するガウスノイズの標準偏差 scale
            gain_range: 音量乗算の最小値と最大値のタプル (min, max)
            shift_max_ratio: 波形長に対する最大時間シフト率 (0.1 = 最大10%シフト)
            speed_range: 速度変化倍率。単音節向けに 1.0 近辺を推奨
            pitch_shift_steps: 半音単位のピッチ変化範囲。単音節向けに弱めを推奨
            noise_mix_snr_db_range: 実機ノイズ混合時の SNR 範囲
            noise_file_paths: 混合用の実機ノイズ wav ファイル群
            sample_rate: pitch shift とノイズ読み込みに使うサンプルレート
            p: 各拡張が適用される確率 (0.0 ~ 1.0)
            seed: 乱数シード (テスト用)
        """
        self.noise_level = noise_level
        self.gain_range = gain_range
        self.shift_max_ratio = shift_max_ratio
        self.speed_range = speed_range
        self.pitch_shift_steps = pitch_shift_steps
        self.noise_mix_snr_db_range = noise_mix_snr_db_range
        self.sample_rate = sample_rate
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.noise_waveforms = self._load_noise_waveforms(noise_file_paths or [])

    def _load_noise_waveforms(self, noise_file_paths: list[str | Path]) -> list[np.ndarray]:
        noise_waveforms: list[np.ndarray] = []
        for noise_path in noise_file_paths:
            try:
                waveform, sr = sf.read(noise_path, dtype="float32", always_2d=False)
            except (OSError, RuntimeError, ValueError):
                continue
            waveform = np.asarray(waveform, dtype=np.float32)
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1, dtype=np.float32)
            waveform = waveform.reshape(-1)
            if waveform.size == 0:
                continue
            if sr != self.sample_rate:
                try:
                    import librosa

                    waveform = librosa.resample(
                        waveform,
                        orig_sr=sr,
                        target_sr=self.sample_rate,
                    ).astype(np.float32)
                except Exception:
                    continue
            noise_waveforms.append(np.ascontiguousarray(waveform, dtype=np.float32))
        return noise_waveforms

    def _fit_length(self, waveform: np.ndarray, target_length: int) -> np.ndarray:
        if len(waveform) == target_length:
            return waveform.astype(np.float32)
        if len(waveform) > target_length:
            return waveform[:target_length].astype(np.float32)
        return np.pad(waveform, (0, target_length - len(waveform))).astype(np.float32)

    def add_noise(self, waveform: np.ndarray) -> np.ndarray:
        """ガウスノイズを加算"""
        if self.rng.random() > self.p:
            return waveform
        noise = self.rng.standard_normal(size=waveform.shape, dtype=np.float32) * self.noise_level
        return waveform + noise

    def change_gain(self, waveform: np.ndarray) -> np.ndarray:
        """音量をランダムに拡大・縮小"""
        if self.rng.random() > self.p:
            return waveform
        gain = self.rng.uniform(self.gain_range[0], self.gain_range[1])
        return (waveform * gain).astype(np.float32)

    def change_speed(self, waveform: np.ndarray) -> np.ndarray:
        """軽い速度変化。出力長は元波形と同じに戻す。"""
        if self.rng.random() > self.p:
            return waveform
        if len(waveform) < 2:
            return waveform
        speed = float(self.rng.uniform(self.speed_range[0], self.speed_range[1]))
        if abs(speed - 1.0) < 1e-3:
            return waveform

        target_length = len(waveform)
        stretched_length = max(2, round(target_length / speed))
        source_positions = np.linspace(0, target_length - 1, stretched_length)
        resampled = np.interp(
            source_positions,
            np.arange(target_length),
            waveform,
        ).astype(np.float32)
        return self._fit_length(resampled, target_length)

    def shift_pitch(self, waveform: np.ndarray) -> np.ndarray:
        """軽いピッチ変化。librosa が使えない場合は元波形を返す。"""
        if self.rng.random() > self.p:
            return waveform
        n_steps = float(self.rng.uniform(self.pitch_shift_steps[0], self.pitch_shift_steps[1]))
        if abs(n_steps) < 1e-3:
            return waveform
        try:
            import librosa

            shifted = librosa.effects.pitch_shift(
                y=waveform,
                sr=self.sample_rate,
                n_steps=n_steps,
            )
        except Exception:
            return waveform
        return self._fit_length(np.asarray(shifted, dtype=np.float32), len(waveform))

    def mix_device_noise(self, waveform: np.ndarray) -> np.ndarray:
        """録音済みの実機ノイズを SNR 指定で混合する。"""
        if not self.noise_waveforms or self.rng.random() > self.p:
            return waveform

        noise = self.noise_waveforms[int(self.rng.integers(0, len(self.noise_waveforms)))]
        if len(noise) < len(waveform):
            repeats = int(np.ceil(len(waveform) / max(len(noise), 1)))
            noise = np.tile(noise, repeats)
        if len(noise) > len(waveform):
            max_start = len(noise) - len(waveform)
            start = int(self.rng.integers(0, max_start + 1))
            noise = noise[start : start + len(waveform)]

        signal_rms = float(np.sqrt(np.mean(waveform**2) + 1e-8))
        noise_rms = float(np.sqrt(np.mean(noise**2) + 1e-8))
        if signal_rms <= 1e-6 or noise_rms <= 1e-6:
            return waveform

        snr_db = float(
            self.rng.uniform(self.noise_mix_snr_db_range[0], self.noise_mix_snr_db_range[1])
        )
        noise_scale = signal_rms / (noise_rms * (10.0 ** (snr_db / 20.0)))
        mixed = waveform + noise.astype(np.float32) * noise_scale
        return np.clip(mixed, -1.0, 1.0).astype(np.float32)

    def shift_time(self, waveform: np.ndarray) -> np.ndarray:
        """時間軸上で波形を前後シフト（端は0パディング）"""
        if self.rng.random() > self.p:
            return waveform
        max_shift = int(len(waveform) * self.shift_max_ratio)
        if max_shift <= 0:
            return waveform
        shift = int(self.rng.integers(-max_shift, max_shift + 1))
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
        aug_waveform = self.change_speed(aug_waveform)
        aug_waveform = self.shift_pitch(aug_waveform)
        aug_waveform = self.add_noise(aug_waveform)
        aug_waveform = self.mix_device_noise(aug_waveform)
        aug_waveform = self.change_gain(aug_waveform)
        aug_waveform = self.shift_time(aug_waveform)
        return np.ascontiguousarray(aug_waveform, dtype=np.float32)
