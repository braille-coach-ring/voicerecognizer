"""
抽象インターフェース定義
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import torch


class ConfigProvider(ABC):
    """設定プロバイダーのインターフェース"""

    @abstractmethod
    def get_sample_rate(self) -> int:
        pass

    @abstractmethod
    def get_record_seconds(self) -> float:
        pass

    @abstractmethod
    def get_target_length(self) -> float:
        pass

    @abstractmethod
    def get_top_db(self) -> int:
        pass

    @abstractmethod
    def get_n_mels(self) -> int:
        pass

    @abstractmethod
    def get_labels(self) -> list:
        pass

    @abstractmethod
    def get_model_path(self) -> str:
        pass

    @abstractmethod
    def get_device(self) -> torch.device:
        pass

    @abstractmethod
    def get_audio_output_file(self) -> str:
        pass


class AudioRecorder(ABC):
    """音声録音のインターフェース"""

    @abstractmethod
    def record(self, duration: float, sample_rate: int) -> np.ndarray:
        """
        音声を録音する

        Args:
            duration: 録音時間（秒）
            sample_rate: サンプリングレート

        Returns:
            録音済み音声配列 (N,)
        """
        pass


class AudioPreprocessor(ABC):
    """音声前処理のインターフェース"""

    @abstractmethod
    def preprocess(
        self, audio: np.ndarray, sample_rate: int, target_length: float, top_db: int
    ) -> np.ndarray:
        """
        音声を前処理する

        Args:
            audio: 入力音声配列
            sample_rate: サンプリングレート
            target_length: 目標時間
            top_db: 無音除去しきい値

        Returns:
            前処理済み音声配列
        """
        pass


class MelSpectrogramConverter(ABC):
    """メルスペクトログラム変換のインターフェース"""

    @abstractmethod
    def convert(self, audio: np.ndarray, sample_rate: int, n_mels: int) -> torch.Tensor:
        """
        音声をメルスペクトログラムに変換

        Args:
            audio: 入力音声配列
            sample_rate: サンプリングレート
            n_mels: メルフィルタ数

        Returns:
            メルスペクトログラムテンソル (1, 1, n_mels, time_steps)
        """
        pass


class ModelLoader(ABC):
    """モデルロードのインターフェース"""

    @abstractmethod
    def load_model(self, model_path: str, num_classes: int, device: torch.device):
        """モデルを読み込む"""
        pass

    @abstractmethod
    def get_model(self):
        """ロード済みモデルを取得"""
        pass


class Inferencer(ABC):
    """推論処理のインターフェース"""

    @abstractmethod
    def predict(
        self, mel_spec: torch.Tensor, device: torch.device
    ) -> Tuple[int, torch.Tensor]:
        """
        推論を実行

        Args:
            mel_spec: メルスペクトログラムテンソル
            device: GPU/CPUデバイス

        Returns:
            (予測クラスインデックス, 各クラスの確率)
        """
        pass


class AudioSaver(ABC):
    """音声保存のインターフェース"""

    @abstractmethod
    def save(self, audio: np.ndarray, sample_rate: int, filename: str) -> None:
        """音声をファイルに保存"""
        pass


class ResultPresenter(ABC):
    """結果表示のインターフェース"""

    @abstractmethod
    def present(self, predicted_label: str, probabilities: dict) -> None:
        """
        結果を表示

        Args:
            predicted_label: 予測ラベル
            probabilities: {ラベル: 確率} の辞書
        """
        pass


class CountdownDisplay(ABC):
    """カウントダウン表示のインターフェース"""

    @abstractmethod
    def show_countdown(self, seconds: int) -> None:
        """秒数分のカウントダウンを表示"""
        pass

    @abstractmethod
    def show_message(self, message: str) -> None:
        """メッセージを表示"""
        pass
