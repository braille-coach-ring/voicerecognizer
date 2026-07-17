"""
インターフェースのデフォルト実装
"""

from typing import Tuple
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import torch
from time import sleep

from core.interfaces import (
    ConfigProvider,
    AudioRecorder,
    AudioPreprocessor,
    MelSpectrogramConverter,
    ModelLoader,
    Inferencer,
    AudioSaver,
    ResultPresenter,
    CountdownDisplay,
)
from model import HiraganaCNN


class DefaultConfig(ConfigProvider):
    """デフォルト設定プロバイダー"""

    def __init__(
        self,
        sample_rate: int = 16000,
        record_seconds: float = 1.0,
        target_length: float = 1.0,
        top_db: int = 30,
        n_mels: int = 64,
        labels: list = None,
        model_path: str = "best_model.pth",
        device: torch.device = None,
        audio_output_file: str = "predicted_audio.wav",
    ):
        self._sample_rate = sample_rate
        self._record_seconds = record_seconds
        self._target_length = target_length
        self._top_db = top_db
        self._n_mels = n_mels
        self._labels = labels or sorted(["a", "e", "i", "o", "u"])
        self._model_path = model_path
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._audio_output_file = audio_output_file

    def get_sample_rate(self) -> int:
        return self._sample_rate

    def get_record_seconds(self) -> float:
        return self._record_seconds

    def get_target_length(self) -> float:
        return self._target_length

    def get_top_db(self) -> int:
        return self._top_db

    def get_n_mels(self) -> int:
        return self._n_mels

    def get_labels(self) -> list:
        return self._labels

    def get_model_path(self) -> str:
        return self._model_path

    def get_device(self) -> torch.device:
        return self._device

    def get_audio_output_file(self) -> str:
        return self._audio_output_file


class DefaultAudioRecorder(AudioRecorder):
    """デフォルト音声レコーダー（マイク入力）"""

    def record(self, duration: float, sample_rate: int) -> np.ndarray:
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        return audio.flatten()


class DefaultAudioPreprocessor(AudioPreprocessor):
    """デフォルト音声前処理"""

    def preprocess(
        self,
        audio: np.ndarray,
        sample_rate: int,
        target_length: float,
        top_db: int,
    ) -> np.ndarray:
        # 無音除去
        y, _ = librosa.effects.trim(audio, top_db=top_db)

        # 音量正規化
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        # 長さ統一
        target_samples = int(target_length * sample_rate)

        if len(y) > target_samples:
            y = y[:target_samples]
        else:
            y = np.pad(y, (0, target_samples - len(y)))

        return y


class DefaultMelSpectrogramConverter(MelSpectrogramConverter):
    """デフォルトメルスペクトログラム変換"""

    def convert(
        self,
        audio: np.ndarray,
        sample_rate: int,
        n_mels: int,
    ) -> torch.Tensor:
        # メルスペクトログラム計算
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=n_mels,
        )

        # デシベル変換
        mel = librosa.power_to_db(mel, ref=np.max)

        # 標準化
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        # Tensor 変換
        mel = torch.tensor(mel, dtype=torch.float32)

        # バッチ化 (64,101) → (1,1,64,101)
        mel = mel.unsqueeze(0).unsqueeze(0)

        return mel


class DefaultModelLoader(ModelLoader):
    """デフォルトモデルローダー"""

    def __init__(self):
        self._model = None

    def load_model(self, model_path: str, num_classes: int, device: torch.device):
        self._model = HiraganaCNN(num_classes=num_classes)
        self._model.load_state_dict(torch.load(model_path, map_location=device))
        self._model.to(device)
        self._model.eval()

    def get_model(self):
        return self._model


class DefaultInferencer(Inferencer):
    """デフォルト推論エンジン"""

    def __init__(self, model_loader: ModelLoader):
        self._model_loader = model_loader

    def predict(
        self,
        mel_spec: torch.Tensor,
        device: torch.device,
    ) -> Tuple[int, torch.Tensor]:
        model = self._model_loader.get_model()
        mel_spec = mel_spec.to(device)

        with torch.no_grad():
            output = model(mel_spec)
            probs = torch.softmax(output, dim=1)[0]
            pred = torch.argmax(probs).item()

        return pred, probs


class DefaultAudioSaver(AudioSaver):
    """デフォルト音声セーバー"""

    def save(self, audio: np.ndarray, sample_rate: int, filename: str) -> None:
        sf.write(filename, audio, sample_rate)


class DefaultResultPresenter(ResultPresenter):
    """デフォルト結果表示（コンソール出力）"""

    def present(
        self,
        predicted_label: str,
        probabilities: dict,
    ) -> None:
        print("\n" + "=" * 30)
        print("認識結果")
        print("=" * 30)

        print(f"\n予測: {predicted_label}\n")

        for label, prob in probabilities.items():
            print(f"{label} : {prob * 100:.2f}%")


class DefaultCountdownDisplay(CountdownDisplay):
    """デフォルトカウントダウン表示"""

    def show_countdown(self, seconds: int) -> None:
        for i in range(seconds, 0, -1):
            print(i)
            sleep(1)

    def show_message(self, message: str) -> None:
        print(message)
