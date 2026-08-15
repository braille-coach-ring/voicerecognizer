import json
import logging
import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch

from voicerecognizer.config import (
    DEFAULT_AUDIO_CONFIG,
    DEFAULT_PREPROCESS_CONFIG,
    DEFAULT_RECOGNITION_CONFIG,
    PUBLIC_DEFAULT_HF_REPO_ID,
    HuggingFaceConfig,
)
from voicerecognizer.core.exceptions import ModelNotFoundError
from voicerecognizer.core.interfaces import RecognitionStrategy
from voicerecognizer.models.cnn.hiragana_cnn import HiraganaCNN
from voicerecognizer.preprocessing.audio_preprocessor import AudioPreprocessor
from voicerecognizer.preprocessing.threshold_calculator import AbstractSilenceThresholdCalculator
from voicerecognizer.utils.model_uploader import download_latest_team_weights_if_needed

logger = logging.getLogger(__name__)


class CNNRecognizer(RecognitionStrategy):
    def __init__(
        self,
        model_path: str | Path = DEFAULT_RECOGNITION_CONFIG.cnn_weight_path,
        labels: tuple[str, ...] | list[str] = DEFAULT_RECOGNITION_CONFIG.labels,
        sample_rate: int = DEFAULT_AUDIO_CONFIG.sample_rate,
        target_length_seconds: float = DEFAULT_RECOGNITION_CONFIG.target_length_seconds,
        top_db: float = DEFAULT_PREPROCESS_CONFIG.top_db,
        n_mels: int = DEFAULT_PREPROCESS_CONFIG.n_mels,
        n_fft: int = DEFAULT_PREPROCESS_CONFIG.n_fft,
        hop_length: int = DEFAULT_PREPROCESS_CONFIG.hop_length,
        device: torch.device | None = None,
        threshold_calculator: AbstractSilenceThresholdCalculator | None = None,
        auto_download: bool = True,
        hf_repo_id: str | None = None,
        hf_token: str | None = None,
    ):
        self.model_path = Path(model_path)
        self.auto_download = auto_download
        self._last_download_error: str | None = None
        if hf_repo_id is not None and hf_token is not None:
            self.hf_config = HuggingFaceConfig(repo_id=hf_repo_id, token=hf_token)
        elif hf_repo_id is not None:
            self.hf_config = HuggingFaceConfig(repo_id=hf_repo_id)
        elif hf_token is not None:
            self.hf_config = HuggingFaceConfig(token=hf_token)
        else:
            self.hf_config = HuggingFaceConfig()

        if not self.model_path.exists() and self.auto_download:
            logger.info(
                "ローカルに CNN モデル重みが見つかりません (%s)。Hugging Face Hub より自動ダウンロードを開始します...",
                self.model_path,
            )
            try:
                download_latest_team_weights_if_needed(
                    model_type="cnn",
                    hf_config=self.hf_config,
                    weights_dir=self.model_path.parent,
                )
                if self.model_path.exists():
                    logger.info(
                        "CNN モデル重みのダウンロードと配置が完了しました: %s", self.model_path
                    )
            except Exception as e:
                self._last_download_error = str(e)
                logger.warning("CNN 重みの自動ダウンロード中に例外が発生しました: %s", e)

        labels_json_path = self.model_path.parent / DEFAULT_RECOGNITION_CONFIG.labels_filename
        if labels_json_path.exists():
            with open(labels_json_path, encoding="utf-8") as f:
                self.labels = tuple(json.load(f))
        else:
            self.labels = tuple(labels)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.audio_preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            target_length_seconds=target_length_seconds,
            top_db=top_db,
            threshold_calculator=threshold_calculator,
        )
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.model: HiraganaCNN | None = None
        logger.info("CNNレコグナイザーの初期化完了")

    def warmup(self, audio_seconds: float = 1.0) -> None:
        """モデルを明示的にロードし、ダミー波形推論を実行してエンジンをウォームアップする。"""
        import contextlib

        self._ensure_model_loaded()
        dummy_audio = np.zeros(int(self.sample_rate * audio_seconds), dtype=np.float32)
        with contextlib.suppress(Exception):
            self.recognize(dummy_audio)
        logger.info("CNNRecognizer のウォームアップが完了しました。")

    def _ensure_model_loaded(self) -> None:
        if self.model is not None:
            return
        if not self.model_path.exists():
            if self.auto_download:
                try:
                    download_latest_team_weights_if_needed(
                        model_type="cnn",
                        hf_config=self.hf_config,
                        weights_dir=self.model_path.parent,
                    )
                except Exception as e:
                    self._last_download_error = str(e)
            if not self.model_path.exists():
                raise ModelNotFoundError(self._build_model_not_found_message())
        self.model = self._load_model()

    def _build_model_not_found_message(self, err: Exception | None = None) -> str:
        last_err = getattr(self, "_last_download_error", None)
        download_err_info = f"\n  ダウンロード例外詳細: {last_err}" if last_err else ""
        load_err_info = f"\n  ロード例外詳細: {err}" if err else ""
        repo_id = getattr(self, "hf_config", None)
        repo_id_str = repo_id.repo_id if repo_id else PUBLIC_DEFAULT_HF_REPO_ID
        return (
            f"voicerecognizer の CNN モデル重み ({DEFAULT_RECOGNITION_CONFIG.cnn_model_filename}) が見つかりません。\n\n"
            "【原因】\n"
            f"  ローカルパス ({self.model_path}) にモデルが存在せず、\n"
            f"  Hugging Face Hub ({repo_id_str}) からの自動ダウンロードも完了できませんでした。{download_err_info}{load_err_info}\n\n"
            "【使い方の確認・解決手順】\n"
            "  1. [インターネット接続]\n"
            "     初回実行時は Hugging Face Hub より自動的にモデルがダウンロードされます。\n"
            "     ネットワーク接続を確認の上、再度実行してください。\n"
            "  2. [Hugging Face 認証トークン]\n"
            "     アクセス制限やレートリミットを回避する場合は環境変数を設定してください:\n"

            '     - Windows (PowerShell): $env:HF_TOKEN = "your_token"\n'
            '     - Linux / macOS (Bash): export HF_TOKEN="your_token"\n'
            "     - または .env ファイルに HF_TOKEN=your_token を記述\n"
            "  3. [ローカルモデルの指定]\n"
            "     ローカルに既にあるモデルファイルを使用したい場合は、初期化時に model_path を渡してください:\n"
            "     >>> import voicerecognizer as vr\n"
            '     >>> recognizer = vr.CNNRecognizer(model_path="/path/to/your/best_model.pth")\n'
        )

    def recognize(self, audio: Any) -> str:
        self._ensure_model_loaded()
        t_start = time.perf_counter()

        t_prep_start = time.perf_counter()
        waveform = self._preprocess(audio)
        mel_tensor = self._create_mel(waveform)
        t_prep_end = time.perf_counter()

        t_inf_start = time.perf_counter()
        probabilities = self._predict(mel_tensor)
        t_inf_end = time.perf_counter()

        predicted_index = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_index].item())

        prep_stats = getattr(self.audio_preprocessor, "last_stats", {})
        self.last_timing_stats = {
            "onset_ms": prep_stats.get("onset_ms", 0.0),
            "offset_ms": prep_stats.get("offset_ms", 0.0),
            "speech_duration_ms": prep_stats.get("speech_duration_ms", 0.0),
            "preprocess_latency_ms": (t_prep_end - t_prep_start) * 1000.0,
            "inference_latency_ms": (t_inf_end - t_inf_start) * 1000.0,
            "total_latency_ms": (t_inf_end - t_start) * 1000.0,
            "confidence": confidence,
        }

        logger.debug("CNN 推論確率: %s", probabilities)
        return self._label_for_index(predicted_index)

    def _load_model(self) -> HiraganaCNN:
        try:
            model = HiraganaCNN(num_classes=len(self.labels))
            state_dict = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error("CNN モデルのロードに失敗しました: %s", e)
            raise ModelNotFoundError(self._build_model_not_found_message(e)) from e

    def _preprocess(self, audio: Any) -> np.ndarray:
        return self.audio_preprocessor.preprocess_waveform(audio)

    def _create_mel(self, waveform: np.ndarray) -> torch.Tensor:
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        return torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

    def _predict(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise ModelNotFoundError(self._build_model_not_found_message())
        with torch.no_grad():
            logits = self.model(mel_tensor)
            return torch.softmax(logits, dim=1)[0]

    def _label_for_index(self, index: int) -> str:
        if 0 <= index < len(self.labels):
            return self.labels[index]
        return str(index)
