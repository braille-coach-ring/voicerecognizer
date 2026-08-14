import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from voicerecognizer.config import (
    DEFAULT_HUGGINGFACE_CONFIG,
    DEFAULT_RECOGNITION_CONFIG,
    HuggingFaceConfig,
)
from voicerecognizer.core.exceptions import ModelNotFoundError
from voicerecognizer.core.interfaces import RecognitionStrategy
from voicerecognizer.preprocessing.audio_preprocessor import AudioPreprocessor
from voicerecognizer.utils.model_uploader import download_latest_team_weights_if_needed

logger = logging.getLogger(__name__)


class Wav2Vec2Recognizer(RecognitionStrategy):
    """
    Wav2Vec2 ONNX Runtime 推論ストラテジー

    ONNX モデル (model_mel_int8.onnx / model_int8.onnx / model_fp32.onnx) を用いて
    CPU 上で超高速かつ高精度な音声を推論認識します。
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_RECOGNITION_CONFIG.wav2vec2_best_model_dir,
        labels: tuple[str, ...] | list[str] = DEFAULT_RECOGNITION_CONFIG.labels,
        sample_rate: int = DEFAULT_RECOGNITION_CONFIG.sample_rate,
        target_length_seconds: float = DEFAULT_RECOGNITION_CONFIG.target_length_seconds,
        top_db: float = DEFAULT_RECOGNITION_CONFIG.top_db,
        dynamic_trimming: bool = True,
        auto_download: bool = True,
        candidate_filenames: tuple[str, ...] | list[str] = DEFAULT_RECOGNITION_CONFIG.wav2vec2_onnx_candidate_filenames,
        hf_repo_id: str | None = None,
        hf_token: str | None = None,
    ):
        self.model_path = Path(model_path)
        self.labels = list(labels)
        self.dynamic_trimming = dynamic_trimming
        self.candidate_filenames = tuple(candidate_filenames)
        self._last_download_error: str | None = None
        if hf_repo_id is not None and hf_token is not None:
            self.hf_config = HuggingFaceConfig(repo_id=hf_repo_id, token=hf_token)
        elif hf_repo_id is not None:
            self.hf_config = HuggingFaceConfig(repo_id=hf_repo_id)
        elif hf_token is not None:
            self.hf_config = HuggingFaceConfig(token=hf_token)
        else:
            self.hf_config = HuggingFaceConfig()

        # ONNX モデルファイルの探索
        self.onnx_model_path = self._find_onnx_model()
        if self.onnx_model_path is None and auto_download:
            logger.info(
                "ローカルに Wav2Vec2 ONNX モデルが見つかりません (%s)。Hugging Face Hub より自動ダウンロードを開始します...",
                self.model_path,
            )
            try:
                download_latest_team_weights_if_needed(
                    model_type="wav2vec2",
                    hf_config=self.hf_config,
                    weights_dir=self.model_path.parent,
                )
                self.onnx_model_path = self._find_onnx_model()
                if self.onnx_model_path is not None:
                    logger.info(
                        "Wav2Vec2 ONNX モデルのダウンロードと配置が完了しました: %s",
                        self.onnx_model_path,
                    )
            except Exception as e:
                self._last_download_error = str(e)
                logger.warning("Wav2Vec2 重みの自動ダウンロード中に例外が発生しました: %s", e)

        # 前処理内包型 (model_mel_*) 以外のモデルにフォールバックした場合の警告
        if self.onnx_model_path is not None and not self.onnx_model_path.name.startswith("model_mel_"):
            logger.warning(
                "前処理内包型 ONNX モデル (%s) が見つかりませんでした。通常モデル (%s) にフォールバックして推論を実行します。(前処理オーバーヘッドが増加する可能性があります)",
                DEFAULT_RECOGNITION_CONFIG.wav2vec2_mel_int8_onnx_filename,
                self.onnx_model_path.name,
            )

        # labels.json のロード
        labels_json = self.model_path / DEFAULT_RECOGNITION_CONFIG.labels_filename
        if labels_json.exists():
            try:
                with open(labels_json, encoding="utf-8") as f:
                    loaded_labels = json.load(f)
                    if isinstance(loaded_labels, list) and len(loaded_labels) > 0:
                        self.labels = loaded_labels
            except Exception as e:
                logger.warning("labels.json の読込失敗: %s", e)

        self.sample_rate = sample_rate
        self.audio_preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            target_length_seconds=target_length_seconds,
            top_db=top_db,
        )
        self.feature_extractor: Any | None = None
        self.session: Any | None = None
        self.input_name: str | None = None
        self.last_confidence: float | None = None
        self.last_timing_stats: dict[str, float] = {}
        logger.info(
            "Wav2Vec2Recognizer (ONNX) 初期化完了 (動的トリミング=%s): %s",
            self.dynamic_trimming,
            self.onnx_model_path,
        )

    def _find_onnx_model(self) -> Path | None:
        """ONNX モデルファイルの優先順位探索"""
        for candidate_name in self.candidate_filenames:
            candidate = self.model_path / candidate_name
            if candidate.exists():
                return candidate
        return None

    def _build_model_not_found_message(self) -> str:
        last_err = getattr(self, "_last_download_error", None)
        download_err_info = (
            f"\n  ダウンロード例外詳細: {last_err}"
            if last_err
            else ""
        )
        repo_id = getattr(self, "hf_config", DEFAULT_HUGGINGFACE_CONFIG).repo_id
        return (
            "voicerecognizer の Wav2Vec2 ONNX モデルが見つかりません。\n\n"
            "【原因】\n"
            f"  ローカルパス ({self.model_path}) にモデルが存在せず、\n"
            f"  Hugging Face Hub ({repo_id}) からの自動ダウンロードも完了できませんでした。{download_err_info}\n\n"
            "【使い方の確認・解決手順】\n"
            "  1. [インターネット接続]\n"
            "     初回実行時は Hugging Face Hub より自動的にモデルがダウンロードされます。\n"
            "     ネットワーク接続を確認の上、再度実行してください。\n"
            "  2. [Hugging Face 認証トークン]\n"
            "     アクセス制限やレートリミットを回避する場合は環境変数を設定してください:\n"
            "     - Windows (PowerShell): $env:VOICERECOGNIZER_HF_TOKEN = \"your_token\"\n"
            "     - Linux / macOS (Bash): export VOICERECOGNIZER_HF_TOKEN=\"your_token\"\n"
            "     - または環境変数 HF_TOKEN (フォールバック) を設定\n"
            "  3. [ローカルモデルの指定]\n"
            "     ローカルに既にあるモデルフォルダを使用したい場合は、初期化時に model_path を渡してください:\n"
            "     >>> import voicerecognizer as vr\n"
            "     >>> recognizer = vr.Wav2Vec2Recognizer(model_path=\"/path/to/your/model_dir\")\n"
        )

    def _prepare_input_values(self, audio: Any) -> tuple[np.ndarray, float, float]:
        """波形前処理を行い ONNX に渡す input_values とタイミング計測値を返します。"""
        t_prep_start = time.perf_counter()
        waveform = self.audio_preprocessor.preprocess_waveform(
            audio,
            pad_to_target=not self.dynamic_trimming,
        )
        if self.input_name == "waveform":
            # 前処理内包型 ONNX: 特徴量抽出処理を行わずダイレクトに生波形を渡す
            if waveform.ndim == 1:
                input_values = np.expand_dims(waveform, axis=0).astype(np.float32)
            else:
                input_values = waveform.astype(np.float32)
        else:
            if self.feature_extractor is None:
                logger.error("Wav2Vec2 FeatureExtractor が初期化されていません (None)。モデルパスを確認してください: %s", self.model_path)
                raise ModelNotFoundError(f"Wav2Vec2 FeatureExtractor がロードされていません: {self.model_path}")
            # 従来型 ONNX: FeatureExtractor を呼び出し
            inputs = self.feature_extractor(
                waveform,
                sampling_rate=self.sample_rate,
                return_tensors="np",
                padding=True,
            )
            input_values = inputs["input_values"].astype(np.float32)
        t_prep_end = time.perf_counter()
        return input_values, t_prep_start, t_prep_end

    def recognize(self, audio: Any) -> str:
        self._ensure_model_loaded()
        if self.session is None:
            logger.error("Wav2Vec2 ONNX セッション (session) がロードされていません (None)。モデルパスを確認してください: %s", self.model_path)
            raise ModelNotFoundError(f"Wav2Vec2 ONNX セッションがロードされていません: {self.model_path}")
        t_start = time.perf_counter()

        input_values, t_prep_start, t_prep_end = self._prepare_input_values(audio)

        t_inf_start = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: input_values})
        t_inf_end = time.perf_counter()

        logits = outputs[0][0]  # shape: (num_classes,)
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        predicted_index = int(np.argmax(probabilities))
        self.last_confidence = float(probabilities[predicted_index])

        prep_stats = getattr(self.audio_preprocessor, "last_stats", {})
        self.last_timing_stats = {
            "onset_ms": prep_stats.get("onset_ms", 0.0),
            "offset_ms": prep_stats.get("offset_ms", 0.0),
            "speech_duration_ms": prep_stats.get("speech_duration_ms", 0.0),
            "prep_latency_ms": (t_prep_end - t_prep_start) * 1000.0,
            "inference_latency_ms": (t_inf_end - t_inf_start) * 1000.0,
            "total_latency_ms": (t_inf_end - t_start) * 1000.0,
            "confidence": self.last_confidence,
        }

        logger.debug("Wav2Vec2 ONNX 確率: %s", probabilities)
        return self._label_for_index(predicted_index)

    def recognize_with_candidates(self, audio: Any, top_k: int = 3) -> list[tuple[str, float]]:
        """上位 top_k 個の認識候補ラベルと確信度スコアのリストを返します"""
        self._ensure_model_loaded()
        if self.session is None:
            logger.error("Wav2Vec2 ONNX セッション (session) がロードされていません (None)。モデルパスを確認してください: %s", self.model_path)
            raise ModelNotFoundError(f"Wav2Vec2 ONNX セッションがロードされていません: {self.model_path}")
        input_values, _, _ = self._prepare_input_values(audio)
        outputs = self.session.run(None, {self.input_name: input_values})
        logits = outputs[0][0]
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        top_indices = np.argsort(probabilities)[::-1][:top_k]
        candidates = [
            (self._label_for_index(int(idx)), float(probabilities[idx])) for idx in top_indices
        ]
        self.last_confidence = candidates[0][1] if candidates else 0.0
        return candidates

    def _ensure_model_loaded(self) -> None:
        if self.session is not None and (
            self.input_name == "waveform" or self.feature_extractor is not None
        ):
            return

        if self.onnx_model_path is None or not self.onnx_model_path.exists():
            raise ModelNotFoundError(self._build_model_not_found_message())

        try:
            import onnxruntime as ort
            from transformers import AutoFeatureExtractor
        except ImportError as exc:
            raise ImportError(
                "Wav2Vec2 ONNX サポートには 'onnxruntime' と 'transformers' が必要です。"
            ) from exc

        # ONNX Runtime セッション初期化 (CPU 最適化)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        cpu_threads = max(1, min(4, os.cpu_count() or 4))
        sess_options.intra_op_num_threads = cpu_threads
        sess_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.onnx_model_path),
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

        # Feature Extractor ロード (入力が input_values の場合のみ)
        if self.input_name != "waveform":
            try:
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_path)
            except Exception:
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                    DEFAULT_RECOGNITION_CONFIG.wav2vec2_pretrained_model_name
                )
        logger.info(
            "ONNX モデルを正常ロードしました (input_name=%s): %s",
            self.input_name,
            self.onnx_model_path,
        )

    def _label_for_index(self, index: int) -> str:
        if 0 <= index < len(self.labels):
            return self.labels[index]
        return str(index)
