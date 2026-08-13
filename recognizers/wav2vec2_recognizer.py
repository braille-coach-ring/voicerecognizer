import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from config import DEFAULT_RECOGNITION_CONFIG
from core.exceptions import ModelNotFoundError
from core.interfaces import RecognitionStrategy
from preprocessing.audio_preprocessor import AudioPreprocessor

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
    ):
        self.model_path = Path(model_path)
        self.labels = list(labels)
        self.dynamic_trimming = dynamic_trimming

        # ONNX モデルファイルの優先順位 (model_mel_int8.onnx > model_mel_fp32.onnx > model_int8.onnx > model_fp32.onnx > model.onnx)
        self.onnx_model_path: Path | None = None
        for candidate_name in (
            "model_mel_int8.onnx",
            "model_mel_fp32.onnx",
            "model_int8.onnx",
            "model_fp32.onnx",
            "model.onnx",
        ):
            candidate = self.model_path / candidate_name
            if candidate.exists():
                self.onnx_model_path = candidate
                break

        # labels.json のロード
        labels_json = self.model_path / "labels.json"
        if labels_json.exists():
            try:
                with open(labels_json, "r", encoding="utf-8") as f:
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
            raise ModelNotFoundError(
                f"Wav2Vec2 ONNX モデルが見つかりません: {self.model_path}\n"
                "以下のエクスポートコマンドを実行して ONNX ファイルを生成してください:\n"
                "  uv run python models/wav2vec2/export_mel_prepended_onnx.py"
            )

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
