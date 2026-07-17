"""
voicelibrary.core パッケージ

リファクタリングされた音声認識パイプラインの実装
"""

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
from core.implementations import (
    DefaultConfig,
    DefaultAudioRecorder,
    DefaultAudioPreprocessor,
    DefaultMelSpectrogramConverter,
    DefaultModelLoader,
    DefaultInferencer,
    DefaultAudioSaver,
    DefaultResultPresenter,
    DefaultCountdownDisplay,
)
from core.pipeline import (
    DatasetEvaluationPipeline,
    MicrophonePredictionPipeline,
    TrainingPipeline,
)
from core.factory import PipelineFactory

__all__ = [
    # インターフェース
    "ConfigProvider",
    "AudioRecorder",
    "AudioPreprocessor",
    "MelSpectrogramConverter",
    "ModelLoader",
    "Inferencer",
    "AudioSaver",
    "ResultPresenter",
    "CountdownDisplay",
    # デフォルト実装
    "DefaultConfig",
    "DefaultAudioRecorder",
    "DefaultAudioPreprocessor",
    "DefaultMelSpectrogramConverter",
    "DefaultModelLoader",
    "DefaultInferencer",
    "DefaultAudioSaver",
    "DefaultResultPresenter",
    "DefaultCountdownDisplay",
    # パイプライン
    "MicrophonePredictionPipeline",
    "DatasetEvaluationPipeline",
    "TrainingPipeline",
    # ファクトリ
    "PipelineFactory",
]
