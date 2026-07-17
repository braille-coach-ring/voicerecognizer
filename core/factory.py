"""
Dependency Injection コンテナ
全てのコンポーネントの組み立てを管理
"""

from pathlib import Path

import torch
import torch.nn as nn

from dataset import HiraganaDataset
from model import HiraganaCNN
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


class PipelineFactory:
    """推論パイプラインをDIで組み立てるファクトリ"""

    @staticmethod
    def create_default_pipeline(
        sample_rate: int = 16000,
        record_seconds: float = 1.0,
        target_length: float = 1.0,
        top_db: int = 30,
        n_mels: int = 64,
        labels: list = None,
        model_path: str = "best_model.pth",
        device: torch.device = None,
        audio_output_file: str = "predicted_audio.wav",
    ) -> MicrophonePredictionPipeline:
        """
        デフォルト実装でパイプラインを作成

        Args:
            sample_rate: サンプリングレート
            record_seconds: 録音時間
            target_length: 目標時間
            top_db: 無音除去しきい値
            n_mels: メルフィルタ数
            labels: クラスラベル
            model_path: モデルファイルパス
            device: GPU/CPUデバイス
            audio_output_file: 出力ファイル名

        Returns:
            MicrophonePredictionPipeline
        """
        # 設定
        config: ConfigProvider = DefaultConfig(
            sample_rate=sample_rate,
            record_seconds=record_seconds,
            target_length=target_length,
            top_db=top_db,
            n_mels=n_mels,
            labels=labels,
            model_path=model_path,
            device=device,
            audio_output_file=audio_output_file,
        )

        # UI層
        countdown_display: CountdownDisplay = DefaultCountdownDisplay()

        # 入出力層
        recorder: AudioRecorder = DefaultAudioRecorder()
        audio_saver: AudioSaver = DefaultAudioSaver()

        # 処理層
        preprocessor: AudioPreprocessor = DefaultAudioPreprocessor()
        mel_converter: MelSpectrogramConverter = DefaultMelSpectrogramConverter()

        # モデル層
        model_loader: ModelLoader = DefaultModelLoader()
        inferencer: Inferencer = DefaultInferencer(model_loader)

        # 出力層
        result_presenter: ResultPresenter = DefaultResultPresenter()

        # パイプラインの組み立て
        pipeline = MicrophonePredictionPipeline(
            config=config,
            recorder=recorder,
            preprocessor=preprocessor,
            mel_converter=mel_converter,
            model_loader=model_loader,
            inferencer=inferencer,
            audio_saver=audio_saver,
            result_presenter=result_presenter,
            countdown_display=countdown_display,
        )

        return pipeline

    @staticmethod
    def create_custom_pipeline(
        config: ConfigProvider,
        recorder: AudioRecorder,
        preprocessor: AudioPreprocessor,
        mel_converter: MelSpectrogramConverter,
        model_loader: ModelLoader,
        inferencer: Inferencer,
        audio_saver: AudioSaver,
        result_presenter: ResultPresenter,
        countdown_display: CountdownDisplay,
    ) -> MicrophonePredictionPipeline:
        """
        カスタム実装でパイプラインを作成

        Args:
            config: 設定プロバイダー
            recorder: 音声レコーダー
            preprocessor: 音声前処理器
            mel_converter: メルスペクトログラム変換器
            model_loader: モデルローダー
            inferencer: 推論エンジン
            audio_saver: 音声セーバー
            result_presenter: 結果表示器
            countdown_display: カウントダウン表示

        Returns:
            MicrophonePredictionPipeline
        """
        return MicrophonePredictionPipeline(
            config=config,
            recorder=recorder,
            preprocessor=preprocessor,
            mel_converter=mel_converter,
            model_loader=model_loader,
            inferencer=inferencer,
            audio_saver=audio_saver,
            result_presenter=result_presenter,
            countdown_display=countdown_display,
        )

    @staticmethod
    def create_default_evaluation_pipeline(
        root_dir: str = "processed_dataset",
        sample_rate: int = 16000,
        n_mels: int = 64,
        model_path: str = "best_model.pth",
        device: torch.device = None,
    ) -> DatasetEvaluationPipeline:
        """
        デフォルト実装で評価パイプラインを作成
        """
        dataset = HiraganaDataset(
            root_dir=root_dir,
            sample_rate=sample_rate,
            n_mels=n_mels,
        )

        config: ConfigProvider = DefaultConfig(
            sample_rate=sample_rate,
            n_mels=n_mels,
            labels=dataset.labels,
            model_path=model_path,
            device=device,
        )

        mel_converter: MelSpectrogramConverter = DefaultMelSpectrogramConverter()
        model_loader: ModelLoader = DefaultModelLoader()
        inferencer: Inferencer = DefaultInferencer(model_loader)

        return DatasetEvaluationPipeline(
            config=config,
            dataset_root=Path(root_dir),
            mel_converter=mel_converter,
            model_loader=model_loader,
            inferencer=inferencer,
        )

    @staticmethod
    def create_default_training_pipeline(
        root_dir: str = "processed_dataset",
        sample_rate: int = 16000,
        n_mels: int = 64,
        batch_size: int = 8,
        epochs: int = 150,
        learning_rate: float = 0.001,
        val_rate: float = 0.2,
        target_acc: float = 0.97,
        seed: int = 42,
        best_model_path: str = "best_model.pth",
        last_model_path: str = "last_model.pth",
        loss_plot_path: str = "loss.png",
        accuracy_plot_path: str = "accuracy.png",
        device: torch.device = None,
    ) -> TrainingPipeline:
        """
        デフォルト実装で学習パイプラインを作成
        """
        dataset = HiraganaDataset(
            root_dir=root_dir,
            sample_rate=sample_rate,
            n_mels=n_mels,
        )

        config: ConfigProvider = DefaultConfig(
            sample_rate=sample_rate,
            n_mels=n_mels,
            labels=dataset.labels,
            device=device,
        )

        return TrainingPipeline(
            config=config,
            dataset=dataset,
            model_factory=lambda num_classes: HiraganaCNN(num_classes=num_classes),
            criterion_factory=nn.CrossEntropyLoss,
            optimizer_factory=lambda parameters: torch.optim.Adam(
                parameters,
                lr=learning_rate,
            ),
            batch_size=batch_size,
            epochs=epochs,
            val_rate=val_rate,
            target_acc=target_acc,
            seed=seed,
            best_model_path=best_model_path,
            last_model_path=last_model_path,
            loss_plot_path=loss_plot_path,
            accuracy_plot_path=accuracy_plot_path,
        )
