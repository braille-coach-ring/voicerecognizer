"""
マイク推論パイプライン
Dependency Injectionパターンで実装
"""

import random
from pathlib import Path
from typing import Callable

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

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


class MicrophonePredictionPipeline:
    """マイク入力からの推論パイプライン"""

    def __init__(
        self,
        config: ConfigProvider,
        recorder: AudioRecorder,
        preprocessor: AudioPreprocessor,
        mel_converter: MelSpectrogramConverter,
        model_loader: ModelLoader,
        inferencer: Inferencer,
        audio_saver: AudioSaver,
        result_presenter: ResultPresenter,
        countdown_display: CountdownDisplay,
    ):
        """
        Dependency Injectionで全てのコンポーネントを注入

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
        """
        self._config = config
        self._recorder = recorder
        self._preprocessor = preprocessor
        self._mel_converter = mel_converter
        self._model_loader = model_loader
        self._inferencer = inferencer
        self._audio_saver = audio_saver
        self._result_presenter = result_presenter
        self._countdown_display = countdown_display

    def setup(self) -> None:
        """初期化処理"""
        num_classes = len(self._config.get_labels())
        self._model_loader.load_model(
            self._config.get_model_path(),
            num_classes,
            self._config.get_device(),
        )

    def run(self) -> str:
        """
        推論パイプラインを実行

        Returns:
            予測ラベル
        """
        # ステップ1: カウントダウン表示
        self._show_countdown()

        # ステップ2: 音声録音
        raw_audio = self._record_audio()

        # ステップ3: 音声前処理
        processed_audio = self._preprocess_audio(raw_audio)

        # ステップ4: 音声を保存
        self._save_audio(processed_audio)

        # ステップ5: メルスペクトログラム変換
        mel_spec = self._convert_to_mel_spectrogram(processed_audio)

        # ステップ6: 推論実行
        pred_idx, probs = self._run_inference(mel_spec)

        # ステップ7: 結果を表示
        predicted_label = self._config.get_labels()[pred_idx]
        self._present_results(predicted_label, probs)

        return predicted_label

    def _show_countdown(self) -> None:
        """カウントダウン表示"""
        self._countdown_display.show_message("3秒後に1秒間録音します...")
        self._countdown_display.show_countdown(3)
        self._countdown_display.show_message("発声してください。")

    def _record_audio(self):
        """音声を録音"""
        audio = self._recorder.record(
            self._config.get_record_seconds(),
            self._config.get_sample_rate(),
        )
        print("録音終了")
        return audio

    def _preprocess_audio(self, raw_audio):
        """音声を前処理"""
        processed_audio = self._preprocessor.preprocess(
            raw_audio,
            self._config.get_sample_rate(),
            self._config.get_target_length(),
            self._config.get_top_db(),
        )
        return processed_audio

    def _save_audio(self, audio):
        """音声をファイルに保存"""
        self._audio_saver.save(
            audio,
            self._config.get_sample_rate(),
            self._config.get_audio_output_file(),
        )

    def _convert_to_mel_spectrogram(self, audio):
        """メルスペクトログラムに変換"""
        mel_spec = self._mel_converter.convert(
            audio,
            self._config.get_sample_rate(),
            self._config.get_n_mels(),
        )
        return mel_spec

    def _run_inference(self, mel_spec):
        """推論を実行"""
        pred_idx, probs = self._inferencer.predict(
            mel_spec,
            self._config.get_device(),
        )
        return pred_idx, probs

    def _present_results(self, predicted_label: str, probs):
        """結果を表示"""
        labels = self._config.get_labels()
        probabilities = {label: prob.item() for label, prob in zip(labels, probs)}
        self._result_presenter.present(predicted_label, probabilities)

        print("\n" + "=" * 30)
        print(f"📁 {self._config.get_audio_output_file()} に保存")
        print("=" * 30)


class DatasetEvaluationPipeline:
    """データセット評価パイプライン"""

    def __init__(
        self,
        config: ConfigProvider,
        dataset_root: Path,
        mel_converter: MelSpectrogramConverter,
        model_loader: ModelLoader,
        inferencer: Inferencer,
    ):
        self._config = config
        self._dataset_root = dataset_root
        self._mel_converter = mel_converter
        self._model_loader = model_loader
        self._inferencer = inferencer

    def setup(self) -> None:
        """初期化処理"""
        self._model_loader.load_model(
            self._config.get_model_path(),
            len(self._config.get_labels()),
            self._config.get_device(),
        )

    def run(self) -> None:
        """評価パイプラインを実行"""
        labels = self._config.get_labels()
        total = 0
        correct = 0
        class_total = {label: 0 for label in labels}
        class_correct = {label: 0 for label in labels}

        print("=" * 50)

        for label in labels:
            folder = self._dataset_root / label

            for wav in sorted(folder.glob("*.wav")):
                pred_label, probs = self._evaluate_file(wav)
                ok = pred_label == label

                total += 1
                class_total[label] += 1

                if ok:
                    correct += 1
                    class_correct[label] += 1

                self._present_file_result(label, wav, pred_label, probs, ok)

        self._present_class_accuracy(labels, class_total, class_correct)
        self._present_total_accuracy(total, correct)

    def _evaluate_file(self, wav_path: Path):
        """1ファイルを評価"""
        audio, sample_rate = librosa.load(
            wav_path,
            sr=self._config.get_sample_rate(),
        )
        mel_spec = self._mel_converter.convert(
            audio,
            sample_rate,
            self._config.get_n_mels(),
        )
        pred, probs = self._inferencer.predict(
            mel_spec,
            self._config.get_device(),
        )
        pred_label = self._config.get_labels()[pred]

        return pred_label, probs

    def _present_file_result(
        self,
        label: str,
        wav_path: Path,
        pred_label: str,
        probs,
        ok: bool,
    ) -> None:
        """1ファイルの評価結果を表示"""
        labels = self._config.get_labels()
        mark = "○" if ok else "×"

        print("=" * 50)
        print(f"{label}/{wav_path.name}")
        print()
        print(f"予測 : {pred_label}")
        print(f"正解 : {label}")
        print()

        for i, c in enumerate(labels):
            print(f"{c} : {probs[i].item() * 100:.2f}%")

        print()
        print(f"{mark} {'正解' if ok else '不正解'}")
        print()
        print()
        print("=" * 50)
        print("文字ごとの正答率")
        print("=" * 50)

    def _present_class_accuracy(
        self,
        labels: list,
        class_total: dict,
        class_correct: dict,
    ) -> None:
        """文字ごとの正答率を表示"""
        for label in labels:
            acc = class_correct[label] / class_total[label] * 100
            print(f"{label} : {class_correct[label]}/{class_total[label]} ({acc:.2f}%)")

    def _present_total_accuracy(self, total: int, correct: int) -> None:
        """全体の正答率を表示"""
        print()
        print("=" * 50)
        print("全体")
        print("=" * 50)
        print(f"{correct}/{total}")
        print(f"Accuracy : {correct / total * 100:.2f}%")


class TrainingPipeline:
    """モデル学習パイプライン"""

    def __init__(
        self,
        config: ConfigProvider,
        dataset,
        model_factory: Callable[[int], torch.nn.Module],
        criterion_factory: Callable[[], torch.nn.Module],
        optimizer_factory: Callable,
        batch_size: int,
        epochs: int,
        val_rate: float,
        target_acc: float,
        seed: int,
        best_model_path: str,
        last_model_path: str,
        loss_plot_path: str,
        accuracy_plot_path: str,
    ):
        self._config = config
        self._dataset = dataset
        self._model_factory = model_factory
        self._criterion_factory = criterion_factory
        self._optimizer_factory = optimizer_factory
        self._batch_size = batch_size
        self._epochs = epochs
        self._val_rate = val_rate
        self._target_acc = target_acc
        self._seed = seed
        self._best_model_path = best_model_path
        self._last_model_path = last_model_path
        self._loss_plot_path = loss_plot_path
        self._accuracy_plot_path = accuracy_plot_path

        self._train_loader = None
        self._val_loader = None
        self._model = None
        self._criterion = None
        self._optimizer = None

        self._train_losses = []
        self._val_losses = []
        self._train_accs = []
        self._val_accs = []

    def setup(self) -> None:
        """初期化処理"""
        self._fix_seed()
        device = self._config.get_device()

        print("Device:", device)

        train_dataset, val_dataset = self._split_dataset()
        self._train_loader = DataLoader(
            train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
        )
        self._val_loader = DataLoader(
            val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
        )

        print("Train      :", len(train_dataset))
        print("Validation :", len(val_dataset))

        self._model = self._model_factory(len(self._dataset.labels))
        self._model.to(device)
        self._criterion = self._criterion_factory()
        self._optimizer = self._optimizer_factory(self._model.parameters())

    def run(self) -> None:
        """学習パイプラインを実行"""
        best_acc = 0.0

        for epoch in range(self._epochs):
            train_loss, train_acc = self._train_epoch(epoch)
            val_loss, val_acc = self._validate()

            self._record_history(train_loss, val_loss, train_acc, val_acc)
            self._present_epoch_result(train_loss, train_acc, val_loss, val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self._model.state_dict(), self._best_model_path)
                print("Best model saved!")

            if val_acc >= self._target_acc:
                print()
                print(f"Validation Accuracy {self._target_acc * 100:.0f}% 到達")
                break

        torch.save(self._model.state_dict(), self._last_model_path)
        print("\nModel Saved!")

        self._save_training_plots()
        print("\nTraining Finished!")

    def _fix_seed(self) -> None:
        """乱数を固定"""
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)

    def _split_dataset(self):
        """データセットを学習用と検証用に分割"""
        labels = [label for _, label in self._dataset.data]
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self._val_rate,
            random_state=self._seed,
        )
        train_idx, val_idx = next(splitter.split(range(len(labels)), labels))

        return Subset(self._dataset, train_idx), Subset(self._dataset, val_idx)

    def _train_epoch(self, epoch: int):
        """1 epoch 分の学習"""
        self._model.train()

        train_loss = 0
        train_correct = 0
        train_total = 0
        device = self._config.get_device()
        progress = tqdm(self._train_loader)

        for mel, label in progress:
            mel = mel.to(device)
            label = label.to(device)

            self._optimizer.zero_grad()

            output = self._model(mel)
            loss = self._criterion(output, label)

            loss.backward()
            self._optimizer.step()

            train_loss += loss.item()

            pred = output.argmax(dim=1)
            train_correct += (pred == label).sum().item()
            train_total += label.size(0)

            progress.set_description(f"Epoch {epoch + 1}/{self._epochs}")
            progress.set_postfix(loss=f"{loss.item():.3f}")

        train_loss /= len(self._train_loader)
        train_acc = train_correct / train_total

        return train_loss, train_acc

    def _validate(self):
        """検証データで評価"""
        self._model.eval()

        val_loss = 0
        val_correct = 0
        val_total = 0
        device = self._config.get_device()

        with torch.no_grad():
            for mel, label in self._val_loader:
                mel = mel.to(device)
                label = label.to(device)

                output = self._model(mel)
                loss = self._criterion(output, label)

                val_loss += loss.item()

                pred = output.argmax(dim=1)
                val_correct += (pred == label).sum().item()
                val_total += label.size(0)

        val_loss /= len(self._val_loader)
        val_acc = val_correct / val_total

        return val_loss, val_acc

    def _record_history(
        self,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
    ) -> None:
        """学習履歴を保存"""
        self._train_losses.append(train_loss)
        self._val_losses.append(val_loss)
        self._train_accs.append(train_acc)
        self._val_accs.append(val_acc)

    def _present_epoch_result(
        self,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
    ) -> None:
        """epoch ごとの結果を表示"""
        print()
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Val Acc    : {val_acc:.4f}")

    def _save_training_plots(self) -> None:
        """学習履歴のグラフを保存"""
        plt.figure(figsize=(8, 5))
        plt.plot(self._train_losses, label="Train")
        plt.plot(self._val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(self._loss_plot_path)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(self._train_accs, label="Train")
        plt.plot(self._val_accs, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(self._accuracy_plot_path)
        plt.close()
