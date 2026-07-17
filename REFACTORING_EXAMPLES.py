"""
リファクタリング後の使用例とカスタマイズ方法

このファイルでは、リファクタリングされたコードを様々なシナリオで使用する例を示します。
"""

# ============================================================================
# 例1: デフォルト実装を使用（最もシンプル）
# ============================================================================

from core import PipelineFactory


def example_default_usage():
    """デフォルト実装を使った基本的な使用例"""
    pipeline = PipelineFactory.create_default_pipeline()
    pipeline.setup()
    result = pipeline.run()
    print(f"認識結果: {result}")


# ============================================================================
# 例2: 設定をカスタマイズ
# ============================================================================


def example_custom_config():
    """設定値をカスタマイズする例"""
    pipeline = PipelineFactory.create_default_pipeline(
        top_db=25,  # 無音除去のしきい値を変更
        n_mels=128,  # メルフィルタ数を増加
        model_path="best_model.pth",
        audio_output_file="my_custom_audio.wav",
    )
    pipeline.setup()
    result = pipeline.run()


# ============================================================================
# 例3: 結果表示をカスタマイズ（ファイル出力など）
# ============================================================================

from core.interfaces import ResultPresenter
from core import (
    PipelineFactory,
    DefaultConfig,
    DefaultAudioRecorder,
    DefaultAudioPreprocessor,
    DefaultMelSpectrogramConverter,
    DefaultModelLoader,
    DefaultInferencer,
    DefaultAudioSaver,
    DefaultCountdownDisplay,
)
from core.pipeline import MicrophonePredictionPipeline


class FileResultPresenter(ResultPresenter):
    """結果をファイルに出力するカスタム実装"""

    def __init__(self, output_file: str = "results.txt"):
        self.output_file = output_file

    def present(self, predicted_label: str, probabilities: dict) -> None:
        """結果をファイルに保存"""
        with open(self.output_file, "a") as f:
            f.write(f"予測: {predicted_label}\n")
            for label, prob in probabilities.items():
                f.write(f"  {label}: {prob * 100:.2f}%\n")
            f.write("-" * 30 + "\n")

        # 同時にコンソールにも出力
        print(f"結果を {self.output_file} に保存しました")


def example_custom_presenter():
    """カスタム結果表示を使用する例"""
    config = DefaultConfig()
    recorder = DefaultAudioRecorder()
    preprocessor = DefaultAudioPreprocessor()
    mel_converter = DefaultMelSpectrogramConverter()
    model_loader = DefaultModelLoader()
    inferencer = DefaultInferencer(model_loader)
    audio_saver = DefaultAudioSaver()
    countdown_display = DefaultCountdownDisplay()

    # カスタム実装を注入
    custom_presenter = FileResultPresenter(output_file="prediction_results.txt")

    pipeline = PipelineFactory.create_custom_pipeline(
        config=config,
        recorder=recorder,
        preprocessor=preprocessor,
        mel_converter=mel_converter,
        model_loader=model_loader,
        inferencer=inferencer,
        audio_saver=audio_saver,
        result_presenter=custom_presenter,  # ← カスタム実装
        countdown_display=countdown_display,
    )

    pipeline.setup()
    result = pipeline.run()


# ============================================================================
# 例4: 異なる前処理アルゴリズムを使用
# ============================================================================

import numpy as np
import librosa
from core.interfaces import AudioPreprocessor


class AdvancedAudioPreprocessor(AudioPreprocessor):
    """より高度な前処理を行うカスタム実装"""

    def preprocess(
        self,
        audio: np.ndarray,
        sample_rate: int,
        target_length: float,
        top_db: int,
    ) -> np.ndarray:
        # 無音除去
        y, _ = librosa.effects.trim(audio, top_db=top_db)

        # 高周波成分の強調（プリエンファシス）
        y = librosa.effects.preemphasis(y)

        # 音量正規化
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        # ノイズ低減
        # （より高度なノイズ低減アルゴリズムをここに追加可能）

        # 長さ統一
        target_samples = int(target_length * sample_rate)
        if len(y) > target_samples:
            y = y[:target_samples]
        else:
            y = np.pad(y, (0, target_samples - len(y)))

        return y


def example_advanced_preprocessing():
    """高度な前処理を使用する例"""
    config = DefaultConfig()
    recorder = DefaultAudioRecorder()
    preprocessor = AdvancedAudioPreprocessor()  # ← カスタム実装
    mel_converter = DefaultMelSpectrogramConverter()
    model_loader = DefaultModelLoader()
    inferencer = DefaultInferencer(model_loader)
    audio_saver = DefaultAudioSaver()
    countdown_display = DefaultCountdownDisplay()

    from core.interfaces import ResultPresenter

    class DefaultResultPresenter(ResultPresenter):
        def present(self, predicted_label: str, probabilities: dict) -> None:
            print(f"予測: {predicted_label}")
            for label, prob in probabilities.items():
                print(f"{label}: {prob * 100:.2f}%")

    result_presenter = DefaultResultPresenter()

    pipeline = PipelineFactory.create_custom_pipeline(
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

    pipeline.setup()
    result = pipeline.run()


# ============================================================================
# 例5: テスト用のモック実装
# ============================================================================

import torch


class MockAudioRecorder(AudioRecorder):
    """テスト用のモック音声レコーダー"""

    def __init__(self, test_audio: np.ndarray):
        self.test_audio = test_audio

    def record(self, duration: float, sample_rate: int) -> np.ndarray:
        """テスト用の固定音声を返す"""
        return self.test_audio


class MockResultPresenter(ResultPresenter):
    """テスト用のモック結果表示"""

    def __init__(self):
        self.results = []

    def present(self, predicted_label: str, probabilities: dict) -> None:
        """結果をメモリに保存（出力しない）"""
        self.results.append({"label": predicted_label, "probs": probabilities})


def example_testing():
    """テスト用のモック実装を使用する例"""
    # テスト用の固定音声
    test_audio = np.random.normal(0, 0.1, 16000)

    config = DefaultConfig()
    recorder = MockAudioRecorder(test_audio)  # ← モック実装
    preprocessor = DefaultAudioPreprocessor()
    mel_converter = DefaultMelSpectrogramConverter()
    model_loader = DefaultModelLoader()
    inferencer = DefaultInferencer(model_loader)
    audio_saver = DefaultAudioSaver()
    countdown_display = DefaultCountdownDisplay()
    result_presenter = MockResultPresenter()  # ← モック実装

    pipeline = PipelineFactory.create_custom_pipeline(
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

    pipeline.setup()
    result = pipeline.run()

    # テスト結果を確認
    print(f"テスト結果: {result_presenter.results}")


# ============================================================================
# リファクタリングの利点
# ============================================================================

"""
1. 保守性の向上
   - 各責務が明確に分離されている
   - 各クラスは単一の責務のみを持つ（SRP）

2. 拡張性の向上
   - 新しいアルゴリズムを追加する際は、インターフェースを実装するだけ
   - 既存コードへの影響がない

3. テスト容易性の向上
   - モック実装を作成して、ユニットテストが可能
   - 各コンポーネントを独立してテストできる

4. 将来の変更に強い
   - Strategy Patternでアルゴリズムを動的に切り替え可能
   - 新しい出力形式、入力ソースの追加が容易

5. 再利用性の向上
   - インターフェースに依存しているため、他のプロジェクトでも再利用可能
   - 異なる実装を簡単に組み合わせられる

6. 依存性の管理
   - Dependency Injectionで全ての依存性を明示的に管理
   - 循環依存や隠れた依存関係がない
"""


if __name__ == "__main__":
    # 実行例
    print("例1: デフォルト実装")
    # example_default_usage()

    print("\n例5: テスト用モック")
    # example_testing()
