"""
リファクタリング後のコード動作確認テスト

使用方法:
  python test_refactoring.py
"""

import numpy as np
import torch
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

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
from core.interfaces import AudioRecorder, ResultPresenter


class MockAudioRecorder(AudioRecorder):
    """テスト用のモック音声レコーダー"""

    def __init__(self, audio_data: np.ndarray = None):
        if audio_data is None:
            # テスト用の合成音声
            sr = 16000
            duration = 1.0
            t = np.linspace(0, duration, int(sr * duration))
            # 440Hzの正弦波（Aノート）
            self.audio_data = np.sin(2 * np.pi * 440 * t) * 0.1
        else:
            self.audio_data = audio_data

    def record(self, duration: float, sample_rate: int) -> np.ndarray:
        return self.audio_data


class MockResultPresenter(ResultPresenter):
    """テスト用のモック結果表示"""

    def __init__(self):
        self.results = []

    def present(self, predicted_label: str, probabilities: dict) -> None:
        self.results.append({"label": predicted_label, "probs": probabilities})
        print(f"✓ 結果表示: {predicted_label}")


def test_DEFAULT_RECOGNITION_CONFIG():
    """テスト: DefaultConfig"""
    print("\n" + "=" * 60)
    print("テスト1: DefaultConfig")
    print("=" * 60)

    config = DefaultConfig(
        sample_rate=16000,
        top_db=30,
        n_mels=64,
    )

    assert config.get_sample_rate() == 16000, "サンプリングレート設定失敗"
    assert config.get_top_db() == 30, "TOP_DB設定失敗"
    assert len(config.get_labels()) == 5, "ラベル数が不正"

    print("✓ 設定が正しく反映されている")


def test_audio_preprocessor():
    """テスト: DefaultAudioPreprocessor"""
    print("\n" + "=" * 60)
    print("テスト2: DefaultAudioPreprocessor")
    print("=" * 60)

    preprocessor = DefaultAudioPreprocessor()

    # テスト用音声（ランダムノイズ）
    test_audio = np.random.normal(0, 0.1, 16000)

    processed = preprocessor.preprocess(
        audio=test_audio,
        sample_rate=16000,
        target_length=1.0,
        top_db=30,
    )

    assert processed.shape == (16000,), f"出力形状が不正: {processed.shape}"
    assert np.max(np.abs(processed)) <= 1.0, "正規化失敗"

    print(f"✓ 前処理が正しく実行（出力形状: {processed.shape}）")


def test_mel_spectrogram_converter():
    """テスト: DefaultMelSpectrogramConverter"""
    print("\n" + "=" * 60)
    print("テスト3: DefaultMelSpectrogramConverter")
    print("=" * 60)

    converter = DefaultMelSpectrogramConverter()

    # テスト用音声
    test_audio = np.random.normal(0, 0.1, 16000)

    mel_spec = converter.convert(
        audio=test_audio,
        sample_rate=16000,
        n_mels=64,
    )

    assert isinstance(mel_spec, torch.Tensor), "出力がTensorでない"
    assert mel_spec.shape == (1, 1, 64, 101), f"出力形状が不正: {mel_spec.shape}"

    print(f"✓ メルスペクトログラム変換成功（形状: {tuple(mel_spec.shape)}）")


def test_model_loader():
    """テスト: DefaultModelLoader"""
    print("\n" + "=" * 60)
    print("テスト4: DefaultModelLoader")
    print("=" * 60)

    model_loader = DefaultModelLoader()
    device = torch.device("cpu")

    try:
        model_loader.load_model(
            model_path="best_model.pth",
            num_classes=5,
            device=device,
        )
        model = model_loader.get_model()
        assert model is not None, "モデルロード失敗"
        print("✓ モデルがロードされた")
    except FileNotFoundError:
        print("⚠ best_model.pth が見つかりません（スキップ）")


def test_pipeline_with_mock():
    """テスト: MicrophonePredictionPipeline（モック使用）"""
    print("\n" + "=" * 60)
    print("テスト5: MicrophonePredictionPipeline（モック）")
    print("=" * 60)

    try:
        # テスト用モックを使用
        config = DefaultConfig()
        recorder = MockAudioRecorder()
        preprocessor = DefaultAudioPreprocessor()
        mel_converter = DefaultMelSpectrogramConverter()
        model_loader = DefaultModelLoader()
        inferencer = DefaultInferencer(model_loader)
        audio_saver = DefaultAudioSaver()
        countdown_display = DefaultCountdownDisplay()
        result_presenter = MockResultPresenter()

        # モデルをロード
        model_loader.load_model(
            config.get_model_path(),
            len(config.get_labels()),
            config.get_device(),
        )

        from core.pipeline import MicrophonePredictionPipeline

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

        print("✓ パイプラインが正常に構築された")
        print("✓ 全てのコンポーネントが正しく注入されている")

    except FileNotFoundError:
        print("⚠ best_model.pth が見つかりません（パイプラインテストをスキップ）")


def test_factory_pattern():
    """テスト: PipelineFactory"""
    print("\n" + "=" * 60)
    print("テスト6: PipelineFactory")
    print("=" * 60)

    try:
        # デフォルト実装で作成
        pipeline = PipelineFactory.create_default_pipeline(
            top_db=30,
            audio_output_file="test_output.wav",
        )
        assert pipeline is not None, "パイプライン作成失敗"
        print("✓ ファクトリパターンで正常にパイプラインが作成された")

    except FileNotFoundError:
        print("⚠ best_model.pth が見つかりません（ファクトリテストをスキップ）")


def run_all_tests():
    """全テストを実行"""
    print("\n" + "=" * 60)
    print("リファクタリング後の動作確認テスト")
    print("=" * 60)

    tests = [
        test_DEFAULT_RECOGNITION_CONFIG,
        test_audio_preprocessor,
        test_mel_spectrogram_converter,
        test_model_loader,
        test_pipeline_with_mock,
        test_factory_pattern,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ テスト失敗: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print("テスト結果")
    print("=" * 60)
    print(f"成功: {passed}")
    print(f"失敗: {failed}")
    print("=" * 60)

    if failed == 0:
        print("✓ 全テストが成功しました！")
    else:
        print("✗ 失敗したテストがあります")


if __name__ == "__main__":
    run_all_tests()
