"""
マイク入力からリアルタイム音声認識を実行

リファクタリング後: エントリーポイントのみ
処理は全て core パッケージで管理
"""

from core import PipelineFactory


def main():
    """メイン処理"""
    # パイプラインの作成（デフォルト実装で自動組み立て）
    pipeline = PipelineFactory.create_default_pipeline(
        sample_rate=16000,
        record_seconds=1.0,
        target_length=1.0,
        top_db=30,
        n_mels=64,
        labels=sorted(["a", "e", "i", "o", "u"]),
        model_path="best_model.pth",
        audio_output_file="predicted_audio.wav",
    )

    # 初期化
    pipeline.setup()

    # 推論実行
    pipeline.run()


if __name__ == "__main__":
    main()
