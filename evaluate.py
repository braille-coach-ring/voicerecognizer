"""
学習済みモデルを processed_dataset で評価するエントリーポイント
"""

from core import PipelineFactory


def main():
    """メイン処理"""
    pipeline = PipelineFactory.create_default_evaluation_pipeline(
        root_dir="processed_dataset",
        sample_rate=16000,
        n_mels=64,
        model_path="best_model.pth",
    )

    pipeline.setup()
    pipeline.run()


if __name__ == "__main__":
    main()
