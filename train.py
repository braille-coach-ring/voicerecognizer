"""
モデル学習のエントリーポイント
"""

from core import PipelineFactory


def main():
    """メイン処理"""
    pipeline = PipelineFactory.create_default_training_pipeline(
        root_dir="processed_dataset",
        sample_rate=16000,
        n_mels=64,
        batch_size=8,
        epochs=150,
        learning_rate=0.001,
        val_rate=0.2,
        target_acc=0.97,
        seed=42,
        best_model_path="best_model.pth",
        last_model_path="last_model.pth",
        loss_plot_path="loss.png",
        accuracy_plot_path="accuracy.png",
    )

    pipeline.setup()
    pipeline.run()


if __name__ == "__main__":
    main()
