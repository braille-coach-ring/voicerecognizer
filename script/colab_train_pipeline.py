"""
Colab Self-Contained Training & Evaluation Pipeline for VoiceRecognizer
Executed directly on Colab GPU environment.
"""

import csv
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

# ヘッドレスColab環境でPortAudioライブラリ未検出によるインポートエラーを完全防護
try:
    import sounddevice
except Exception:
    sys.modules["sounddevice"] = MagicMock()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("colab_pipeline")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def step1_prepare_dataset():
    logger.info("=== [Step 1] Preparing merged_dataset and processed_dataset ===")
    splits_dir = PROJECT_ROOT / "data_splits"
    train_csv = splits_dir / "combined_train.csv"
    val_csv = splits_dir / "combined_val.csv"

    merged_dir = PROJECT_ROOT / "merged_dataset"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_index = merged_dir / "index.csv"

    rows = []
    for csv_path in [train_csv, val_csv]:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows.extend(list(reader))

    logger.info("Total samples for training + validation (Test 156 excluded): %d", len(rows))
    with open(merged_index, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "predicted_text"])
        writer.writeheader()
        writer.writerows(rows)

    from voicerecognizer.preprocessing.dataset_builder import DatasetBuilder

    builder = DatasetBuilder()
    builder.preprocess_dataset(input_root=merged_dir, output_root=PROJECT_ROOT / "processed_dataset")
    logger.info("Preprocessing complete: processed_dataset/ is ready.")


def step2_train_wav2vec2(epochs: int = 5, batch_size: int = 8, lr: float = 3e-5):
    logger.info("=== [Step 2] Training Wav2Vec2 on Colab GPU ===")
    from voicerecognizer.models.wav2vec2.train import build_parser, train

    parser = build_parser()
    args = parser.parse_args([
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--learning-rate", str(lr),
        "--patience", "3",
        "--skip-prep",
        "--no-hf-upload",
    ])
    train(args)
    logger.info("Wav2Vec2 GPU training complete.")


def step3_evaluate_unseen_test():
    logger.info("=== [Step 3] Evaluating Unseen Speakerphone Test Set (156 samples) ===")
    from voicerecognizer.core.factory.recognizer_factory import RecognizerFactory
    from voicerecognizer.evaluation.evaluator import Evaluator

    test_eval_dir = PROJECT_ROOT / "data_splits" / "speakerphone_test_eval"
    recognizer = RecognizerFactory.create("wav2vec2", use_last=False)
    evaluator = Evaluator(
        model=recognizer,
        dataset_path=test_eval_dir,
    )
    result = evaluator.evaluate()

    logger.info("--- Speakerphone Unseen Test Set Final Results ---")
    logger.info("Accuracy    : %.4f (Before was: 0.6154)", result.overall.accuracy)
    logger.info("Macro F1    : %.4f (Before was: 0.5638)", result.overall.macro_f1)
    logger.info("Weighted F1 : %.4f (Before was: 0.5630)", result.overall.weighted_f1)
    logger.info("Total       : %d samples", result.overall.total_samples)

    results_dir = PROJECT_ROOT / "evaluation_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "speakerphone_test_wav2vec2_colab_after.json"
    html_path = results_dir / "speakerphone_test_wav2vec2_colab_after.html"

    evaluator.export_json(json_path)
    evaluator.export_html(html_path, title="Wav2Vec2 Speakerphone Adaptation Final Report (Colab GPU)")
    logger.info("Evaluation report saved to: %s", json_path)

    # Print JSON output directly to stdout for immediate display
    print("\n" + "=" * 50 + " EVALUATION JSON OUTPUT " + "=" * 50)
    with open(json_path, encoding="utf-8") as f:
        print(f.read())
    print("=" * 124 + "\n")


def step4_upload_to_hf():
    logger.info("=== [Step 4] Uploading to Hugging Face Hub ===")
    from voicerecognizer.utils.model_uploader import upload_weights_to_hf

    success = upload_weights_to_hf(model_type="wav2vec2")
    if success:
        logger.info("Successfully uploaded adapted Wav2Vec2 weights to Hugging Face Hub.")
    else:
        logger.warning("Upload to Hugging Face Hub returned False.")


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    lr = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-5

    t0 = time.time()
    step1_prepare_dataset()
    step2_train_wav2vec2(epochs=epochs, batch_size=batch_size, lr=lr)
    step3_evaluate_unseen_test()
    step4_upload_to_hf()
    elapsed = time.time() - t0
    logger.info("=== All Pipeline Steps Completed Successfully in %.1f seconds ===", elapsed)


if __name__ == "__main__":
    main()
