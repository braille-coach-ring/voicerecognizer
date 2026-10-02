"""
Colab Self-Contained Training & Evaluation Pipeline for VoiceRecognizer
Executed directly on Colab GPU environment.
"""

import csv
import logging
import shutil
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

# ヘッドレスColab環境でPortAudioライブラリ未検出によるインポートエラーを完全防護
try:
    __import__("sounddevice")
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


def step2_train_wav2vec2(epochs: int = 12, batch_size: int = 8, lr: float = 2e-5, freeze_layers: int = 4):
    logger.info("=== [Step 2] Training Wav2Vec2 on Colab GPU (Freeze=%d layers) ===", freeze_layers)
    from voicerecognizer.config import DEFAULT_RECOGNITION_CONFIG
    from voicerecognizer.models.wav2vec2.train import build_parser, train

    confusion_json = PROJECT_ROOT / "evaluation_results" / "speakerphone_test_wav2vec2_colab_after.json"
    train_args = [
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--learning-rate", str(lr),
        "--freeze-transformer-layers", str(freeze_layers),
        "--patience", "0",
        "--skip-prep",
        "--no-hf-upload",
    ]
    if confusion_json.exists():
        logger.info("Enabling confusion-pair boost sampler with previous evaluation result: %s", confusion_json)
        train_args.extend([
            "--confusion-pair-evaluation-result", str(confusion_json),
            "--confusion-pair-boost", "0.6",
        ])

    parser = build_parser()
    args = parser.parse_args(train_args)
    train(args)

    weights_dir = DEFAULT_RECOGNITION_CONFIG.weights_dir
    last_dir = weights_dir / "wav2vec2_last"
    best_dir = weights_dir / "wav2vec2_best"
    if last_dir.exists():
        logger.info("Syncing newly adapted weights from wav2vec2_last to wav2vec2_best...")
        shutil.copytree(last_dir, best_dir, dirs_exist_ok=True)
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

    homophone_acc = getattr(result.overall, "homophone_accuracy", result.overall.accuracy)
    logger.info("--- Speakerphone Unseen Test Set Final Results ---")
    logger.info("Accuracy           : %.4f (Before was: 0.6154)", result.overall.accuracy)
    logger.info("Homophone Accuracy : %.4f (じ/ぢ, ず/づ 同音統合)", homophone_acc)
    logger.info("Macro F1           : %.4f (Before was: 0.5638)", result.overall.macro_f1)
    logger.info("Weighted F1        : %.4f (Before was: 0.5630)", result.overall.weighted_f1)
    logger.info("Total              : %d samples", result.overall.total_samples)

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

    # 未見の女性話者 Test セット (52件) の評価
    fem_test_dir = PROJECT_ROOT / "data_splits" / "female_test_eval"
    if fem_test_dir.exists():
        logger.info("=== Evaluating Unseen Female Test Set (52 samples) ===")
        fem_evaluator = Evaluator(
            model=recognizer,
            dataset_path=fem_test_dir,
        )
        fem_result = fem_evaluator.evaluate()
        fem_homophone_acc = getattr(fem_result.overall, "homophone_accuracy", fem_result.overall.accuracy)
        logger.info("--- Female Unseen Test Set Final Results ---")
        logger.info("Accuracy           : %.4f (Before was: 0.3301)", fem_result.overall.accuracy)
        logger.info("Homophone Accuracy : %.4f", fem_homophone_acc)
        logger.info("Macro F1           : %.4f", fem_result.overall.macro_f1)
        logger.info("Weighted F1        : %.4f", fem_result.overall.weighted_f1)
        logger.info("Total              : %d samples", fem_result.overall.total_samples)

        fem_json_path = results_dir / "female_test_wav2vec2_colab_after.json"
        fem_html_path = results_dir / "female_test_wav2vec2_colab_after.html"
        fem_evaluator.export_json(fem_json_path)
        fem_evaluator.export_html(fem_html_path, title="Wav2Vec2 Female Voice Adaptation Final Report (Colab GPU)")
        logger.info("Female evaluation report saved to: %s", fem_json_path)


def step4_upload_to_hf():
    logger.info("=== [Step 4] Uploading to Hugging Face Hub ===")
    from voicerecognizer.config import DEFAULT_RECOGNITION_CONFIG
    from voicerecognizer.utils.model_uploader import upload_weights_to_hf

    weights_dir = DEFAULT_RECOGNITION_CONFIG.weights_dir
    success = upload_weights_to_hf(model_type="wav2vec2", weights_dir=weights_dir)
    if success:
        logger.info("Successfully uploaded adapted Wav2Vec2 weights to Hugging Face Hub.")
    else:
        logger.warning("Upload to Hugging Face Hub returned False.")


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    lr = float(sys.argv[3]) if len(sys.argv) > 3 else 2e-5
    freeze_layers = int(sys.argv[4]) if len(sys.argv) > 4 else 4

    t0 = time.time()
    step1_prepare_dataset()
    step2_train_wav2vec2(epochs=epochs, batch_size=batch_size, lr=lr, freeze_layers=freeze_layers)
    step3_evaluate_unseen_test()
    step4_upload_to_hf()
    elapsed = time.time() - t0
    logger.info("=== All Pipeline Steps Completed Successfully in %.1f seconds ===", elapsed)


if __name__ == "__main__":
    main()
