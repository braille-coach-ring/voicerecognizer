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
    logger.info("=== [Step 1] Preparing merged_dataset and processed_dataset (Speaker-Independent) ===")
    splits_dir = PROJECT_ROOT / "data_splits" / "speaker_independent"
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"

    if not train_csv.exists() or not val_csv.exists():
        logger.info("Speaker-independent splits not found, generating now...")
        import subprocess
        subprocess.run([sys.executable, str(PROJECT_ROOT / "script" / "create_speaker_independent_splits.py")], check=True)

    merged_dir = PROJECT_ROOT / "merged_dataset"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_index = merged_dir / "index.csv"

    rows = []
    for csv_path in [train_csv, val_csv]:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows.extend(list(reader))

    logger.info("Total samples for training + validation (Test 620 excluded): %d", len(rows))
    with open(merged_index, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "speaker", "predicted_text"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    from voicerecognizer.preprocessing.dataset_builder import DatasetBuilder

    builder = DatasetBuilder()
    builder.preprocess_dataset(input_root=merged_dir, output_root=PROJECT_ROOT / "processed_dataset")
    logger.info("Preprocessing complete: processed_dataset/ is ready.")


def step2_train_wav2vec2(
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 3e-5,
    freeze_layers: int = 4,
    from_scratch: bool = True,
):
    logger.info(
        "=== [Step 2] Training Wav2Vec2 on Colab GPU (Freeze=%d layers, From-Scratch=%s) ===",
        freeze_layers,
        from_scratch,
    )
    from voicerecognizer.config import DEFAULT_RECOGNITION_CONFIG
    from voicerecognizer.models.wav2vec2.train import build_parser, train

    train_args = [
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--learning-rate", str(lr),
        "--freeze-transformer-layers", str(freeze_layers),
        "--patience", "5",
        "--skip-prep",
        "--no-hf-upload",
        "--no-confusion-pair-sampler",
        "--normalize-homophones",
        "--train-csv", str(PROJECT_ROOT / "data_splits/speaker_independent/train.csv"),
        "--val-csv", str(PROJECT_ROOT / "data_splits/speaker_independent/val.csv"),
    ]
    if from_scratch:
        train_args.append("--no-resume")

    parser = build_parser()
    args = parser.parse_args(train_args)
    train(args)

    weights_dir = DEFAULT_RECOGNITION_CONFIG.weights_dir
    last_dir = weights_dir / "wav2vec2_last"
    best_dir = weights_dir / "wav2vec2_best"
    if last_dir.exists():
        logger.info("Syncing newly trained weights from wav2vec2_last to wav2vec2_best...")
        shutil.copytree(last_dir, best_dir, dirs_exist_ok=True)
    logger.info("Wav2Vec2 GPU training complete.")


def step3_evaluate_unseen_test():
    logger.info("=== [Step 3] Evaluating Unseen Speaker-Independent Test Set (620 samples) ===")
    from voicerecognizer.core.factory.recognizer_factory import RecognizerFactory
    from voicerecognizer.evaluation.evaluator import Evaluator

    test_eval_dir = PROJECT_ROOT / "data_splits" / "speaker_independent" / "test_eval"
    recognizer = RecognizerFactory.create("wav2vec2", use_last=False)
    evaluator = Evaluator(
        model=recognizer,
        dataset_path=test_eval_dir,
    )
    result = evaluator.evaluate()

    homophone_acc = getattr(result.overall, "homophone_accuracy", result.overall.accuracy)
    logger.info("--- Speaker-Independent Unseen Test Set Final Results ---")
    logger.info("Accuracy           : %.4f (Baseline was: 0.5210)", result.overall.accuracy)
    logger.info("Homophone Accuracy : %.4f (じ/ぢ, ず/づ 同音統合)", homophone_acc)
    logger.info("Macro F1           : %.4f (Baseline was: 0.4128)", result.overall.macro_f1)
    logger.info("Weighted F1        : %.4f (Baseline was: 0.4974)", result.overall.weighted_f1)
    logger.info("Total              : %d samples", result.overall.total_samples)

    if hasattr(result, "speaker_metrics") and result.speaker_metrics:
        logger.info("--- Speaker Breakdown ---")
        for sp, m in result.speaker_metrics.items():
            logger.info("  %10s: Accuracy=%.4f (%d/%d)", sp, m.accuracy, m.correct_samples, m.total_samples)

    results_dir = PROJECT_ROOT / "evaluation_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "speaker_independent_test_after_colab.json"
    html_path = results_dir / "speaker_independent_test_after_colab.html"

    evaluator.export_json(json_path)
    evaluator.export_html(html_path, title="Wav2Vec2 Speaker-Independent Training Report (Colab GPU)")
    logger.info("Evaluation report saved to: %s", json_path)

    print("\n" + "=" * 50 + " EVALUATION JSON OUTPUT " + "=" * 50)
    with open(json_path, encoding="utf-8") as f:
        print(f.read())
    print("=" * 124 + "\n")


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
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    lr = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-5
    freeze_layers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    from_scratch = sys.argv[5].lower() in ("true", "1", "yes") if len(sys.argv) > 5 else True

    t0 = time.time()
    step1_prepare_dataset()
    step2_train_wav2vec2(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        freeze_layers=freeze_layers,
        from_scratch=from_scratch,
    )
    step3_evaluate_unseen_test()
    if "--upload-to-hf" in sys.argv:
        step4_upload_to_hf()
    elapsed = time.time() - t0
    logger.info("=== All Pipeline Steps Completed Successfully in %.1f seconds ===", elapsed)


if __name__ == "__main__":
    main()
