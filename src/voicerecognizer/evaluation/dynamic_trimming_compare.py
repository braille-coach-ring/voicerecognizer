from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from voicerecognizer.config import DEFAULT_RECOGNITION_CONFIG
from voicerecognizer.core.interfaces import RecognitionStrategy
from voicerecognizer.evaluation.evaluator import EvaluationResult, Evaluator
from voicerecognizer.recognizers.wav2vec2_recognizer import Wav2Vec2Recognizer


@dataclass(frozen=True)
class DynamicTrimmingModeMetrics:
    dynamic_trimming: bool
    accuracy: float
    macro_f1: float
    weighted_f1: float
    total_samples: int


@dataclass(frozen=True)
class DynamicTrimmingComparison:
    fixed_padding: DynamicTrimmingModeMetrics
    dynamic_trimming: DynamicTrimmingModeMetrics
    deltas_dynamic_minus_fixed: dict[str, float]
    recommended_mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


RecognizerFactory = Callable[[bool], RecognitionStrategy]
EvaluatorFactory = Callable[[RecognitionStrategy], Any]


def _metrics_from_result(
    *,
    dynamic_trimming: bool,
    result: EvaluationResult,
) -> DynamicTrimmingModeMetrics:
    return DynamicTrimmingModeMetrics(
        dynamic_trimming=dynamic_trimming,
        accuracy=result.overall.accuracy,
        macro_f1=result.overall.macro_f1,
        weighted_f1=result.overall.weighted_f1,
        total_samples=result.overall.total_samples,
    )


def compare_dynamic_trimming_accuracy(
    *,
    model_path: Path | str = DEFAULT_RECOGNITION_CONFIG.wav2vec2_best_model_dir,
    dataset_dir: Path | str = DEFAULT_RECOGNITION_CONFIG.merged_dataset_dir,
    review_decisions_path: Path | str | None = None,
    labels: Sequence[str] = DEFAULT_RECOGNITION_CONFIG.labels,
    recognizer_factory: RecognizerFactory | None = None,
    evaluator_factory: EvaluatorFactory | None = None,
) -> DynamicTrimmingComparison:
    model_path = Path(model_path)
    dataset_dir = Path(dataset_dir)

    results: dict[bool, DynamicTrimmingModeMetrics] = {}
    for dynamic_trimming in (False, True):
        recognizer = (
            recognizer_factory(dynamic_trimming)
            if recognizer_factory is not None
            else Wav2Vec2Recognizer(
                model_path=model_path,
                dynamic_trimming=dynamic_trimming,
            )
        )
        evaluator = (
            evaluator_factory(recognizer)
            if evaluator_factory is not None
            else Evaluator(
                model=recognizer,
                labels=labels,
                dataset_path=dataset_dir,
                review_decisions_path=review_decisions_path,
            )
        )
        result = evaluator.evaluate()
        results[dynamic_trimming] = _metrics_from_result(
            dynamic_trimming=dynamic_trimming,
            result=result,
        )

    fixed = results[False]
    dynamic = results[True]
    deltas = {
        "accuracy": round(dynamic.accuracy - fixed.accuracy, 4),
        "macro_f1": round(dynamic.macro_f1 - fixed.macro_f1, 4),
        "weighted_f1": round(dynamic.weighted_f1 - fixed.weighted_f1, 4),
    }
    recommended = "dynamic_trimming=True" if dynamic.macro_f1 >= fixed.macro_f1 else "fixed_padding"

    return DynamicTrimmingComparison(
        fixed_padding=fixed,
        dynamic_trimming=dynamic,
        deltas_dynamic_minus_fixed=deltas,
        recommended_mode=recommended,
    )


def write_dynamic_trimming_comparison_json(
    comparison: DynamicTrimmingComparison,
    output_path: Path | str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison.to_dict(), f, ensure_ascii=False, indent=2)


def format_dynamic_trimming_summary(comparison: DynamicTrimmingComparison) -> str:
    fixed = comparison.fixed_padding
    dynamic = comparison.dynamic_trimming
    delta = comparison.deltas_dynamic_minus_fixed
    return "\n".join(
        [
            "=== Dynamic Trimming Accuracy Comparison ===",
            "dynamic_trimming=False "
            f"accuracy={fixed.accuracy:.4f}, macro_f1={fixed.macro_f1:.4f}, "
            f"weighted_f1={fixed.weighted_f1:.4f}, total={fixed.total_samples}",
            "dynamic_trimming=True  "
            f"accuracy={dynamic.accuracy:.4f}, macro_f1={dynamic.macro_f1:.4f}, "
            f"weighted_f1={dynamic.weighted_f1:.4f}, total={dynamic.total_samples}",
            "delta(dynamic - fixed) "
            f"accuracy={delta['accuracy']:+.4f}, macro_f1={delta['macro_f1']:+.4f}, "
            f"weighted_f1={delta['weighted_f1']:+.4f}",
            f"recommended_mode={comparison.recommended_mode}",
        ]
    )
