import unittest
from typing import Any, override

from voicerecognizer.core.interfaces import RecognitionStrategy
from voicerecognizer.evaluation.dynamic_trimming_compare import (
    compare_dynamic_trimming_accuracy,
    format_dynamic_trimming_summary,
)
from voicerecognizer.evaluation.evaluator import EvaluationResult, compute_evaluation_result


class FakeRecognizer(RecognitionStrategy):
    def __init__(self, dynamic_trimming: bool) -> None:
        self.dynamic_trimming = dynamic_trimming

    @override
    def recognize(self, audio: Any) -> str:
        return "a"


class FakeEvaluator:
    def __init__(self, model: RecognitionStrategy) -> None:
        if not isinstance(model, FakeRecognizer):
            raise TypeError(f"FakeEvaluator expects FakeRecognizer, got {type(model).__name__}")
        self.model = model

    def evaluate(self) -> EvaluationResult:
        if self.model.dynamic_trimming:
            return compute_evaluation_result(
                ["a", "a", "e"],
                ["a", "e", "e"],
                labels=("a", "e"),
            )
        return compute_evaluation_result(
            ["a", "a", "e"],
            ["a", "a", "e"],
            labels=("a", "e"),
        )


class TestDynamicTrimmingCompare(unittest.TestCase):
    def test_compare_dynamic_trimming_accuracy_prefers_fixed_when_macro_f1_is_higher(self) -> None:
        comparison = compare_dynamic_trimming_accuracy(
            labels=("a", "e"),
            recognizer_factory=lambda dynamic: FakeRecognizer(dynamic),
            evaluator_factory=lambda recognizer: FakeEvaluator(recognizer),
        )

        self.assertEqual(comparison.fixed_padding.total_samples, 3)
        self.assertLess(comparison.dynamic_trimming.macro_f1, comparison.fixed_padding.macro_f1)
        self.assertEqual(comparison.recommended_mode, "fixed_padding")
        self.assertLess(comparison.deltas_dynamic_minus_fixed["macro_f1"], 0)
        self.assertIn("dynamic_trimming=False", format_dynamic_trimming_summary(comparison))


if __name__ == "__main__":
    unittest.main()
