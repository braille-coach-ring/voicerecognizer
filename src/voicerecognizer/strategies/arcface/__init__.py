"""Strategy 04: Wav2Vec2 + ArcFace Angular Margin Loss."""

from voicerecognizer.strategies.arcface.loss import ArcMarginProduct
from voicerecognizer.strategies.arcface.model import Wav2Vec2ForArcFaceClassification

__all__ = ["ArcMarginProduct", "Wav2Vec2ForArcFaceClassification"]
