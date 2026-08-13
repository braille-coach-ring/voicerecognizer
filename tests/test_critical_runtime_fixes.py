import unittest
from unittest.mock import MagicMock

import torch

from models.wav2vec2.train import initialize_unloaded_classification_head


class TestCriticalRuntimeFixes(unittest.TestCase):
    def test_initialize_unloaded_classification_head_callable_check(self) -> None:
        mock_model = MagicMock(spec=torch.nn.Module)
        mock_model.projector = torch.nn.Linear(2, 2)
        mock_model._init_weights = MagicMock()
        initialize_unloaded_classification_head(mock_model, ["projector"])
        mock_model._init_weights.assert_called_once_with(mock_model.projector)


if __name__ == "__main__":
    unittest.main()
