import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from voicerecognizer.utils.plot_saver import record_and_plot_cumulative_progress


class TestCumulativePlot(unittest.TestCase):
    def test_record_and_plot_cumulative_progress(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with patch("voicerecognizer.utils.plot_saver.PROJECT_ROOT", tmp_path):
                trend_path = record_and_plot_cumulative_progress(
                    model_name="cnn",
                    val_acc=0.85,
                    val_macro_f1=0.65,
                    epochs=10,
                    num_classes=105,
                    num_samples=400,
                )

                self.assertTrue(trend_path.exists())

                history_json = tmp_path / "plots" / "cnn" / "experiment_history.json"
                self.assertTrue(history_json.exists())


if __name__ == "__main__":
    unittest.main()
