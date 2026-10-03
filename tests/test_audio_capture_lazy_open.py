"""AudioCapture がコンストラクタでマイクを掴まないことのテスト。

音声ファイルを 1 件認識するだけの実行 (main.py sample.wav) や、マイクを持たない
CI・ヘッドレス環境でも AudioPipeline は構築される。そこで入力ストリームを
開いてしまうと不要にデバイスを占有する。
"""

import unittest
from unittest.mock import MagicMock, patch

from voicerecognizer.runtime.audio_capture import AudioCapture


class TestAudioCaptureLazyOpen(unittest.TestCase):
    def test_constructor_does_not_open_stream(self) -> None:
        with patch("voicerecognizer.runtime.audio_capture.sd.InputStream") as mock_stream:
            AudioCapture()
            mock_stream.assert_not_called()

    def test_capture_once_opens_stream_on_demand(self) -> None:
        with (
            patch("voicerecognizer.runtime.audio_capture.sd.InputStream") as mock_stream,
            patch("voicerecognizer.runtime.audio_capture.sd.sleep") as mock_sleep,
        ):
            instance = MagicMock()
            instance.active = True
            mock_stream.return_value = instance

            capture = AudioCapture()
            mock_stream.assert_not_called()

            audio = capture.capture_once()

            mock_stream.assert_called_once()
            instance.start.assert_called_once()
            # リングバッファが 1 窓分埋まるまで待つこと
            mock_sleep.assert_called_once()
            self.assertGreaterEqual(mock_sleep.call_args[0][0], int(capture.window_seconds * 1000))
            self.assertEqual(len(audio), int(capture.window_seconds * capture.sample_rate))

    def test_stop_is_safe_when_stream_was_never_opened(self) -> None:
        with patch("voicerecognizer.runtime.audio_capture.sd.InputStream"):
            capture = AudioCapture()
            capture.stop()  # 例外にならないこと


if __name__ == "__main__":
    unittest.main()
