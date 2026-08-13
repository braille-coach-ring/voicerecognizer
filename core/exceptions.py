"""
Voice Recognizer Domain Exceptions

無効な設定、マイク未接続、モデル未存在などの問題が発生した際に、
サイレントに無視せず明示的に通知するためのドメイン例外クラス群。
"""


class VoiceRecognizerError(Exception):
    """Voice Recognizer パッケージの基底例外クラス"""

    pass


class DeviceNotFoundError(VoiceRecognizerError):
    """マイクデバイスが存在しない、または切断された場合に発生する例外"""

    pass


class ModelNotFoundError(VoiceRecognizerError):
    """指定されたモデルファイルが存在しない場合に発生する例外"""

    pass


class AudioPreprocessingError(VoiceRecognizerError):
    """音声波形の前処理やデコードに失敗した場合に発生する例外"""

    pass
