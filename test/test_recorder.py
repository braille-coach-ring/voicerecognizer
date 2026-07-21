import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from core.implementations import DefaultAudioRecorder

if __name__ == "__main__":
    # 設定パラメータ
    DURATION = 5.0        # 録音時間（秒）
    SAMPLE_RATE = 16000   # サンプリングレート（16kHz）
    OUTPUT_FILENAME = "test_output.wav"

    # 現在接続されているオーディオデバイスの一覧を表示（切り分け用）
    print("--- 利用可能なオーディオデバイス一覧 ---")
    print(sd.query_devices())
    print(f"デフォルトのインプットデバイスID: {sd.default.device[0]}")
    print("---------------------------------------\n")

    recorder = DefaultAudioRecorder()
    
    print(f"【録音開始】マイクに向かって何か話してください（{DURATION}秒間）...")
    try:
        audio_data = recorder.record(duration=DURATION, sample_rate=SAMPLE_RATE)
        print("【録音終了】保存処理中...")
        
        # sounddeviceのfloat32(-1.0〜1.0)入力を、WAV保存用にint16に変換
        # （そのままだと再生ソフトによって音が出ないことがあるため）
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # ファイルに書き出し
        wavfile.write(OUTPUT_FILENAME, SAMPLE_RATE, audio_int16)
        print(f"成功！ '{OUTPUT_FILENAME}' に音声が保存されました。再生して確認してください。")
        
    except Exception as e:
        print(f"\n[エラー発生] 録音中に問題が発生しました:\n{e}")
