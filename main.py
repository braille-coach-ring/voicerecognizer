"""
Voice Recognizer CLI / Interactive Annotation Entry Point

Usage:
  uv run python main.py                    # Continuous interactive mic recognition & human grading
  uv run python main.py path/to/sample.wav # Recognize a single audio file
"""

import argparse
import datetime
import logging
from pathlib import Path
import sounddevice as sd

from config import DEFAULT_PREPROCESS_CONFIG, DEFAULT_RECOGNITION_CONFIG
from config_labels import HIRAGANA_TO_ROMAJI, ROMAJI_TO_HIRAGANA
from core.factory.recognizer_factory import RecognizerFactory
from core.services.audio_pipeline import AudioPipeline
from core.services.voice_recognizer import VoiceRecognizer
from runtime.vad import VoiceActivityDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(DEFAULT_RECOGNITION_CONFIG.log_path, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run voice recognition.")
    parser.add_argument(
        "audio",
        nargs="?",
        type=Path,
        help="Path to a wav file. If omitted, continuous interactive recognition mode is started.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_RECOGNITION_CONFIG.model_type,
        choices=RecognizerFactory.available_strategies(),
        help="Recognition strategy to use.",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=DEFAULT_PREPROCESS_CONFIG.vad_silence_threshold,
        help="Peak volume threshold for VAD (default: %(default)s).",
    )
    parser.add_argument(
        "--rms-threshold",
        type=float,
        default=DEFAULT_PREPROCESS_CONFIG.vad_rms_threshold,
        help="RMS volume threshold for VAD (default: %(default)s).",
    )
    return parser


def run_interactive_grading_session(pipeline: AudioPipeline) -> None:
    """
    連続対話・人間丸付け（採点）セッション。
    マイクの音声を聞き取り待機 ➔ 予測テキスト表示 ➔ 録音音声の再生 ➔ 正解ラベル選択・保存 ➔ 繰り返し。
    50音・濁音・半濁音・拗音・その他の全105日本語ひらがなラベルに対応。
    """
    SHORTCUT_CHOICES = {
        "1": "a",
        "2": "i",
        "3": "u",
        "4": "e",
        "5": "o",
        "6": "other",
    }

    print("\n" + "=" * 65)
    print(" 🎤 リアルタイム音声認識 ＆ 連続人間丸付け（採点・アノテーション）モード")
    print("=" * 65)
    print("※ 発声を検知すると自動で予測結果を表示し、録音音声を再生します。")
    print("※ 全105ひらがなラベル（50音・濁音・半濁音・拗音・その他）の直打ちに対応。")
    print("※ 画面の指示に従って正解を選択してください（Ctrl+C で終了）。\n")

    session_count = 0
    try:
        while True:
            session_count += 1
            print(f"\n--- [ セッション #{session_count} ] ---")
            print("👂 音声待機中... マイクに向かって声を出してください。")

            res = pipeline.capture_until_speech()
            if res is None:
                print("\nセッションを終了しました。")
                break

            audio_data, predicted_text, stats = res
            hiragana_pred = ROMAJI_TO_HIRAGANA.get(predicted_text, predicted_text)
            disp_pred = (
                f"{predicted_text} ({hiragana_pred})"
                if hiragana_pred != predicted_text
                else predicted_text
            )

            confidence_str = ""
            if "confidence" in stats:
                confidence_str = f" (確信度: {stats['confidence']*100:.1f}%)"

            print("\n" + "─" * 60)
            print(f"🤖 予測結果: 【 {disp_pred} 】{confidence_str}")
            print("─" * 60)

            # 詳細タイムライン & 計測内訳の表示
            start_dt_str = (
                stats["speech_start_time"].strftime("%H:%M:%S.%f")[:-3]
                if "speech_start_time" in stats
                else "N/A"
            )
            end_dt_str = (
                stats["speech_end_time"].strftime("%H:%M:%S.%f")[:-3]
                if "speech_end_time" in stats
                else "N/A"
            )
            onset_ms = stats.get("onset_ms", 0.0)
            offset_ms = stats.get("offset_ms", 0.0)
            speech_dur_ms = stats.get("speech_duration_ms", 0.0)
            prep_ms = stats.get("preprocess_latency_ms", 0.0)
            inf_ms = stats.get("inference_latency_ms", 0.0)
            total_ms = stats.get("total_latency_ms", 0.0)

            print("⏱️  詳細タイムライン & 処理時間計測内訳:")
            print(f"  ・話し始め時刻   : {start_dt_str} (録音波形内: {onset_ms:.1f} ms)")
            print(f"  ・話し終わり時刻 : {end_dt_str} (録音波形内: {offset_ms:.1f} ms)")
            print(f"  ・発声持続時間   : {speech_dur_ms:.1f} ms ({speech_dur_ms/1000.0:.2f}秒)")
            print(f"  ・前処理時間     : {prep_ms:5.1f} ms (VADトリミング・特徴量抽出)")
            print(f"  ・推論時間       : {inf_ms:5.1f} ms (モデル推論)")
            print(f"  ・合計処理時間   : {total_ms:5.1f} ms (前処理 + 推論)")
            print("─" * 60)

            # 音声の再生
            try:
                sd.play(audio_data, pipeline.audio_capture.sample_rate)
                sd.wait()
            except Exception as e:
                logger.warning("録音音声の再生に失敗しました: %s", e)

            # 人間による丸付け（採点）ターン
            while True:
                prompt = (
                    f"正解ラベルを選択してください:\n"
                    f"  [Enter] : 予測通り「{disp_pred}」として確定\n"
                    f"  [文字入力] : ひらがな（例: 「か」「きゃ」）またはローマ字（例: 「ka」「kya」）\n"
                    f"  [1-5]   : あ/い/う/え/お  [6] other (その他/雑音)\n"
                    f"  [r] 音声を再再生  [q] 終了\n"
                    f"選択 > "
                )
                raw_input = input(prompt).strip()
                user_input = raw_input.lower()

                if user_input == "q":
                    print("セッションを終了します。お疲れ様でした！")
                    return
                elif user_input == "r":
                    try:
                        sd.play(audio_data, pipeline.audio_capture.sample_rate)
                        sd.wait()
                    except Exception as e:
                        logger.warning("録音音声の再生に失敗しました: %s", e)
                    continue
                elif user_input == "":
                    ground_truth = predicted_text
                    break
                elif user_input in SHORTCUT_CHOICES:
                    ground_truth = SHORTCUT_CHOICES[user_input]
                    break
                elif raw_input in HIRAGANA_TO_ROMAJI:
                    ground_truth = HIRAGANA_TO_ROMAJI[raw_input]
                    break
                elif user_input in HIRAGANA_TO_ROMAJI:
                    ground_truth = HIRAGANA_TO_ROMAJI[user_input]
                    break
                elif user_input in ROMAJI_TO_HIRAGANA:
                    h_char = ROMAJI_TO_HIRAGANA[user_input]
                    ground_truth = HIRAGANA_TO_ROMAJI.get(h_char, user_input)
                    break
                else:
                    print("⚠️ 無効な入力です。ひらがな（例: 「か」「きゃ」）またはローマ字（例: 「ka」「kya」）を入力してください。")

            gt_hiragana = ROMAJI_TO_HIRAGANA.get(ground_truth, ground_truth)
            disp_gt = (
                f"{ground_truth} ({gt_hiragana})"
                if gt_hiragana != ground_truth
                else ground_truth
            )

            # データ保存 (metadata.csv と .wav)
            pipeline.output_worker.save(
                audio_data=audio_data,
                predicted_text=predicted_text,
                ground_truth=ground_truth,
                timestamp=datetime.datetime.now().timestamp(),
                sample_rate=pipeline.audio_capture.sample_rate,
            )
            print(f"✅ 保存完了: 予測=「{disp_pred}」, 正解=「{disp_gt}」 (データセットに蓄積されました)")

    except KeyboardInterrupt:
        print("\n\nユーザー操作によりセッションを停止しました。お疲れ様でした！")


def main() -> None:
    args = build_parser().parse_args()

    logger.info("Voice Recognizer を起動しました。")
    logger.info("使用モデル: %s", args.model)
    logger.info("VAD設定: Peak閾値=%.4f / RMS閾値=%.4f", args.silence_threshold, args.rms_threshold)

    strategy = RecognizerFactory.create(args.model)
    recognizer = VoiceRecognizer(strategy)
    vad = VoiceActivityDetector(
        silence_threshold=args.silence_threshold,
        rms_threshold=args.rms_threshold,
    )
    pipeline = AudioPipeline(recognizer, vad=vad)
    logger.info("パイプラインの構築が完了しました。")

    if args.audio is None:
        run_interactive_grading_session(pipeline)
        return

    logger.info("音声ファイルを入力します...")
    result = pipeline.run(args.audio)
    logger.info("ファイル認識結果: %s", result)


if __name__ == "__main__":
    main()
