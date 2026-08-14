from pathlib import Path
import shutil
import re

# ============================================================
# 1. 元データ
# ============================================================

SOURCE_DIR = Path(
    r"C:\Users\rinry\Downloads\E4"
)

# 整理後の保存先
DEST_DIR = Path(
    r"C:\voicelibrary\dataset\name3"
)


# ============================================================
# 2. かな → ラベル
# ============================================================

KANA_TO_LABEL = {
    # -------------------------
    # 清音 46
    # -------------------------
    "あ": "a",
    "い": "i",
    "う": "u",
    "え": "e",
    "お": "o",

    "か": "ka",
    "き": "ki",
    "く": "ku",
    "け": "ke",
    "こ": "ko",

    "さ": "sa",
    "し": "shi",
    "す": "su",
    "せ": "se",
    "そ": "so",

    "た": "ta",
    "ち": "chi",
    "つ": "tsu",
    "て": "te",
    "と": "to",

    "な": "na",
    "に": "ni",
    "ぬ": "nu",
    "ね": "ne",
    "の": "no",

    "は": "ha",
    "ひ": "hi",
    "ふ": "fu",
    "へ": "he",
    "ほ": "ho",

    "ま": "ma",
    "み": "mi",
    "む": "mu",
    "め": "me",
    "も": "mo",

    "や": "ya",
    "ゆ": "yu",
    "よ": "yo",

    "ら": "ra",
    "り": "ri",
    "る": "ru",
    "れ": "re",
    "ろ": "ro",

    "わ": "wa",
    "を": "wo",
    "ん": "n",

    # -------------------------
    # 濁音 20
    # -------------------------
    "が": "ga",
    "ぎ": "gi",
    "ぐ": "gu",
    "げ": "ge",
    "ご": "go",

    "ざ": "za",
    "じ": "ji",
    "ず": "zu",
    "ぜ": "ze",
    "ぞ": "zo",

    "だ": "da",
    "ぢ": "di",
    "づ": "du",
    "で": "de",
    "ど": "do",

    "ば": "ba",
    "び": "bi",
    "ぶ": "bu",
    "べ": "be",
    "ぼ": "bo",

    # -------------------------
    # 半濁音 5
    # -------------------------
    "ぱ": "pa",
    "ぴ": "pi",
    "ぷ": "pu",
    "ぺ": "pe",
    "ぽ": "po",

    # -------------------------
    # 拗音 33
    # -------------------------
    "きゃ": "kya",
    "きゅ": "kyu",
    "きょ": "kyo",

    "しゃ": "sha",
    "しゅ": "shu",
    "しょ": "sho",

    "ちゃ": "cha",
    "ちゅ": "chu",
    "ちょ": "cho",

    "にゃ": "nya",
    "にゅ": "nyu",
    "にょ": "nyo",

    "ひゃ": "hya",
    "ひゅ": "hyu",
    "ひょ": "hyo",

    "みゃ": "mya",
    "みゅ": "myu",
    "みょ": "myo",

    "りゃ": "rya",
    "りゅ": "ryu",
    "りょ": "ryo",

    "ぎゃ": "gya",
    "ぎゅ": "gyu",
    "ぎょ": "gyo",

    "じゃ": "ja",
    "じゅ": "ju",
    "じょ": "jo",

    "びゃ": "bya",
    "びゅ": "byu",
    "びょ": "byo",

    "ぴゃ": "pya",
    "ぴゅ": "pyu",
    "ぴょ": "pyo",
}


# ============================================================
# 3. データセット整理
# ============================================================

def main():

    print("=" * 60)
    print("音声データセット整理開始")
    print("=" * 60)

    # 元フォルダ確認
    if not SOURCE_DIR.exists():
        print("❌ 元フォルダが見つかりません")
        print("探した場所:")
        print(SOURCE_DIR)
        return

    print("✓ 元フォルダ:")
    print(SOURCE_DIR)
    print()

    # 保存先作成
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # サブフォルダも含めてwavを検索
    wav_files = list(SOURCE_DIR.rglob("*.wav"))

    print(f"✓ 見つかったwavファイル: {len(wav_files)}個")
    print()

    if not wav_files:
        print("❌ wavファイルが見つかりません")
        return

    success_count = 0
    skip_count = 0

    # ========================================================
    # ファイルを1つずつ処理
    # ========================================================

    for wav_file in wav_files:

        # ----------------------------------------------------
        # ファイル名から拡張子を除く
        #
        # あ_B4.wav       → あ_B4
        # きゃ_B4.wav     → きゃ_B4
        # ----------------------------------------------------

        filename = wav_file.stem

        # "_" より前をラベルとして取得
        #
        # あ_B4 → あ
        # きゃ_B4 → きゃ
        kana = filename.split("_")[0]

        # 対応するラベルを取得
        label = KANA_TO_LABEL.get(kana)

        # 対応していない音
        if label is None:
            print(
                f"⚠️ 対応なし: {wav_file.name}"
                f" → 抽出した文字: {kana}"
            )
            skip_count += 1
            continue

        # labelフォルダ
        label_dir = DEST_DIR / label
        label_dir.mkdir(parents=True, exist_ok=True)

        # 既存ファイルを確認
        existing_files = list(label_dir.glob("*.wav"))

        # 連番
        number = len(existing_files) + 1

        destination = label_dir / f"{number:03d}.wav"

        # コピー
        shutil.copy2(wav_file, destination)

        print(
            f"✓ {wav_file.name}"
            f" → {label}/{number:03d}.wav"
        )

        success_count += 1

    # ========================================================
    # 結果
    # ========================================================

    print()
    print("=" * 60)
    print("整理完了")
    print("=" * 60)

    print(f"成功    : {success_count}個")
    print(f"スキップ: {skip_count}個")
    print(f"保存先  : {DEST_DIR}")

    print()
    print("ラベル数:", len(KANA_TO_LABEL))


if __name__ == "__main__":
    main()