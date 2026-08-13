# 📚 Voice Recognizer 外部モジュール導入・利用ガイド

`voicerecognizer` パッケージは、別プロジェクト、別リポジトリ、GUI/Webアプリ、ゲーム、ロボット制御ソフトなどから **単体 Python モジュールとしてインポートし、数行のコードで高性能な日本語ひらがな音声認識を利用できる** よう設計されています。

---

## 🚀 1. インストール方法

他プロジェクトの Python 環境へ追加する場合、以下のいずれかの方法で簡単にインストールできます。

### 【方法 A】GitHub から直接インストール（一番おすすめ）
リポジトリの URL を指定するだけで、依存ライブラリ込みで 1 行で一括インストールされます。

* **`uv` を使う場合**:
  ```bash
  uv add git+https://github.com/braille-coach-ring/voicerecognizer.git
  ```
* **通常の `pip` を使う場合**:
  ```bash
  pip install git+https://github.com/braille-coach-ring/voicerecognizer.git
  ```

### 【方法 B】ローカルの別フォルダからインストール（開発用）
同じ PC 内の別アプリで開発・テストする場合、ローカルパスを指定して編集可能モード (`-e`) でインストールします。

* **`uv` の場合**:
  ```bash
  uv add --editable /path/to/voicerecognizer
  ```
* **`pip` の場合**:
  ```bash
  pip install -e /path/to/voicerecognizer
  ```

---

## 💡 2. 基本的な使い方（クイックスタート）

### ① ワンライナーで音声ファイルをテキスト認識する
```python
import voicerecognizer as vr

# 音声ファイル (.wav) または NumPy 波形配列を渡すだけで即座に認識
text = vr.recognize("sample.wav")
print(f"認識結果: {text}")  # 例: "あ"
```

### ② 非同期 (async/await) でメインスレッドを止めずに認識する
```python
import asyncio
import voicerecognizer as vr

async def main():
    text = await vr.recognize_async("sample.wav")
    print(f"非同期認識結果: {text}")

asyncio.run(main())
```

### ③ Recognizer インスタンスを作成して確信度・上位候補を取得する
```python
import voicerecognizer as vr

# Wav2Vec2 ONNX 高速推論エンジンの初期化
recognizer = vr.Wav2Vec2Recognizer()

# 1. 通常認識
text = recognizer.recognize("sample.wav")
confidence = recognizer.last_confidence  # 確信度スコア (例: 0.985 -> 98.5%)
print(f"結果: {text} (確信度: {confidence:.1%})")

# 2. Top-K 上位候補の取得
candidates = recognizer.recognize_with_candidates("sample.wav", top_k=3)
for rank, (label, prob) in enumerate(candidates, 1):
    print(f"第{rank}候補: {label} ({prob:.1%})")
```

---

## 🎤 3. リアルタイム非同期ストリーミング認識 (`AudioStreamListener`)

マイクからの音声をリアルタイムで常時キャプチャし、**発声を検知してひらがなが認識された「その瞬間だけ」非同期で文字を返却**します。

### 基本的なストリーミング受信コード
```python
import asyncio
import voicerecognizer as vr

async def main():
    listener = vr.AudioStreamListener()
    print("🎤 マイクのリアルタイム聴取を開始しました（喋った時だけ文字が表示されます）")

    # 発声検知時のみ yield される
    async for text in listener.listen():
        print(f"🗣️ 認識: {text}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("終了しました。")
```

### ライフサイクル制御 (`start`, `pause`, `close`)
コンテキストマネージャ (`with` 文) に対応しており、安全かつ再開待機時間ゼロの制御が可能です。

* **`start()`**: 聴取の開始（一時停止状態からの即座再開も含む）
* **`pause()`**: マイクとモデルはメモリ上に保持したまま、キャプチャ・推論のみ一時停止（再開コストゼロ）
* **`close()`**: マイクデバイスを完全にシャットダウンし、リソースを全解放

```python
import voicerecognizer as vr

# with 文を抜けると自動で listener.close() が呼ばれて安全解放される
with vr.AudioStreamListener() as listener:
    
    # 聴取処理を一時停止（モデルはロードしたまま保持）
    listener.pause()
    
    # 待機時間ゼロで即座に再開！
    listener.start()
```

### 詳細情報付きストリーミング (`listen_details`)
認識文字だけでなく、確信度スコアや Top-3 候補、レイテンシ時間が必要な高度なアプリ用 API です。

```python
import asyncio
import voicerecognizer as vr

async def main():
    listener = vr.AudioStreamListener()
    
    async for result in listener.listen_details():
        print(f"文字: {result.text}")
        print(f"確信度: {result.confidence:.1%}")
        print(f"Top-3候補: {result.top3_candidates}")

asyncio.run(main())
```

---

## 🛡️ 4. エラーハンドリング・例外設計

サイレントな誤作動を防ぐため、問題発生時には明確なドメイン例外クラスが発生します。

| 例外クラス | 発生条件 |
| :--- | :--- |
| **`vr.ModelNotFoundError`** | 指定された ONNX モデルや重みファイルが存在しない場合 |
| **`vr.DeviceNotFoundError`** | マイクが存在しない、または録音中に切断された場合 |
| **`vr.AudioPreprocessingError`** | 音声波形のデコードや前処理に失敗した場合 |
| **`vr.VoiceRecognizerError`** | パッケージ全体の基底例外 |

```python
import voicerecognizer as vr

try:
    recognizer = vr.Wav2Vec2Recognizer(model_path="invalid/path")
except vr.ModelNotFoundError as e:
    print(f"⚠️ モデルファイルが見つかりません: {e}")
except vr.DeviceNotFoundError as e:
    print(f"⚠️ マイクデバイスエラー: {e}")
```

---

## ⚙️ 5. 主要クラス・関数リファレンスまとめ

| エクスポート名 | 種別 | 概要 |
| :--- | :--- | :--- |
| `vr.recognize()` | 関数 | ワンライナーで音声ファイル/波形をテキスト認識する関数 |
| `vr.recognize_async()` | 関数 | 非同期 (async/await) でノンブロッキング認識する関数 |
| `vr.Wav2Vec2Recognizer` | クラス | Wav2Vec2 ONNX 高速推論ストラテジー |
| `vr.CNNRecognizer` | クラス | CNN 認識ストラテジー |
| `vr.AudioStreamListener` | クラス | リアルタイムマイクストリーミング非同期聴取器 |
| `vr.RecognitionResult` | データ | 詳細結果（文字・自信度・Top3候補）オブジェクト |
| `vr.ModelNotFoundError` | 例外 | モデル未存在例外 |
| `vr.DeviceNotFoundError` | 例外 | マイク未接続例外 |
