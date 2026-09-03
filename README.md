# voicerecognizer

点字学習装置向けの、短い日本語音声を 105 ラベルに分類する音声認識ライブラリです。

清音、濁音、半濁音、拗音、`other` を含む 105 クラスを扱います。現在の主力は
Wav2Vec2 + ONNX Runtime の推論で、CPU 環境でも使いやすい低遅延推論と、収集データを
継続的に学習へ戻すワークフローを重視しています。

## 主な用途

- 点字学習装置での単音節ひらがな認識
- マイク入力のリアルタイム認識
- 認識結果を人間が丸付けしながら `dataset/collected/` に蓄積
- `merged_dataset/index.csv` と `processed_dataset/index.csv` を使った再現性のある学習
- 評価結果、混同行列、レビュー候補を使った精度改善

## 現在の特徴

- 105 ラベル分類: 清音 46、濁音 20、半濁音 5、拗音 33、その他 1
- Wav2Vec2 推論: `Wav2Vec2Recognizer` が既定の認識器
- CNN 推論: 軽量な代替認識器として `CNNRecognizer` も利用可能
- モデル自動取得: Hugging Face Hub の公開モデルをローカルキャッシュへ同期
- 精度優先の前処理: `dynamic_trimming=False` を Wav2Vec2 の既定値に設定
- 学習時 augmentation: ノイズ、音量、時間シフト、軽い速度変化、軽いピッチ変化、実機ノイズ混合
- 混同ペア重点サンプラー: 過去の混同行列から間違えやすいラベルを多めに学習
- speaker-aware split: 話者単位で validation を分ける評価モード
- dataset audit: 欠損、古い前処理済みデータ、ラベル偏り、重複、未レビュー候補を一括確認

## セットアップ

基本作業ディレクトリはリポジトリ直下です。

```powershell
cd C:\voicelibrary
uv sync
```

初回セットアップとして、必要ディレクトリの作成、モデル取得、データセット統合、前処理をまとめて実行できます。

```powershell
uv run python script\setup_environment.py
```

モデルだけ取得したい場合:

```powershell
uv run python script\download_from_hf.py --type all
uv run python script\download_from_hf.py --type wav2vec2
uv run python script\download_from_hf.py --type cnn
```

既定のモデル保存先は `~/.cache/voicerecognizer/weights` です。ローカルの別モデルを使う場合は、
各コマンドで `--model-path` や Python API の `model_path` を指定します。

## クイックスタート

音声ファイルを 1 つ認識します。

```powershell
uv run python main.py dataset\mikeryu\a\001.wav --model wav2vec2
uv run python main.py dataset\mikeryu\a\001.wav --model cnn
```

音声ファイルを省略すると、マイクからの連続認識と人間による丸付けセッションを開始します。丸付け結果は
`dataset/collected/` に保存されます。

```powershell
uv run python main.py --model wav2vec2
```

VAD のしきい値を調整して起動する例:

```powershell
uv run python main.py --model wav2vec2 --silence-threshold 0.03 --rms-threshold 0.008
```

Python から使う例:

```python
import voicerecognizer as vr

recognizer = vr.Wav2Vec2Recognizer()
text = recognizer.recognize("sample.wav")
print(text)

cnn = vr.CNNRecognizer()
print(cnn.recognize("sample.wav"))
```

## データセット構成

```text
dataset\                       生音声データ
dataset\<speaker>\<label>\     手動配置された話者別 wav
dataset\collected\pc_xxxx\     対話モードで保存された wav と metadata.csv
merged_dataset\index.csv       学習対象の統合 index
processed_dataset\             前処理済み wav
processed_dataset\index.csv    前処理済み wav と元ファイル情報の対応表
evaluation_results\            評価 JSON、HTML、レビュー候補、audit 結果
```

`processed_dataset/index.csv` には、前処理済み wav だけでなく、元ファイル、話者、予測テキスト、
発話開始位置、発話長、前処理時間も保存されます。話者別評価やハード例分析ではこの index を使います。

主な列:

```text
filepath,label,source_filepath,speaker,predicted_text,onset_ms,offset_ms,
speech_duration_ms,processed_duration_ms,preprocess_latency_ms
```

## データ収集

対話式の収集スクリプトを使います。保存先は `dataset\<speaker_id>\<label>\*.wav` です。

```powershell
uv run python script\collect_with_label_sound.py rinry --repeat 10 --no-upload
```

不足分だけ集める例:

```powershell
uv run python script\collect_with_label_sound.py rinry --target-per-label 50 --no-upload
```

特定ラベルだけ集める例:

```powershell
uv run python script\collect_with_label_sound.py rinry --labels shu,chu,sho,cho,sha,cha --repeat 20 --no-upload
```

グループを絞る例:

```powershell
uv run python script\collect_with_label_sound.py rinry --groups seion --repeat 10 --no-upload
uv run python script\collect_with_label_sound.py rinry --groups dakuon,handakuon --repeat 10 --no-upload
uv run python script\collect_with_label_sound.py rinry --groups yoon --repeat 10 --no-upload
```

`other` も集める例:

```powershell
uv run python script\collect_with_label_sound.py rinry --include-other --repeat 10 --no-upload
```

## データ統合と前処理

生データと `dataset/collected/` を統合して `merged_dataset/index.csv` を作ります。

```powershell
uv run python script\merge_data.py
```

統合済みデータを前処理して `processed_dataset/` と `processed_dataset/index.csv` を作ります。

```powershell
uv run python script\preprocess.py
```

学習コマンドは既定で統合と前処理を自動実行します。前処理済みデータをそのまま使う場合は
`--skip-prep` を指定します。

```powershell
uv run python train.py --model wav2vec2 --skip-prep
```

## Dataset Audit

データセットの健康状態を一括確認します。

```powershell
uv run python script\dataset_audit.py
```

確認できるもの:

- `merged_dataset/index.csv` の欠損 wav
- `processed_dataset` の wav 数、index 数、stale 判定
- ラベル数の偏り
- 重複 filepath、重複 source、同一音声 hash
- `review_candidates.json` の未レビュー数
- `evaluation_result.json` 由来の上位混同ペア

現在のローカルデータ状態:

```text
Index rows: 5671 (existing=5671, missing=0)
Processed wavs: 5671 (index rows=5671, stale=False)
Labels below 50: 19/105
Review candidates: total=5608, unreviewed=5532
Top confusion pairs: shu->chu, sho->cho, mi->ni, ji->di, pi->bi, sha->cha, su->tsu, du->zu
```

レポートは `evaluation_results/dataset_audit.json` に保存されます。

## 学習

CNN を学習します。

```powershell
uv run python train.py
uv run python train.py --model cnn
```

Wav2Vec2 を学習します。

```powershell
uv run python train.py --model wav2vec2
```

精度改善向けの Wav2Vec2 学習例:

```powershell
uv run python train.py --model wav2vec2 --speaker-aware-split --epochs 30 --batch-size 4 --learning-rate 0.00003
```

テスト学習で Hugging Face にアップロードしたくない場合:

```powershell
uv run python train.py --model wav2vec2 --speaker-aware-split --no-hf-upload
```

実機ノイズ wav を用意した場合:

```powershell
uv run python train.py --model wav2vec2 --speaker-aware-split --augmentation-noise-dir dataset\noise\device
```

Wav2Vec2 学習の主な既定値:

- `--epochs 30`
- `--batch-size 4`
- `--learning-rate 3e-5`
- `--freeze-transformer-layers 10`
- class weights 有効
- augmentation 有効
- balanced sampler 有効
- confusion-pair sampler 有効
- 学習後の ONNX export 有効

主なオプション:

```powershell
uv run python train.py --model wav2vec2 --from-scratch
uv run python train.py --model wav2vec2 --resume-from weights\wav2vec2_last
uv run python train.py --model wav2vec2 --no-augment
uv run python train.py --model wav2vec2 --no-balanced-sampler
uv run python train.py --model wav2vec2 --no-confusion-pair-sampler
uv run python train.py --model wav2vec2 --no-onnx-export
uv run python train.py --model wav2vec2 --no-hf-upload
uv run python train.py --model wav2vec2 --num-workers 0
```

## 評価

モデルを評価し、JSON と HTML レポートを出力します。

```powershell
uv run python script\evaluate.py --model-type wav2vec2
uv run python script\evaluate.py --model-type cnn
```

`index.csv` に `speaker` 列、または推定可能な `dataset\<speaker>\<label>\*.wav` 形式のパスがある場合、
評価 JSON と HTML には話者別の Accuracy、Macro F1、誤認識数、主な混同ペアも出力されます。

最新モデルの予測で `merged_dataset/index.csv` の `predicted_text` を更新してから評価します。

```powershell
uv run python script\evaluate.py --model-type wav2vec2 --update-index
```

index 内の `predicted_text` だけで評価します。

```powershell
uv run python script\evaluate.py --from-dataset-only
```

出力先を指定する例:

```powershell
uv run python script\evaluate.py --model-type wav2vec2 --dataset-dir merged_dataset --output-json evaluation_results\evaluation_result.json --output-html evaluation_results\evaluation_report.html
```

## Dynamic Trimming 比較

Wav2Vec2 推論時の `dynamic_trimming=True/False` を同一データで比較します。

```powershell
uv run python script\compare_dynamic_trimming_accuracy.py --model-path weights\wav2vec2_best --dataset-dir merged_dataset
```

現在の比較結果:

```text
dynamic_trimming=False accuracy=0.3851, macro_f1=0.3766, weighted_f1=0.3838
dynamic_trimming=True  accuracy=0.3835, macro_f1=0.3621, weighted_f1=0.3689
recommended_mode=fixed_padding
```

この結果に基づき、`Wav2Vec2Recognizer` の既定値は `dynamic_trimming=False` です。

## レビュー

評価で出力された誤判定や品質懸念の候補を HTML で確認できます。

```powershell
uv run python script\review_server.py
```

ブラウザで次を開きます。

```text
http://127.0.0.1:8765/evaluation_results/review_report.html
```

レビュー結果は `evaluation_results/review_decisions.json` に保存され、以後の評価や audit に反映されます。

## 精度改善で優先するデータ

現在は、次の混同ペアを重点的に増やすのが有効です。

```text
shu/chu, sho/cho, sha/cha, mi/ni, ji/di, pi/bi, su/tsu, du/zu, pa/ta
```

不足しているラベルは次のあたりです。

```text
other, ha, go, za, zu, be, do, no, mi, ji, bu, pya
```

収集時の目安:

- 1 ラベルあたり 20 から 30 件を追加
- 実機マイク、実際の利用距離、実際の部屋の音で録る
- 強く言い過ぎず、短く自然に発音する
- クリップ、二重発話、ラベル間違いは残さない
- 実機ノイズだけの wav も `dataset\noise\device\` に集める

## 品質チェック

関連テスト:

```powershell
uv run python -m pytest
```

Ruff:

```powershell
uv run python -m ruff check .
uv run python -m ruff format .
```

品質ゲート:

```powershell
uv run python script\check_quality_gate.py
```

## 環境変数

公開モデルの利用だけなら設定不要です。プライベートモデルやアップロードを使う場合は `.env` または環境変数で設定します。

| 変数 | 説明 |
| --- | --- |
| `VOICERECOGNIZER_HF_REPO_ID` / `HF_REPO_ID` | Hugging Face Hub のモデルリポジトリ |
| `VOICERECOGNIZER_HF_TOKEN` / `HF_TOKEN` | Hugging Face 認証トークン |
| `VOICERECOGNIZER_HF_AUTO_UPLOAD` / `HF_AUTO_UPLOAD` | 学習後にモデルを自動アップロードするか |
| `VOICERECOGNIZER_CACHE_DIR` | モデルキャッシュの保存先 |

既定の公開リポジトリ:

```text
braille-mate/braille-mate-hiragana-recognizer
```

## 関連ドキュメント

- `README_EXECUTION.md`: コマンド逆引き
- `PROJECT_OVERVIEW.md`: 古い構成メモ。現行 README より古い情報が含まれる可能性があります
- `Docs/MODEL_HUB_GUIDE.md`: Hugging Face Hub 運用
- `Docs/MODULE_USAGE_GUIDE.md`: モジュール利用ガイド
- `Docs/REFACTORING_AND_FUTURE_ROADMAP.md`: リファクタリングと今後の方向性

## ライセンス

MIT License
