# voicelibrary 実行方法まとめ

このファイルは、環境セットアップから音声収集、前処理、学習、評価、推論までの実行コマンドをまとめたものです。基本はプロジェクト直下 `C:\voicelibrary` で実行します。

## 1. セットアップ

依存関係を入れます。

```powershell
uv sync
```

`uv run python ...` が使えない環境では、仮想環境の Python を直接使えます。

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe main.py --help
```

Hugging Face へモデルをアップロードする場合だけ、`.env.example` を参考に `.env` を設定します（リポジトリ ID は `config.py` で固定されているため、トークンのみで動作します）。

```text
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## 2. まず動かす

マイクから連続認識します。

```powershell
uv run python main.py
```

wav ファイルを 1 つだけ認識します。

```powershell
uv run python main.py dataset\mikeryu\a\001.wav --model cnn
```

モデルを切り替えます。

```powershell
uv run python main.py --model cnn
uv run python main.py --model wav2vec2
uv run python main.py --model whisper
```

VAD のしきい値を調整して起動します。

```powershell
uv run python main.py --model cnn --silence-threshold 0.03 --rms-threshold 0.008
```

## 3. 音声データ収集

保存先は `dataset\<speaker_id>\<label>\001.wav` の形式です。

デフォルトでは、清音・濁音・半濁音・拗音の 104 ラベルを収集します。`other` も集める場合は `--include-other` を付けます。

発話検知モードで効率よく収集します。

```powershell
uv run python script\collect.py username --repeat 10 --no-upload
```
実行後にgitにpushしてね

各ラベルが指定件数になるまで、不足分だけ集めます。

```powershell
uv run python script\collect.py rinry --target-per-label 20 --no-upload
```

`other` も含めて 105 ラベルを集めます。

```powershell
uv run python script\collect.py rinry --repeat 10 --include-other --no-upload
```

固定秒数録音に切り替えます。

```powershell
uv run python script\collect.py rinry --mode fixed --repeat 10 --start-delay 0.25 --interval 0.25 --no-upload
```

ラベルグループを絞ります。

```powershell
uv run python script\collect.py rinry --groups seion --repeat 10 --no-upload
uv run python script\collect.py rinry --groups dakuon,handakuon --repeat 10 --no-upload
uv run python script\collect.py rinry --groups yoon --repeat 10 --no-upload
```

特定ラベルだけ集めます。

```powershell
uv run python script\collect.py rinry --labels a,i,u,e,o --repeat 20 --no-upload
uv run python script\collect.py rinry --labels kya,kyu,kyo,gya,gyu,gyo --repeat 10 --no-upload
```

途中のラベルから再開します。

```powershell
uv run python script\collect.py rinry --target-per-label 20 --start-at sha --no-upload
```

ラベルごとに開始確認を入れます。プロンプトで Enter は開始、`s` はスキップ、`q` は終了です。

```powershell
uv run python script\collect.py rinry --repeat 10 --confirm-each-label --no-upload
```

発話検知の細かい調整例です。

```powershell
uv run python script\collect.py rinry --repeat 10 --silence-threshold 0.03 --rms-threshold 0.008 --speech-end-seconds 0.25 --max-utterance-seconds 1.2 --no-upload
```

収集後に GitHub へアップロードまで行う場合です。

```powershell
uv run python script\collect.py rinry --repeat 10 --upload
```

## 4. データ統合と前処理

Raw データと `dataset\collected` のメタデータを統合し、`merged_dataset\index.csv` を作ります。

```powershell
uv run python script\merge_data.py
```

統合済みデータを前処理し、`processed_dataset\` を作ります。

```powershell
uv run python script\preprocess.py
```

学習コマンドはデフォルトで統合と前処理を自動実行します。すでに前処理済みで飛ばしたい場合は `--skip-prep` を使います。

```powershell
uv run python train.py --skip-prep
```

## 5. 学習

CNN を学習します。デフォルトでは既存重みを再利用し、自動でデータ統合と前処理も行います。

```powershell
uv run python train.py
```

CNN を 0 から学習します。

```powershell
uv run python train.py --from-scratch
```

CNN の学習パラメータを指定します。

```powershell
uv run python train.py --epochs 150 --batch-size 8 --learning-rate 0.001
```

Wav2Vec2 を学習します。

```powershell
uv run python train.py --model wav2vec2
```

Wav2Vec2 をベースモデルから学習します。

```powershell
uv run python train.py --model wav2vec2 --from-scratch
```

Wav2Vec2 の学習パラメータを指定します。

```powershell
uv run python train.py --model wav2vec2 --epochs 30 --batch-size 4 --learning-rate 0.00003
```

モデル別スクリプトを直接実行することもできます。

```powershell
uv run python models\cnn\train.py
uv run python models\cnn\train.py --from-scratch
uv run python models\wav2vec2\train.py
uv run python models\wav2vec2\train.py --from-scratch
uv run python models\wav2vec2\train.py --freeze-transformer-layers 10
```

主な出力先です。

```text
weights\best_model.pth
weights\last_model.pth
weights\labels.json
weights\wav2vec2_best\
weights\wav2vec2_last\
wav2vec2_loss.png
wav2vec2_accuracy.png
```

## 6. 評価

汎用評価スクリプトです。JSON と HTML レポートを出力します。

```powershell
uv run python script\evaluate.py
uv run python script\evaluate.py --model-type wav2vec2
uv run python script\evaluate.py --model-type whisper
```

モデル推論を使わず、`index.csv` の `predicted_text` だけで評価します。

```powershell
uv run python script\evaluate.py --from-dataset-only
```

最新モデルの予測で `index.csv` の `predicted_text` を更新してから評価します。

```powershell
uv run python script\evaluate.py --model-type cnn --update-index
```

入出力パスを指定します。

```powershell
uv run python script\evaluate.py --dataset-dir merged_dataset --output-json evaluation_results\result.json --output-html evaluation_results\report.html
```

モデル別評価スクリプトです。

```powershell
uv run python models\cnn\evaluate.py
uv run python models\cnn\evaluate.py --root-dir processed_dataset --model-path weights\best_model.pth
uv run python models\wav2vec2\evaluate.py
uv run python models\wav2vec2\evaluate.py --dataset-dir merged_dataset --model-path weights\wav2vec2_best
```

## 7. エクスポートとアップロード

CNN モデルを TorchScript に出力します。

```powershell
uv run python models\cnn\export.py
uv run python models\cnn\export.py --model-path weights\best_model.pth --output-path weights\hiragana_cnn.pt
```

Hugging Face Hub へモデルをアップロードします。`.env` に `HF_TOKEN`（Write 権限付き）が設定されている必要があります。

```powershell
uv run python script\upload_to_hf.py --type cnn
uv run python script\upload_to_hf.py --type wav2vec2
uv run python script\upload_to_hf.py --type cnn --force
```

Wav2Vec2 の ONNX エクスポートおよび INT8 量子化（ベンチマーク計測込み）を実行します。

```powershell
uv run python models\wav2vec2\export_onnx.py
```

## 8. 補助スクリプト

マイク音量とノイズフロアを測定します。`measured_audio.wav` と `Docs\charts\audio_level_measurement.png` を出力します。

```powershell
uv run python script\measure_audio_level.py
```

古い `dataset\collected` 形式を現在の PC 別フォルダ形式へ移行します。

```powershell
uv run python script\migrate_collected_dataset.py
```

Raw データの wav ファイル名を `001.wav` からの連番に直します。既存ファイル名を変更するので、実行前に対象を確認してください。

```powershell
uv run python script\rename.py
```

無音判定された wav を削除します。削除系なので、必要なら先にバックアップしてください。

```powershell
uv run python script\delete_silence_file.py
```

## 9. テストとチェック

ユニットテストを実行します。

```powershell
uv run python -m unittest discover -s tests
```

構文チェックを実行します。

```powershell
uv run python -m compileall -q main.py config.py config_labels.py core recognizers models preprocessing runtime dataset utils tests script
```

Ruff を実行します。

```powershell
uv run ruff check .
uv run ruff format . --check
```

Makefile が使える環境では次も使えます。

```powershell
make install
make lint
make check
```

## 10. よく使う一連の流れ

新しい音声を集めて CNN を再学習する最短ルートです。

```powershell
uv sync
uv run python script\collect.py rinry --target-per-label 20 --no-upload
uv run python script\merge_data.py
uv run python script\preprocess.py
uv run python train.py
uv run python script\evaluate.py --model-type cnn
uv run python main.py --model cnn
```

Wav2Vec2 まで学習する場合です。

```powershell
uv sync
uv run python script\collect.py rinry --target-per-label 20 --no-upload
uv run python train.py --model wav2vec2
uv run python script\evaluate.py --model-type wav2vec2
uv run python main.py --model wav2vec2
```

## 11. 主なディレクトリ

```text
dataset\                 Raw 音声データ
dataset\collected\       対話モードで保存された音声と metadata.csv
merged_dataset\          統合 index.csv
processed_dataset\       前処理済み学習データ
weights\                 学習済みモデル
evaluation_results\      評価結果 JSON / HTML
Docs\charts\             測定グラフなど
log                      実行ログ
```
